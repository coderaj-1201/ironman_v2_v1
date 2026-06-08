"""
provision_cosmos.py
===================
Provisions Azure Cosmos DB database, containers, and composite indexes.

This script is idempotent — safe to run multiple times.

Containers created:
  conversations  (partition: /conversation_id)
    composite index: conversation_id ASC + timestamp DESC
  feedback       (partition: /user_id)
  memory         (partition: /user_id)
    composite index: user_id ASC + created_at DESC

Without composite indexes, ORDER BY queries in conversation_store and
memory_store will fail with Cosmos 400 errors.

Prerequisites:
  - az login
  - pip install azure-mgmt-cosmosdb azure-cosmos azure-identity
  - AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP in .env or environment

Usage:
  cd <project-root>
  python scripts/provision_cosmos.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

SUBSCRIPTION_ID = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP  = os.environ.get("AZURE_RESOURCE_GROUP", "")
LOCATION        = os.environ.get("AZURE_LOCATION", "eastus")
COSMOS_ACCOUNT  = os.environ.get("COSMOS_ACCOUNT_NAME", "rag-enterprise-cosmos")
DATABASE_NAME   = os.environ.get("AZURE_COSMOS_DATABASE", "rag-enterprise")

CONTAINERS = [
    {
        "name":          "conversations",
        "partition_key": "/conversation_id",
        "default_ttl":   None,
        "description":   "Q&A turns — question_id, answer_id, query, answer, confidence",
        # Needed for ORDER BY timestamp in get_history()
        "composite_indexes": [
            [
                {"path": "/conversation_id", "order": "ascending"},
                {"path": "/timestamp",        "order": "descending"},
            ]
        ],
    },
    {
        "name":          "feedback",
        "partition_key": "/user_id",
        "default_ttl":   None,
        "description":   "User feedback — feedback_id, answer_id, rating, comment",
        "composite_indexes": [],
    },
    {
        "name":          "memory",
        "partition_key": "/user_id",
        "default_ttl":   -1,
        "description":   "Long-term user memory — memory_id, content, created_at",
        # Needed for ORDER BY created_at in get_long_term()
        "composite_indexes": [
            [
                {"path": "/user_id",    "order": "ascending"},
                {"path": "/created_at", "order": "descending"},
            ]
        ],
    },
]


def _check_prereqs():
    missing = []
    if not SUBSCRIPTION_ID:
        missing.append("AZURE_SUBSCRIPTION_ID")
    if not RESOURCE_GROUP:
        missing.append("AZURE_RESOURCE_GROUP")
    if missing:
        print(f"❌ Missing required env vars: {', '.join(missing)}")
        sys.exit(1)


def provision_account() -> tuple[str, str]:
    """Creates the Cosmos DB account if needed. Returns (endpoint, primary_key)."""
    try:
        from azure.identity import AzureCliCredential
        from azure.mgmt.cosmosdb import CosmosDBManagementClient
        from azure.mgmt.cosmosdb.models import (
            DatabaseAccountCreateUpdateParameters,
            Location as CosmosLocation,
        )
    except ImportError:
        print("❌ azure-mgmt-cosmosdb not installed. Run: pip install azure-mgmt-cosmosdb")
        sys.exit(1)

    credential   = AzureCliCredential()
    mgmt_client  = CosmosDBManagementClient(credential, SUBSCRIPTION_ID)

    try:
        mgmt_client.database_accounts.get(RESOURCE_GROUP, COSMOS_ACCOUNT)
        print(f"✅ Account '{COSMOS_ACCOUNT}' already exists.")
    except Exception:
        print(f"   Creating account '{COSMOS_ACCOUNT}' in '{LOCATION}' (3-5 min)...")
        params = DatabaseAccountCreateUpdateParameters(
            location=LOCATION,
            locations=[CosmosLocation(location_name=LOCATION, failover_priority=0)],
            database_account_offer_type="Standard",
            kind="GlobalDocumentDB",
        )
        mgmt_client.database_accounts.begin_create_or_update(
            RESOURCE_GROUP, COSMOS_ACCOUNT, params
        ).result()

    account   = mgmt_client.database_accounts.get(RESOURCE_GROUP, COSMOS_ACCOUNT)
    endpoint  = account.document_endpoint
    keys      = mgmt_client.database_accounts.list_keys(RESOURCE_GROUP, COSMOS_ACCOUNT)
    return endpoint, keys.primary_master_key


def provision_database_and_containers(endpoint: str, key: str):
    try:
        from azure.cosmos import CosmosClient, PartitionKey
    except ImportError:
        print("❌ azure-cosmos not installed. Run: pip install azure-cosmos")
        sys.exit(1)

    client = CosmosClient(url=endpoint, credential=key)

    print(f"\n   Creating database '{DATABASE_NAME}'...")
    db = client.create_database_if_not_exists(id=DATABASE_NAME)
    print(f"✅ Database ready")

    for c in CONTAINERS:
        name = c["name"]
        pk   = c["partition_key"]
        print(f"\n   Container '{name}' (partition: {pk})")

        kwargs: dict = dict(id=name, partition_key=PartitionKey(path=pk))
        if c["default_ttl"] is not None:
            kwargs["default_ttl"] = c["default_ttl"]

        # Set indexing policy with composite indexes if needed
        if c["composite_indexes"]:
            kwargs["indexing_policy"] = {
                "indexingMode": "consistent",
                "automatic": True,
                "includedPaths": [{"path": "/*"}],
                "excludedPaths": [{"path": '/"_etag"/?'}],
                "compositeIndexes": c["composite_indexes"],
            }

        container = db.create_container_if_not_exists(**kwargs)

        # If container already existed, patch the indexing policy to add composite indexes
        if c["composite_indexes"]:
            try:
                props = container.read()
                existing_policy = props.get("indexingPolicy", {})
                existing_ci     = existing_policy.get("compositeIndexes", [])
                if not existing_ci:
                    print(f"   Patching composite index on '{name}'...")
                    existing_policy["compositeIndexes"] = c["composite_indexes"]
                    container.replace_container(
                        partition_key=PartitionKey(path=pk),
                        indexing_policy=existing_policy,
                    )
                    print(f"✅ Composite index set on '{name}'")
                else:
                    print(f"✅ Composite index already present on '{name}'")
            except Exception as exc:
                print(f"⚠️  Could not patch composite index on '{name}': {exc}")
        else:
            print(f"✅ '{name}' ready")


def print_env_snippet(endpoint: str, key: str):
    print("\n" + "=" * 60)
    print("Add these to your .env file:")
    print("=" * 60)
    print(f"AZURE_COSMOS_ENDPOINT={endpoint}")
    print(f"AZURE_COSMOS_KEY={key}")
    print(f"AZURE_COSMOS_DATABASE={DATABASE_NAME}")
    print(f"AZURE_COSMOS_CONTAINER_CONVERSATIONS=conversations")
    print(f"AZURE_COSMOS_CONTAINER_FEEDBACK=feedback")
    print(f"AZURE_COSMOS_CONTAINER_MEMORY=memory")
    print("=" * 60)


def main():
    print("=" * 60)
    print("RAG Enterprise — Cosmos DB Provisioning")
    print("=" * 60)
    _check_prereqs()

    print("\n── Step 1: Account ────────────────────────────────────────")
    endpoint, key = provision_account()

    print("\n── Step 2: Database + Containers + Indexes ────────────────")
    provision_database_and_containers(endpoint, key)

    print_env_snippet(endpoint, key)
    print("\n✅ Provisioning complete.")


if __name__ == "__main__":
    main()
