"""
provision_cosmos.py
===================
Provisions Azure Cosmos DB database and all required containers for the RAG pipeline.

This script is idempotent — safe to run multiple times.
It uses the Cosmos DB Management SDK (azure-mgmt-cosmosdb) to create the account
if it doesn't exist, then the data-plane SDK to create the database + containers.

Containers created:
  - conversations  (partition: /conversation_id)  — conversation history
  - feedback       (partition: /user_id)           — user feedback on answers
  - memory         (partition: /user_id)           — long-term user memory

Prerequisites:
  - az login
  - pip install azure-mgmt-cosmosdb azure-cosmos azure-identity
  - Set AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP in .env or environment

Usage:
  cd <project-root>
  python scripts/provision_cosmos.py

  # Or with overrides:
  COSMOS_ACCOUNT_NAME=my-cosmos python scripts/provision_cosmos.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from dotenv import load_dotenv
load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
SUBSCRIPTION_ID    = os.environ.get("AZURE_SUBSCRIPTION_ID", "")
RESOURCE_GROUP     = os.environ.get("AZURE_RESOURCE_GROUP", "")
LOCATION           = os.environ.get("AZURE_LOCATION", "eastus")
COSMOS_ACCOUNT     = os.environ.get("COSMOS_ACCOUNT_NAME", "rag-enterprise-cosmos")
DATABASE_NAME      = os.environ.get("AZURE_COSMOS_DATABASE", "rag-enterprise")

CONTAINERS = [
    {
        "name":          "conversations",
        "partition_key": "/conversation_id",
        "default_ttl":   None,    # keep indefinitely
        "description":   "Conversation turns — question_id, answer_id, query, answer",
    },
    {
        "name":          "feedback",
        "partition_key": "/user_id",
        "default_ttl":   None,
        "description":   "User feedback — feedback_id, answer_id, rating, comment",
    },
    {
        "name":          "memory",
        "partition_key": "/user_id",
        "default_ttl":   -1,      # -1 = items control their own TTL via 'ttl' field
        "description":   "Long-term user memory — memory_id, content, created_at",
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
        print("   Set them in .env or export them before running this script.")
        sys.exit(1)


def provision_via_management_sdk():
    """
    Creates the Cosmos DB account using the Management SDK.
    Skips if account already exists.
    Returns the account endpoint and primary key.
    """
    try:
        from azure.identity import AzureCliCredential
        from azure.mgmt.cosmosdb import CosmosDBManagementClient
        from azure.mgmt.cosmosdb.models import (
            DatabaseAccountCreateUpdateParameters,
            Location as CosmosLocation,
        )
    except ImportError:
        print("❌ azure-mgmt-cosmosdb not installed.")
        print("   Run: pip install azure-mgmt-cosmosdb")
        sys.exit(1)

    credential = AzureCliCredential()
    mgmt_client = CosmosDBManagementClient(credential, SUBSCRIPTION_ID)

    # Check if account exists
    existing = None
    try:
        existing = mgmt_client.database_accounts.get(RESOURCE_GROUP, COSMOS_ACCOUNT)
        print(f"✅ Cosmos account '{COSMOS_ACCOUNT}' already exists — skipping creation.")
    except Exception:
        pass

    if existing is None:
        print(f"   Creating Cosmos DB account '{COSMOS_ACCOUNT}' in '{LOCATION}'...")
        print("   (This takes 3–5 minutes — please wait)")
        params = DatabaseAccountCreateUpdateParameters(
            location=LOCATION,
            locations=[CosmosLocation(location_name=LOCATION, failover_priority=0)],
            database_account_offer_type="Standard",
            kind="GlobalDocumentDB",
            capabilities=[],
            enable_automatic_failover=False,
        )
        poller = mgmt_client.database_accounts.begin_create_or_update(
            RESOURCE_GROUP, COSMOS_ACCOUNT, params
        )
        account = poller.result()   # blocks until done
        print(f"✅ Cosmos account created: {account.document_endpoint}")

    # Retrieve endpoint + primary key
    account_obj = mgmt_client.database_accounts.get(RESOURCE_GROUP, COSMOS_ACCOUNT)
    endpoint    = account_obj.document_endpoint

    keys = mgmt_client.database_accounts.list_keys(RESOURCE_GROUP, COSMOS_ACCOUNT)
    primary_key = keys.primary_master_key

    return endpoint, primary_key


def provision_database_and_containers(endpoint: str, key: str):
    """
    Creates the database and all containers using the data-plane SDK.
    Idempotent — uses create_if_not_exists throughout.
    """
    try:
        from azure.cosmos import CosmosClient, PartitionKey
    except ImportError:
        print("❌ azure-cosmos not installed.")
        print("   Run: pip install azure-cosmos")
        sys.exit(1)

    client = CosmosClient(url=endpoint, credential=key)

    print(f"\n   Creating database '{DATABASE_NAME}'...")
    db = client.create_database_if_not_exists(id=DATABASE_NAME)
    print(f"✅ Database '{DATABASE_NAME}' ready")

    for c in CONTAINERS:
        name = c["name"]
        pk   = c["partition_key"]
        print(f"   Creating container '{name}' (partition: {pk}) — {c['description']}")
        kwargs = dict(id=name, partition_key=PartitionKey(path=pk))
        if c["default_ttl"] is not None:
            kwargs["default_ttl"] = c["default_ttl"]
        db.create_container_if_not_exists(**kwargs)
        print(f"✅ Container '{name}' ready")


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

    print(f"\n── Step 1: Cosmos DB Account ──────────────────────────────")
    endpoint, key = provision_via_management_sdk()

    print(f"\n── Step 2: Database + Containers ──────────────────────────")
    provision_database_and_containers(endpoint, key)

    print_env_snippet(endpoint, key)
    print("\n✅ Provisioning complete.")


if __name__ == "__main__":
    main()
