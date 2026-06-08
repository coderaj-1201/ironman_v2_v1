"""
seed_data.py — Index sample documents into idx-rag for local testing.
Run once after creating the index.

Usage:
  cd <project-root>
  python scripts/seed_data.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents import SearchClient
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration, SearchField, SearchFieldDataType,
    SearchIndex, SearchableField, SemanticConfiguration, SemanticField,
    SemanticPrioritizedFields, SemanticSearch, SimpleField,
    VectorSearch, VectorSearchProfile,
)
from dotenv import load_dotenv

load_dotenv()

from shared.config import settings
from shared.azure_clients import get_openai_client

DOCS = [
    # HR
    {"id": "hr-1", "domain": "hr", "source": "HR Policy v2.3",
     "content": "Full-time employees are entitled to 20 days of annual leave per year. Leave accrues at 1.67 days per month. Up to 5 unused days can carry over."},
    {"id": "hr-2", "domain": "hr", "source": "HR Policy v2.3",
     "content": "Maternity leave is 26 weeks fully paid for employees with over 1 year of service. Paternity leave is 2 weeks paid."},
    {"id": "hr-3", "domain": "hr", "source": "HR Policy v2.3",
     "content": "Sick leave allowance is 10 days per calendar year. A medical certificate is required for absences exceeding 3 consecutive days."},
    # Legal
    {"id": "legal-1", "domain": "legal", "source": "Legal Guidelines v1.1",
     "content": "All vendor contracts above $50,000 must be reviewed by Legal before signing. NDAs must use the company standard template."},
    {"id": "legal-2", "domain": "legal", "source": "GDPR Policy v1.0",
     "content": "Under GDPR, employee personal data must be retained for no longer than 7 years after employment ends. Data must be stored in EU-approved regions."},
    # IT
    {"id": "it-1", "domain": "it", "source": "IT Handbook v3.0",
     "content": "To reset your VPN credentials, go to the IT portal at portal.company.com, click Reset VPN, and follow the steps. Contact it-help@company.com or ext. 5000 if locked out."},
    {"id": "it-2", "domain": "it", "source": "IT Handbook v3.0",
     "content": "Software installation requests must be submitted via the IT portal. VS Code, Python, and Office 365 are pre-approved. All other software requires manager and IT approval."},
]


def create_index():
    cred = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value())
    idx_client = SearchIndexClient(str(settings.AZURE_SEARCH_ENDPOINT), cred)

    fields = [
        SimpleField(name="id",     type=SearchFieldDataType.String, key=True, filterable=True),
        SearchableField(name="content", type=SearchFieldDataType.String, analyzer_name="en.microsoft"),
        SimpleField(name="source", type=SearchFieldDataType.String, filterable=True),
        SimpleField(name="domain", type=SearchFieldDataType.String, filterable=True, facetable=True),
        SimpleField(name="url",    type=SearchFieldDataType.String),
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True, vector_search_dimensions=1536,
            vector_search_profile_name="hnsw-profile",
        ),
    ]
    vector_search = VectorSearch(
        algorithms=[HnswAlgorithmConfiguration(name="hnsw")],
        profiles=[VectorSearchProfile(name="hnsw-profile", algorithm_configuration_name="hnsw")],
    )
    semantic_search = SemanticSearch(configurations=[
        SemanticConfiguration(
            name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
            prioritized_fields=SemanticPrioritizedFields(
                content_fields=[SemanticField(field_name="content")],
                keywords_fields=[SemanticField(field_name="source"), SemanticField(field_name="domain")],
            ),
        )
    ])
    index = SearchIndex(name=settings.AZURE_SEARCH_INDEX, fields=fields,
                        vector_search=vector_search, semantic_search=semantic_search)
    idx_client.create_or_update_index(index)
    print(f"✅ Index '{settings.AZURE_SEARCH_INDEX}' ready")


def embed_and_upload():
    oai = get_openai_client()
    cred = AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value())
    search_client = SearchClient(str(settings.AZURE_SEARCH_ENDPOINT), settings.AZURE_SEARCH_INDEX, cred)

    for doc in DOCS:
        print(f"  Embedding {doc['id']}...")
        doc["content_vector"] = oai.embeddings.create(
            input=doc["content"], model=settings.AZURE_OPENAI_EMBEDDING_DEPLOYMENT
        ).data[0].embedding
        doc["url"] = ""

    search_client.upload_documents(DOCS)
    print(f"✅ {len(DOCS)} documents indexed into '{settings.AZURE_SEARCH_INDEX}'")


if __name__ == "__main__":
    print("Creating index...")
    create_index()
    print("Embedding and uploading documents...")
    embed_and_upload()
    print("\n✅ Done. Run your query now.")
