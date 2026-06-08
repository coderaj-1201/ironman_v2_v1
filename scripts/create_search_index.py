"""
create_search_index.py
======================
Creates/updates the single unified AI Search index (idx-rag).

THIS IS THE SOURCE OF TRUTH FOR THE INDEX SCHEMA.
Run this from either the ingestion-pipeline or rag-enterprise repo.
Both pipelines share the same index.

Schema additions vs original:
  parent_id, chunk_type, doc_name, doc_url, file_type, blob_path,
  ingested_at, page_number, title, section_heading, section_subheading,
  table_raw, is_deleted

Semantic config prioritises: content → section_heading → title → section_subheading

Run:
  cd <project-root>
  python scripts/create_search_index.py

Auth: az login (needs Search Index Data Contributor role)
"""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from azure.core.credentials import AzureKeyCredential
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    HnswAlgorithmConfiguration,
    SearchField,
    SearchFieldDataType,
    SearchIndex,
    SearchableField,
    SemanticConfiguration,
    SemanticField,
    SemanticPrioritizedFields,
    SemanticSearch,
    SimpleField,
    VectorSearch,
    VectorSearchProfile,
)

from shared.config import settings


def create_index():
    client = SearchIndexClient(
        endpoint=str(settings.AZURE_SEARCH_ENDPOINT),
        credential=AzureKeyCredential(settings.AZURE_SEARCH_API_KEY.get_secret_value()),
    )

    fields = [
        # ── Identity ──────────────────────────────────────────────────────────
        SimpleField(name="id",
                    type=SearchFieldDataType.String,
                    key=True, filterable=True),

        SimpleField(name="parent_id",
                    type=SearchFieldDataType.String,
                    filterable=True),

        SimpleField(name="chunk_type",
                    type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
                    # values: title | heading | paragraph | table

        # ── Document provenance ───────────────────────────────────────────────
        SimpleField(name="domain",
                    type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
                    # values: hr | legal | it

        SimpleField(name="doc_name",
                    type=SearchFieldDataType.String,
                    filterable=True, facetable=True),

        SimpleField(name="source",
                    type=SearchFieldDataType.String,
                    filterable=True),
                    # alias for doc_name — used by retrieval pipeline

        SimpleField(name="doc_url",
                    type=SearchFieldDataType.String),

        SimpleField(name="file_type",
                    type=SearchFieldDataType.String,
                    filterable=True, facetable=True),
                    # values: pdf | docx | xlsx | pptx

        SimpleField(name="blob_path",
                    type=SearchFieldDataType.String),

        SimpleField(name="ingested_at",
                    type=SearchFieldDataType.String,
                    filterable=True, sortable=True),

        # ── Position ──────────────────────────────────────────────────────────
        SimpleField(name="page_number",
                    type=SearchFieldDataType.Int32,
                    filterable=True, sortable=True),

        # ── Structure metadata — used by semantic ranker as keywords ──────────
        SearchableField(name="title",
                        type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),

        SearchableField(name="section_heading",
                        type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),

        SearchableField(name="section_subheading",
                        type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),

        # ── Content ───────────────────────────────────────────────────────────
        SearchableField(name="content",
                        type=SearchFieldDataType.String,
                        analyzer_name="en.microsoft"),
                        # Main searchable text — NL summary for tables

        SimpleField(name="table_raw",
                    type=SearchFieldDataType.String),
                    # Original markdown table — NOT searchable, returned to LLM

        # ── Lifecycle ─────────────────────────────────────────────────────────
        SimpleField(name="is_deleted",
                    type=SearchFieldDataType.Boolean,
                    filterable=True),

        # ── Vector ────────────────────────────────────────────────────────────
        SearchField(
            name="content_vector",
            type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
            searchable=True,
            vector_search_dimensions=1536,      # ada-002
            vector_search_profile_name="hnsw-profile",
        ),
    ]

    vector_search = VectorSearch(
        algorithms=[
            HnswAlgorithmConfiguration(
                name="hnsw",
                parameters={
                    "m": 4,
                    "efConstruction": 400,
                    "efSearch": 500,
                    "metric": "cosine",
                },
            )
        ],
        profiles=[
            VectorSearchProfile(
                name="hnsw-profile",
                algorithm_configuration_name="hnsw",
            )
        ],
    )

    # Semantic ranker: content is primary, headings boost precision
    semantic_search = SemanticSearch(
        configurations=[
            SemanticConfiguration(
                name=settings.AZURE_SEARCH_SEMANTIC_CONFIG,
                prioritized_fields=SemanticPrioritizedFields(
                    content_fields=[
                        SemanticField(field_name="content"),
                    ],
                    title_field=SemanticField(field_name="title"),
                    keywords_fields=[
                        SemanticField(field_name="section_heading"),
                        SemanticField(field_name="section_subheading"),
                        SemanticField(field_name="domain"),
                        SemanticField(field_name="doc_name"),
                    ],
                ),
            )
        ]
    )

    index = SearchIndex(
        name=settings.AZURE_SEARCH_INDEX,
        fields=fields,
        vector_search=vector_search,
        semantic_search=semantic_search,
    )

    result = client.create_or_update_index(index)
    print(f"\n✅ Index '{result.name}' created/updated")
    print(f"   Endpoint : {settings.AZURE_SEARCH_ENDPOINT}")
    print(f"   Fields   : {[f.name for f in result.fields]}")
    print(f"   Semantic : {settings.AZURE_SEARCH_SEMANTIC_CONFIG}")


if __name__ == "__main__":
    create_index()
