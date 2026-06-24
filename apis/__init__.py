from .merge import merge_paper_records
from .openalex import OpenAlexClient
from .retrieval import RetrievalClient
from .semantic_scholar import SemanticScholarClient

__all__ = [
    "SemanticScholarClient",
    "OpenAlexClient",
    "RetrievalClient",
    "merge_paper_records",
]
