"""
Benchmark queries, intent keywords, and evaluation datasets for testing RAG performance.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkQuery:
    """Represents a standardized evaluation query with intent keywords."""
    query: str
    intent_keywords: List[str]


DEFAULT_BENCHMARK_SUITE: List[BenchmarkQuery] = [
    BenchmarkQuery(
        query="What is Hibernate and what does it provide?",
        intent_keywords=["hibernate", "orm", "mapping", "java", "relational", "database"]
    ),
    BenchmarkQuery(
        query="What is SessionFactory in Hibernate?",
        intent_keywords=["sessionfactory", "session", "thread-safe", "configuration", "mapping"]
    ),
    BenchmarkQuery(
        query="How does Object-Relational Mapping (ORM) work?",
        intent_keywords=["mapping", "object", "relational", "tables", "domain", "persistence"]
    ),
    BenchmarkQuery(
        query="What is the role of the Session interface in Hibernate?",
        intent_keywords=["session", "transaction", "save", "crud", "persistence", "connection"]
    ),
    BenchmarkQuery(
        query="What is JDBC and how does Hibernate interact with it?",
        intent_keywords=["jdbc", "driver", "sql", "connection", "query", "underlying"]
    )
]
