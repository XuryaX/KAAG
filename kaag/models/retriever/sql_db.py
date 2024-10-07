from typing import Dict, Any
from .base import BaseRetriever
import sqlalchemy

class SQLDBRetriever(BaseRetriever):
    """
    Retriever implementation for SQL databases.
    """

    def __init__(self, connection_string: str):
        self.engine = sqlalchemy.create_engine(connection_string)

    def retrieve(self, query: str) -> Dict[str, Any]:
        with self.engine.connect() as connection:
            result = connection.execute(sqlalchemy.text(query))
            return {"results": [dict(row) for row in result]}