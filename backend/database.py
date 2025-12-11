# =============================================================================
# Database module for AI RAG Assistant
# =============================================================================
# This file handles ALL database operations. The rest of the app doesn't
# need to know anything about DuckDB or embeddings—it just calls db.query()
# =============================================================================

import duckdb
import os
from sentence_transformers import SentenceTransformer
import streamlit as st

# Import config final with embedding name and dimensions 
from config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_NAME, DEFAULT_TOP_K

class RAGDatabase:

    # TO DO: Init class with db_path and model attributes
    def __init__(self, db_path: str):
        # STORE PATH
        self.db_path = db_path
        # LOAD MODEL METHOD
        self.model = self._load_model()
    # @st.cache_resource because Streamlit reruns entire script on ever interaction
    # @st.cache_resource stores information throughout session
    # @staticmethod caches info because functions can't use 'self'
    @staticmethod
    @st.cache_resource

    # TO DO: CREATE _load_model() method that stores model attribute
    def _load_model():
        return SentenceTransformer(EMBEDDING_MODEL_NAME)

    def test_connection(self) -> bool:
        """Test if the database can be connected to."""
        # First check: does the file even exist?
        if not os.path.exists(self.db_path):
            return False
        # Second check: can DuckDB actually open it?
        try:
            conn = duckdb.connect(self.db_path, read_only=True)
            conn.close()
            return True
        except Exception:
            return False

    # TO DO: Update query() method
    def query(self, query_text: str, top_k: int = DEFAULT_TOP_K) -> list[dict]:
        """
        Query the AI RAG database for relevant passages about artificial intelligence.
        
        Args:
            query_text: The search query (about AI uses, risks, ethics, black-box issues, etc.).
            top_k: Number of results to return.
            
        Returns:
            List of dictionaries containing 'text' and 'similarity'.
        """
        try:
            # TO DO: Connect to database in read-only mode
            # read_only=True prevents accidental modifications
            conn = duckdb.connect(self.db_path, read_only = True)
            
            # TO DO: Convert query text to embedding vector
            query_embedding = self.model.encode(query_text).tolist()
            
            # TO DO: Update table name
            # Execute vector search over stored AI document chunks
            # Return top k most similar passages
            # Note: We cast the parameter to FLOAT[384] to match the embedding dimension
            results = conn.execute(f"""
                SELECT text, array_cosine_similarity(embedding, ?::FLOAT[{EMBEDDING_DIMENSION}]) as similarity
                FROM ai_rag_documents
                ORDER BY similarity DESC
                LIMIT ?
            """, [query_embedding, top_k]).fetchall()
            
            # TO DO: Close database connection
            conn.close()
            
            # TO DO: Format results for the agent
            # Each result row is (text, similarity_score)
            return ({"text":row[0], "similarity":float(row[1])} for row in results)
            
        except Exception as e:
            raise Exception(f"Database query failed: {str(e)}")
