import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """Enhanced configuration for LlamaIndex features"""

    # Neo4j settings
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_username: str = os.getenv("NEO4J_USERNAME", "")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "")

    # Ollama settings
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_temperature: float = float(os.getenv("OLLAMA_TEMPERATURE", "0"))
    ollama_max_tokens: int = int(os.getenv("OLLAMA_MAX_TOKENS", "2048"))


    
    
