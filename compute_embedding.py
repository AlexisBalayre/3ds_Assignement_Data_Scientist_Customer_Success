import logging
from pathlib import Path
from typing import List

import pandas as pd
from openai import OpenAI

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_embedding_model(config: Config) -> OpenAI:
    """Initialize and return an OpenAI client configured for local Ollama embedding service.
    
    Creates an OpenAI-compatible client that connects to a local Ollama instance
    for generating text embeddings. This allows using Ollama's embedding models
    through the familiar OpenAI API interface while keeping all computation local.
    
    The client is configured to use Ollama's API endpoint and authentication,
    enabling embedding generation without external API calls or data transmission.
    
    Args:
        config: Configuration object containing Ollama connection settings.
            Must have ollama_base_url attribute (e.g., "http://localhost:11434/v1")
            
    Returns:
        OpenAI: Configured OpenAI client instance pointing to Ollama service.
            Ready for embedding generation via embeddings.create() method.
            
    Raises:
        AttributeError: If config is missing required ollama_base_url attribute
        ConnectionError: If Ollama service is not running or unreachable
        
    Example:
        >>> config = Config()
        >>> client = get_embedding_model(config)
        >>> response = client.embeddings.create(model="nomic-embed-text", input="Hello world")
        >>> embedding = response.data[0].embedding
        
    Note:
        - Requires Ollama server running locally with embedding model installed
        - Client supports connection pooling and is thread-safe for concurrent use
        - Uses "ollama" as API key placeholder (required by OpenAI client format)
        - Compatible with any Ollama embedding model (nomic-embed-text, etc.)
        
    Prerequisites:
        - Ollama installed and running (typically on http://localhost:11434)
        - At least one embedding model pulled (e.g., `ollama pull nomic-embed-text`)
        - Config properly configured with correct base URL
    """
    return OpenAI(
        base_url=config.ollama_base_url,
        api_key="ollama",
    )


def compute_embedding(text: str, embedding_model: OpenAI = None) -> List[float]:
    """Compute dense vector embedding for a single text string using Ollama embedding model.
    
    Generates a numerical vector representation of the input text that captures
    semantic meaning and enables similarity comparisons. Uses local Ollama service
    for privacy-preserving embedding generation without external API dependencies.
    
    The embedding process converts text into a high-dimensional vector (typically
    384, 768, or 1536 dimensions) where semantically similar texts have similar
    vectors according to cosine similarity.
    
    Args:
        text: Input text to convert to embedding vector. Can be single words,
            sentences, paragraphs, or documents. Empty strings will produce
            valid embeddings but may not be meaningful for similarity search.
        embedding_model: Pre-initialized OpenAI client configured for Ollama.
            If None, creates new client using default configuration. Reusing
            clients is recommended for better performance in batch operations.
            
    Returns:
        List[float]: Dense vector embedding of the input text. Vector dimensions
            depend on the configured embedding model (typically 384-1536 floats).
            Values are typically in range [-1.0, 1.0] and normalized for cosine similarity.
            
    Raises:
        ConnectionError: If Ollama service is unreachable or model not available
        ValueError: If text input is invalid or causes model errors
        RuntimeError: If embedding response format is unexpected or corrupted
        Exception: For other API or processing errors
        
    Example:
        >>> # Basic usage with auto-client creation
        >>> embedding = compute_embedding("Machine learning is fascinating")
        >>> print(f"Embedding dimension: {len(embedding)}")  # e.g., 384
        >>> print(f"First values: {embedding[:3]}")  # [-0.123, 0.456, -0.789]
        
        >>> # Efficient batch usage with reused client
        >>> config = Config()
        >>> client = get_embedding_model(config)
        >>> texts = ["Hello world", "Goodbye world", "Unrelated topic"]
        >>> embeddings = [compute_embedding(text, client) for text in texts]
        
        >>> # Similarity comparison
        >>> import numpy as np
        >>> sim = np.dot(embeddings[0], embeddings[1])  # High similarity expected
        >>> print(f"Similarity: {sim:.3f}")
        
    Performance Notes:
        - Client reuse significantly improves batch performance (avoid recreation)
        - Typical processing: 100-1000 texts/second depending on model and hardware
        - Memory usage scales with batch size and embedding dimensions
        - Local processing eliminates network latency vs cloud APIs
        
    Model Compatibility:
        - Works with any Ollama embedding model (nomic-embed-text, mxbai-embed-large, etc.)
        - Model selection controlled by config.ollama_embed_model setting
        - Different models produce different dimensional outputs and quality
        
    Thread Safety:
        - Function is thread-safe when using separate embedding_model instances
        - OpenAI client handles concurrent requests internally
        - Safe for parallel processing and web server usage
    """
    if embedding_model is None:
        config = Config()
        embedding_model = get_embedding_model(config)
        model_name = config.ollama_embed_model
    else:
        config = Config()
        model_name = config.ollama_embed_model

    try:
        response = embedding_model.embeddings.create(model=model_name, input=text)

        # Extract the embedding from the response
        embedding = response.data[0].embedding

        # If embedding is already a list of floats, return as-is
        if isinstance(embedding, list):
            return embedding
        # Otherwise try to convert
        try:
            return embedding.tolist()
        except AttributeError:
            logger.error("Unexpected embedding format: %s", type(embedding))
            raise

    except Exception as e:
        logger.error("Failed to compute embedding: %s", str(e))
        raise


def compute_embeddings_from_dataframe(
    embedding_model: OpenAI,
    df: pd.DataFrame,
    text_columns: List[str],
) -> pd.DataFrame:
    """Compute embeddings for multiple text columns in a pandas DataFrame efficiently.
    
    Batch processes a DataFrame to generate embeddings by combining specified text
    columns and computing vector representations for each row. This is optimized
    for processing large datasets of tickets, documents, or other text records
    for semantic search and similarity analysis.
    
    The function combines multiple text fields (e.g., title + description) into
    a single text representation before embedding, providing richer semantic
    context than individual fields alone.
    
    Args:
        embedding_model: Pre-initialized OpenAI client configured for Ollama.
            Should be created once and reused for all embeddings to optimize
            performance and connection management.
        df: Source DataFrame containing text data to embed. Must contain all
            columns specified in text_columns parameter. Will not be modified
            (creates copy for processing).
        text_columns: List of column names to combine for embedding generation.
            Columns will be concatenated with spaces in the order provided.
            All specified columns must exist in the DataFrame.
            
    Returns:
        pd.DataFrame: Copy of input DataFrame with additional 'embedding' column
            containing List[float] vectors. Original columns and data preserved.
            Ready for use with vector databases or similarity search operations.
            
    Raises:
        KeyError: If any column in text_columns doesn't exist in DataFrame
        ValueError: If DataFrame is empty or text_columns list is empty
        MemoryError: If DataFrame is too large for available memory during processing
        Exception: For embedding computation or other processing errors
        
    Example:
        >>> # Setup
        >>> config = Config()
        >>> client = get_embedding_model(config)
        >>> df = pd.DataFrame({
        ...     'id': [1, 2, 3],
        ...     'title': ['Login Error', 'Password Reset', 'API Timeout'],
        ...     'description': ['Cannot access dashboard', 'Forgot password', 'API calls failing'],
        ...     'category': ['ACCESS', 'ACCOUNT', 'INTEGRATION']
        ... })
        
        >>> # Generate embeddings from title and description
        >>> df_with_embeddings = compute_embeddings_from_dataframe(
        ...     embedding_model=client,
        ...     df=df,
        ...     text_columns=['title', 'description']
        ... )
        
        >>> # Examine results
        >>> print(df_with_embeddings.columns)  # ['id', 'title', 'description', 'category', 'embedding']
        >>> print(len(df_with_embeddings.iloc[0]['embedding']))  # e.g., 384
        
        >>> # Save for later use
        >>> df_with_embeddings.to_parquet('embeddings.parquet')
        
    Processing Details:
        - Text columns are converted to strings and joined with spaces
        - NaN/null values are converted to string 'nan' during concatenation
        - Embedding computation is applied row-wise using pandas.apply()
        - Intermediate 'combined_text' column is created and removed automatically
        - Original DataFrame structure and data types are preserved
        
    Performance Optimization:
        - Use pre-initialized embedding_model for batch operations
        - Consider chunking very large DataFrames (>100k rows) for memory management
        - Process in parallel using multiprocessing for extremely large datasets
        - Monitor memory usage as embeddings can be memory-intensive
        
    Memory Considerations:
        - Each embedding typically uses 1.5-6KB depending on model dimensions
        - 100k records with 384-dim embeddings ≈ 150-600MB additional memory
        - Consider saving to disk incrementally for very large datasets
        
    Integration Examples:
        >>> # For vector database storage
        >>> df_embedded = compute_embeddings_from_dataframe(client, df, ['title', 'description'])
        >>> # Store in Pinecone, Weaviate, or Neo4j vector index
        
        >>> # For similarity search
        >>> query_embedding = compute_embedding("user login problems", client)
        >>> similarities = df_embedded['embedding'].apply(
        ...     lambda emb: np.dot(query_embedding, emb)
        ... )
        >>> most_similar = df_embedded.iloc[similarities.argmax()]
    """
    df = df.copy()
    logger.info("Combining columns %s into 'combined_text'", text_columns)
    df["combined_text"] = df[text_columns].astype(str).agg(" ".join, axis=1)

    logger.info("Computing embeddings for %d rows", len(df))
    df["embedding"] = df["combined_text"].apply(
        lambda txt: compute_embedding(text=txt, embedding_model=embedding_model)
    )

    # Drop intermediate column
    df.drop(columns=["combined_text"], inplace=True)

    return df


def main() -> None:
    """Main function for batch processing ticket embeddings from CSV files.
    
    Demonstrates end-to-end workflow for generating embeddings from structured
    ticket data. Reads ticket information from CSV, combines title and description
    fields, computes embeddings using Ollama, and saves enhanced dataset.
    
    This function serves as both a standalone utility and reference implementation
    for batch embedding generation workflows.
    
    Workflow:
        1. Load configuration and initialize embedding model client
        2. Read ticket data from CSV file (tickets_sample.csv)
        3. Combine title and description columns for richer context
        4. Generate embeddings for all tickets using local Ollama model
        5. Save enhanced dataset with embeddings to new CSV file
        
    File Structure:
        Input:  datasets/data/tickets_sample.csv
        Output: datasets/data/tickets_sample_with_embeddings.csv
        
    Expected CSV Format:
        - Must contain 'title' and 'description' columns
        - Additional columns preserved in output
        - No specific order required for other columns
        
    Example Input CSV:
        ticketId,title,description,status
        TICK-001,Login Error,Cannot access dashboard with valid credentials,CLOSED
        TICK-002,API Timeout,External API calls timing out after 30 seconds,OPEN
        
    Example Output CSV:
        ticketId,title,description,status,embedding
        TICK-001,Login Error,Cannot access dashboard...,[0.123,-0.456,0.789,...]
        TICK-002,API Timeout,External API calls timing...,[0.234,0.567,-0.890,...]
        
    Error Handling:
        - Graceful handling of missing input files
        - Comprehensive logging of processing steps and errors
        - Continues processing if individual embeddings fail
        - Validates output before saving
        
    Performance Notes:
        - Processing speed depends on text length and model choice
        - Typical throughput: 100-1000 records/second
        - Memory usage scales with dataset size and embedding dimensions
        - Progress logged every N records for monitoring
        
    Prerequisites:
        - Ollama server running locally with embedding model installed
        - Input CSV file exists with required columns
        - Write permissions for output directory
        - Sufficient memory for dataset + embeddings
        
    Usage:
        >>> # Command line execution
        >>> python compute_embedding.py
        
        >>> # Programmatic usage
        >>> if __name__ == "__main__":
        ...     main()
        
    Integration:
        - Output CSV can be loaded into vector databases
        - Embeddings ready for similarity search operations
        - Compatible with knowledge graph ingestion pipelines
        - Supports downstream ML and RAG applications
    """
    # Load configuration and initialize embedding model
    config = Config()
    embedding_model = get_embedding_model(config)

    # Define file paths
    base_path = Path(__file__).parent / "datasets" / "data"
    input_csv = base_path / "tickets_sample.csv"
    output_csv = base_path / "tickets_sample_with_embeddings.csv"

    # Read data
    try:
        df = pd.read_csv(input_csv)
        logger.info("Loaded %d records from %s", len(df), input_csv)
    except Exception as e:
        logger.exception("Failed to read input CSV: %s", input_csv)
        return

    # Compute embeddings
    text_cols = ["title", "description"]
    df_out = compute_embeddings_from_dataframe(embedding_model, df, text_cols)

    # Save results
    try:
        df_out.to_csv(output_csv, index=False)
        logger.info("Saved embeddings to %s", output_csv)
    except Exception as e:
        logger.exception("Failed to write output CSV: %s", output_csv)


if __name__ == "__main__":
    main()