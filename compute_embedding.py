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

    The client is configured to use Ollama's API endpoint and authentication,
    enabling embedding generation without external API calls or data transmission.

    Args:
        config: Configuration object containing Ollama connection settings.
            Must have ollama_base_url attribute (e.g., "http://localhost:11434/v1")

    Returns:
        OpenAI: Configured OpenAI client instance pointing to Ollama service.
            Ready for embedding generation via embeddings.create() method.

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

    Args:
        text: Input text to convert to embedding vector.
        embedding_model: Pre-initialized OpenAI client configured for Ollama.
            If None, creates new client using default configuration. Reusing
            clients is recommended for better performance in batch operations.

    Returns:
        List[float]: Dense vector embedding of the input text. Vector dimensions
            depend on the configured embedding model (typically 384-1536 floats).
            Values are typically in range [-1.0, 1.0] and normalized for cosine similarity.
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

    The function combines multiple text fields (e.g., title + description) into
    a single text representation before embedding, providing richer semantic
    context than individual fields alone.

    Args:
        embedding_model: Pre-initialized OpenAI client configured for Ollama.
        df: Source DataFrame containing text data to embed.
        text_columns: List of column names to combine for embedding generation.

    Returns:
        pd.DataFrame: Copy of input DataFrame with additional 'embedding' column
            containing List[float] vectors
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

    This function serves as both a standalone utility and reference implementation
    for batch embedding generation workflows.

    Prerequisites:
        - Ollama server running locally with embedding model installed
        - Input CSV file exists with required columns
        - Write permissions for output directory
        - Sufficient memory for dataset + embeddings
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
