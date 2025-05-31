import logging
from pathlib import Path
from typing import List

import pandas as pd
from llama_index.embeddings.ollama import OllamaEmbedding

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def get_embedding_model(config: Config) -> OllamaEmbedding:
    """
    Initialize and return an OllamaEmbedding model instance.
    """
    return OllamaEmbedding(
        model_name=config.ollama_embed_model,
        base_url=config.ollama_base_url,
    )


def compute_embedding(
    text: str, embedding_model: OllamaEmbedding = None
) -> List[float]:
    """
    Compute embedding for a single text string.

    Args:
        text (str): Text to embed.
        embedding_model (OllamaEmbedding): Preinitialized Ollama embedding model. If None, will use the default model.

    Returns:
        List[float]: The embedding vector.
    """
    if embedding_model is None:
        embedding_model = get_embedding_model(Config())
    raw = embedding_model.get_text_embedding(text)
    # If raw is already a list of floats, return as-is
    if isinstance(raw, list):
        return raw
    # Otherwise try to convert
    try:
        return raw.tolist()
    except AttributeError:
        logger.error("Unexpected embedding format: %s", type(raw))
        raise


def compute_embeddings_from_dataframe(
    embedding_model: OllamaEmbedding,
    df: pd.DataFrame,
    text_columns: List[str],
) -> pd.DataFrame:
    """
    Compute embeddings for specified text columns in a DataFrame.

    Args:
        embedding_model (OllamaEmbedding): Preinitialized Ollama embedding model.
        df (pd.DataFrame): DataFrame with text data.
        text_columns (List[str]): Columns to combine for embedding.

    Returns:
        pd.DataFrame: Original DataFrame with new 'embedding' column.
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
