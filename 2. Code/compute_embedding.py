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
    """
    Initialize and return an OpenAI client configured for Ollama.
    """
    return OpenAI(
        base_url=config.ollama_base_url,
        api_key="ollama",
    )


def compute_embedding(text: str, embedding_model: OpenAI = None) -> List[float]:
    """
    Compute embedding for a single text string using OpenAI client with Ollama.

    Args:
        text (str): Text to embed.
        embedding_model (OpenAI): Preinitialized OpenAI client configured for Ollama. If None, will use the default model.

    Returns:
        List[float]: The embedding vector.
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
    """
    Compute embeddings for specified text columns in a DataFrame.

    Args:
        embedding_model (OpenAI): Preinitialized OpenAI client configured for Ollama.
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
