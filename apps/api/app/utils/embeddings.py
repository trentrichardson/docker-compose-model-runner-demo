from sentence_transformers import SentenceTransformer
import logging

# ─── CONFIGURE LOGGER ───────────────────────────────────────────

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ─── LOAD MODEL ONCE AT MODULE LEVEL ────────────────────────────
#
# Model                                    Embedding Dim   Notes
# sentence-transformers/all-MiniLM-L6-v2   384             Fast, strong baseline
# BAAI/bge-small-en-v1.5                   768             Very good local model
# intfloat/e5-large-v2                     1024            Larger, strong multilingual
# intfloat/multilingual-e5-large           1536            Bigger, matches OpenAI’s dim

try:
    model_name = "BAAI/bge-small-en-v1.5"  # 384-dim model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load embedding model: {e}")
    raise

# ─── FUNCTION TO CONVERT TEXT TO EMBEDDING ─────────────────

def generate_embedding_from_text(text: str) -> list[float]:
    embedding = model.encode(text)
    return embedding.tolist()


# ─── FUNCTION TO CONVERT FILE BYTES TO EMBEDDING ─────────────────

def generate_embedding_from_file(file_bytes: bytes) -> list:
    """
    Converts file bytes to text and generates an embedding.

    Args:
        file_bytes (bytes): The content of the uploaded file.

    Returns:
        list: The embedding vector as a list of floats.
    """
    try:
        text = file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("Non-UTF-8 file detected — using fallback decode with errors='ignore'")
        text = file_bytes.decode("utf-8", errors="ignore")

    if not text.strip():
        raise ValueError("File content is empty or could not be decoded to text.")

    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()
