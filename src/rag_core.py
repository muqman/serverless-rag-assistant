"""Shared helpers: split text into chunks, create embeddings, find the closest chunks."""
import json
import os

import boto3

EMBED_MODEL_ID = os.environ.get("EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
EMBED_DIMENSIONS = 512
bedrock = boto3.client("bedrock-runtime")


def chunk_pages(pages, size=800, overlap=150):
    """pages = [(page_number, text), ...]. Returns overlapping chunks that remember their page."""
    chunks = []
    for page_no, text in pages:
        text = " ".join(text.split())  # collapse runs of whitespace
        start = 0
        while start < len(text):
            end = min(start + size, len(text))
            if end < len(text):  # end at a space, not in the middle of a word
                space = text.rfind(" ", start, end)
                if space > start + size // 2:
                    end = space
            piece = text[start:end].strip()
            if piece:
                chunks.append({"page": page_no, "text": piece})
            if end == len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def embed(text):
    """Turn text into a 512-number vector with Amazon Titan Text Embeddings V2."""
    body = json.dumps({"inputText": text, "dimensions": EMBED_DIMENSIONS, "normalize": True})
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID, body=body, contentType="application/json", accept="application/json"
    )
    return json.loads(resp["body"].read())["embedding"]


def top_k(query_vec, chunks, k=4):
    """Rank chunks by cosine similarity. The vectors are normalized, so a dot product is enough."""
    scored = [(sum(a * b for a, b in zip(query_vec, c["embedding"])), c) for c in chunks]
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[:k]
