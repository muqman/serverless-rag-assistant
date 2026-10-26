"""Runs when a PDF lands in uploads/: extract text, chunk it, embed it, save the index."""
import io
import json
import urllib.parse

import boto3
from pypdf import PdfReader

from rag_core import chunk_pages, embed

s3 = boto3.client("s3")


def handler(event, context):
    for record in event["Records"]:
        bucket = record["s3"]["bucket"]["name"]
        key = urllib.parse.unquote_plus(record["s3"]["object"]["key"])  # S3 sends keys URL-encoded
        if not key.startswith("uploads/") or not key.lower().endswith(".pdf"):
            continue
        data = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        reader = PdfReader(io.BytesIO(data))
        pages = [(n, page.extract_text() or "") for n, page in enumerate(reader.pages, start=1)]
        chunks = chunk_pages(pages)
        doc = key.split("/", 1)[1]
        for n, chunk in enumerate(chunks):
            chunk.update(id=f"{doc}#{n}", doc=doc, embedding=embed(chunk["text"]))
        s3.put_object(
            Bucket=bucket, Key=f"index/{doc}.json", Body=json.dumps(chunks), ContentType="application/json"
        )
        print(f"Indexed {doc}: {len(pages)} pages, {len(chunks)} chunks")
    return {"ok": True}
