"""PDF/HTM ingestion for the Chapter 7 Qdrant collections.

Sources (already downloaded to `./data/` by the repo):

- **OpenAI agents guide** (PDF) -> Qdrant collection `opnai_data`
- **Lyft 10-Ks** for 2020-2022 (PDF) and **Uber 10-K 2021** (HTM) -> Qdrant collection `10k_data`

Pipeline: read -> recursive chunk (chunk_size=2048, overlap=50) -> Nomic
embeddings -> Qdrant upsert. Mirrors the original Colab loader at
hamzafarooq/multi-agent-course/Module_3_Agentic_RAG.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from uuid import uuid4

import fitz  # PyMuPDF
from qdrant_client import AsyncQdrantClient, models

# Reuse Chapter 6's chunker so we don't depend on LangChain
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "chapter_06_rag"))
from chunking import simple_recursive_split  # noqa: E402

from agentic_router import COLLECTIONS, EMBEDDING_DIM, _get_encoder  # noqa: E402


DATA_DIR = Path(__file__).resolve().parent / "data"

OPENAI_FILES = [DATA_DIR / "openai_agents_guide.pdf"]

TENK_FILES = [
    DATA_DIR / "lyft_10k_2020.pdf",
    DATA_DIR / "lyft_10k_2021.pdf",
    DATA_DIR / "lyft_10k_2022.pdf",
    DATA_DIR / "uber_10k_2021.htm",
]


def read_pdf(path: Path) -> str:
    doc = fitz.open(path)
    return "".join(page.get_text() for page in doc)


def read_htm(path: Path) -> str:
    """Strip tags from a SEC EDGAR HTM filing — good enough for retrieval."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_document(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        return read_pdf(path)
    if path.suffix.lower() in (".htm", ".html"):
        return read_htm(path)
    raise ValueError(f"Unsupported file type: {path}")


def chunk_documents(paths: list[Path], *, chunk_size: int = 2048, chunk_overlap: int = 50):
    """Returns a flat list of {page_content, metadata} dicts across all files."""
    chunks: list[dict] = []
    for path in paths:
        if not path.exists():
            print(f"  skipping (missing): {path.name}")
            continue
        text = load_document(path)
        if not text.strip():
            print(f"  skipping (empty):   {path.name}")
            continue
        doc = {"page_content": text, "metadata": {"document_info": str(path)}}
        new_chunks = simple_recursive_split(
            doc, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        for c in new_chunks:
            c["metadata"]["uuid"] = str(uuid4())
        chunks.extend(new_chunks)
        print(f"  {path.name:<32} -> {len(new_chunks):>4} chunks")
    return chunks


async def build_collection(
    qdrant: AsyncQdrantClient, collection_name: str, chunks: list[dict]
) -> None:
    if await qdrant.collection_exists(collection_name):
        await qdrant.delete_collection(collection_name=collection_name)
    await qdrant.create_collection(
        collection_name=collection_name,
        vectors_config=models.VectorParams(
            size=EMBEDDING_DIM, distance=models.Distance.COSINE
        ),
    )

    encoder = _get_encoder()
    texts = [c["page_content"] for c in chunks]
    print(f"  embedding {len(texts)} chunks for '{collection_name}'...")
    vectors = encoder.encode(
        texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True
    ).astype("float32")

    points = [
        models.PointStruct(
            id=chunks[i]["metadata"]["uuid"],
            vector=vectors[i].tolist(),
            payload={"metadata": chunks[i]["metadata"], "content": chunks[i]["page_content"]},
        )
        for i in range(len(chunks))
    ]
    await qdrant.upsert(collection_name=collection_name, wait=True, points=points)
    print(f"  upserted {len(points)} points into '{collection_name}'")


async def ingest_all(qdrant: AsyncQdrantClient) -> None:
    print("Chunking OpenAI agents guide...")
    openai_chunks = chunk_documents(OPENAI_FILES)
    print("\nChunking 10-K filings...")
    tenk_chunks = chunk_documents(TENK_FILES)

    print("\nBuilding Qdrant collections...")
    await build_collection(qdrant, COLLECTIONS["OPENAI_QUERY"], openai_chunks)
    await build_collection(qdrant, COLLECTIONS["10K_DOCUMENT_QUERY"], tenk_chunks)
    print("\nDone.")
