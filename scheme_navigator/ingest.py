"""Ingest scheme documents (.md / .txt / .pdf) into ChromaDB.

Usage:  python ingest.py [--reset]

Markdown/text files that contain "## SCHEME: <name>" headings are split into
one chunk per "### " subsection (Benefit, Eligibility, Documents, How to apply),
so each chunk stays small and carries the scheme name + source URL as metadata.
Other files (PDFs, plain text) are split with a recursive character splitter.
"""
import os
import re
import sys
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

DATA_DIR = Path(__file__).parent / "data" / "schemes"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "schemes"


def get_collection(reset: bool = False):
    client = chromadb.PersistentClient(path=str(DB_DIR))
    if reset:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    # Open-source embeddings. Default: all-MiniLM-L6-v2 (English, ONNX, tiny).
    # For Hindi/Kannada/etc. queries set EMBED_MODEL=BAAI/bge-m3 (or
    # paraphrase-multilingual-MiniLM-L12-v2) and re-run `python ingest.py --reset`.
    model = os.getenv("EMBED_MODEL")
    if model:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model)
    else:
        ef = embedding_functions.DefaultEmbeddingFunction()
    return client.get_or_create_collection(COLLECTION, embedding_function=ef)


def chunk_scheme_markdown(text: str, filename: str):
    """Yield (id, document, metadata) from a '## SCHEME:' structured file."""
    blocks = re.split(r"^## SCHEME:\s*", text, flags=re.M)[1:]
    for block in blocks:
        name, _, body = block.partition("\n")
        name = name.strip()
        src = re.search(r"^Source:\s*(\S+)", body, flags=re.M)
        url = src.group(1) if src else ""
        sections = re.split(r"^### ", body, flags=re.M)
        for sec in sections[1:]:
            title, _, content = sec.partition("\n")
            title, content = title.strip(), content.strip()
            if not content:
                continue
            doc = f"Scheme: {name}\nSection: {title}\n{content}"
            yield (
                f"{filename}:{name}:{title}",
                doc,
                {"scheme": name, "section": title, "source": url or filename, "file": filename},
            )



def chunk_generic(path: Path):
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader

        text = "\n".join(
            (p.extract_text() or "")
            for p in PdfReader(str(path)).pages
        )
    else:
        text = path.read_text(encoding="utf-8")

    # Simple text splitting without LangChain.
    chunk_size = 900
    overlap = 120

    start = 0
    pieces = []

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()

        if piece:
            pieces.append(piece)

        if end >= len(text):
            break

        start = end - overlap

    scheme = path.stem.replace("_", " ")

    for i, piece in enumerate(pieces):
        yield (
            f"{path.name}:{i}",
            f"Scheme: {scheme}\n{piece}",
            {
                "scheme": scheme,
                "section": f"part {i + 1}",
                "source": path.name,
                "file": path.name,
            },
        )



def main(reset: bool = False):
    col = get_collection(reset)
    ids, docs, metas = [], [], []
    for path in sorted(DATA_DIR.glob("*")):
        if path.suffix.lower() not in {".md", ".txt", ".pdf"}:
            continue
        text = path.read_text(encoding="utf-8") if path.suffix.lower() != ".pdf" else ""
        if "## SCHEME:" in text:
            chunks = chunk_scheme_markdown(text, path.name)
        else:
            chunks = chunk_generic(path)
        for i, d, m in chunks:
            ids.append(i), docs.append(d), metas.append(m)
    if not ids:
        print(f"No documents found in {DATA_DIR}")
        return
    col.upsert(ids=ids, documents=docs, metadatas=metas)
    print(f"Indexed {len(ids)} chunks into '{COLLECTION}' ({col.count()} total).")


if __name__ == "__main__":
    main(reset="--reset" in sys.argv)