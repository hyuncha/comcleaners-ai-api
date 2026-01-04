import os
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from qdrant_client import QdrantClient
from openai import OpenAI

app = FastAPI(title="ComCleaners RAG Service")

# Environment variables
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "cleaning_knowledge")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")


def get_qdrant() -> QdrantClient:
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)


def get_openai() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


# ---------- Models ----------

class AskRequest(BaseModel):
    question: str


class Source(BaseModel):
    content: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: List[Source]


# ---------- Endpoints ----------

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>ComCleaners RAG Service</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
            h1 { color: #333; }
            a { color: #0066cc; }
            .endpoints { background: #f5f5f5; padding: 15px; border-radius: 8px; }
        </style>
    </head>
    <body>
        <h1>ComCleaners RAG Service</h1>
        <p>Cleaning Knowledge RAG API</p>
        <div class="endpoints">
            <h3>Endpoints</h3>
            <ul>
                <li><a href="/api/health">/api/health</a> - Health check</li>
                <li><a href="/api/qdrant/ping">/api/qdrant/ping</a> - Qdrant status</li>
                <li><a href="/docs">/docs</a> - API Documentation</li>
            </ul>
        </div>
    </body>
    </html>
    """


@app.get("/api/health")
def health():
    return {"ok": True}


@app.get("/api/qdrant/ping")
def qdrant_ping():
    try:
        client = get_qdrant()
        collections = client.get_collections()
        return {
            "ok": True,
            "collections": [c.name for c in collections.collections]
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """
    RAG endpoint: Search Qdrant for relevant documents and generate answer with OpenAI.
    """
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")

    try:
        openai_client = get_openai()
        qdrant_client = get_qdrant()

        # 1. Generate embedding for the question
        embedding_response = openai_client.embeddings.create(
            model=OPENAI_EMBED_MODEL,
            input=request.question
        )
        query_vector = embedding_response.data[0].embedding

        # 2. Search Qdrant for relevant documents
        search_results = qdrant_client.search(
            collection_name=QDRANT_COLLECTION,
            query_vector=query_vector,
            limit=5
        )

        # 3. Build context from search results
        sources = []
        context_parts = []
        for result in search_results:
            content = result.payload.get("content", "") if result.payload else ""
            if content:
                context_parts.append(content)
                sources.append(Source(content=content[:200], score=result.score))

        context = "\n\n".join(context_parts)

        # 4. Generate answer with OpenAI
        if context:
            system_prompt = """You are a helpful cleaning knowledge assistant.
Answer the user's question based on the provided context.
If the context doesn't contain relevant information, say so.
Answer in the same language as the question."""

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {request.question}"}
            ]
        else:
            # No context found, answer directly
            messages = [
                {"role": "system", "content": "You are a helpful cleaning knowledge assistant. Answer in the same language as the question."},
                {"role": "user", "content": request.question}
            ]

        chat_response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=1000
        )

        answer = chat_response.choices[0].message.content

        return AskResponse(answer=answer, sources=sources)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
