"""HTTP server (FastAPI) — exposes news-summarizer query interface.

Endpoints:
  GET  /health                          health check
  GET  /stats                           DB row counts
  GET  /clusters                        list clusters
  GET  /clusters/{slug}                 cluster detail
  GET  /clusters/{slug}/timeline        story events
  POST /think                           ask a question (returns synthesized answer)
  POST /compare                         compare how a topic is covered
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from . import storage
from .query import (
    ask_topic,
    compare_sources,
    get_cluster_summary,
    get_story_timeline,
    list_clusters,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    storage.init_db()
    yield


# ---------- app ----------

app = FastAPI(
    title="news-summarizer HTTP API",
    description="Silver layer query interface over news-aggregator bronze",
    version="0.3.0",
    lifespan=lifespan,
)

# Permissive CORS for local dev / dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- request/response models ----------

class ThinkRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The question/topic to think about")
    limit: int = Field(5, ge=1, le=50, description="Max number of top articles to return")


class CompareRequest(BaseModel):
    topic: str = Field(..., min_length=1, description="The topic to compare")
    limit: int = Field(50, ge=1, le=200, description="Max summaries to return")


# ---------- endpoints ----------

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.3.0"}


@app.get("/stats")
def stats() -> dict:
    with storage.connect() as con:
        clusters = con.execute("SELECT COUNT(*) FROM clusters").fetchone()[0]
        articles = con.execute("SELECT COUNT(*) FROM cluster_articles").fetchone()[0]
        summaries = con.execute("SELECT COUNT(*) FROM summaries").fetchone()[0]
        entities = con.execute("SELECT COUNT(*) FROM entities").fetchone()[0]
        errors = con.execute("SELECT COUNT(*) FROM enrichment_errors WHERE retriable=1").fetchone()[0]
    return {
        "clusters": clusters,
        "cluster_articles": articles,
        "summaries": summaries,
        "entities": entities,
        "pending_errors": errors,
    }


@app.get("/clusters")
def list_clusters_endpoint(limit: int = 100) -> list[dict]:
    return list_clusters(limit=limit)


@app.get("/clusters/{slug}")
def get_cluster_endpoint(slug: str) -> dict:
    c = get_cluster_summary(slug)
    if c is None:
        raise HTTPException(status_code=404, detail=f"cluster not found: {slug}")
    return c


@app.get("/clusters/{slug}/timeline")
def get_timeline_endpoint(slug: str) -> list[dict]:
    c = storage.get_cluster(slug)
    if not c:
        raise HTTPException(status_code=404, detail=f"cluster not found: {slug}")
    return get_story_timeline(slug)


@app.post("/think")
def think_endpoint(req: ThinkRequest) -> dict:
    return ask_topic(req.question, limit=req.limit)


@app.post("/compare")
def compare_endpoint(req: CompareRequest) -> dict:
    return compare_sources(req.topic, limit=req.limit)


# ---------- entry point ----------

def main():
    """Run uvicorn server. Use: python -m news_summarizer.server_http"""
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9120, log_level="info")


if __name__ == "__main__":
    main()
