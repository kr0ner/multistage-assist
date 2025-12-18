"""
Semantic Reranker API - CrossEncoder reranking service.

Home Assistant addon that exposes sentence-transformers CrossEncoder
as HTTP API for use by Multi-Stage Assist semantic cache.
"""

import os
import sys
import logging
from typing import List

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

# Configuration from environment (set by run.sh from addon options)
MODEL_NAME = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEVICE = os.getenv("RERANKER_DEVICE", "cpu")

# Setup logging - DEBUG level for verbose output
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,  # Ensure logs go to stdout for HA addon logs
)
logger = logging.getLogger("reranker")

# Also set uvicorn access log level
logging.getLogger("uvicorn.access").setLevel(logging.INFO)

# Initialize FastAPI
app = FastAPI(
    title="Semantic Reranker API",
    description="CrossEncoder reranking for Multi-Stage Assist",
    version="1.0.0",
)

# Global model reference
model: CrossEncoder = None
model_loading = True


class RerankRequest(BaseModel):
    """Request body for reranking."""

    query: str
    candidates: List[str]


class RerankResponse(BaseModel):
    """Response with reranking scores."""

    scores: List[float]
    best_index: int
    best_score: float


@app.on_event("startup")
async def load_model():
    """Load CrossEncoder model on startup."""
    global model, model_loading
    logger.info("=" * 50)
    logger.info("STARTING SEMANTIC RERANKER ADDON")
    logger.info("=" * 50)
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"Device: {DEVICE}")
    logger.info("This may take a few minutes on first run (downloading model)...")
    
    try:
        model = CrossEncoder(MODEL_NAME, device=DEVICE)
        model_loading = False
        logger.info("=" * 50)
        logger.info("MODEL LOADED SUCCESSFULLY!")
        logger.info("=" * 50)
        logger.info("Ready to accept requests on /health and /rerank")
    except Exception as e:
        logger.error(f"FAILED TO LOAD MODEL: {e}")
        raise


@app.get("/health")
async def health():
    """Health check endpoint."""
    status = "loading" if model_loading else ("ok" if model is not None else "error")
    response = {
        "status": status,
        "model": MODEL_NAME,
        "device": DEVICE,
    }
    logger.debug(f"Health check: {response}")
    return response


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank candidates against a query.

    Returns sigmoid probabilities (0-1) for each candidate.
    """
    logger.info(f"Rerank request: query='{request.query[:50]}...' candidates={len(request.candidates)}")
    
    if model is None or model_loading:
        logger.warning("Model not loaded yet, returning 503")
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.candidates:
        logger.warning("No candidates provided")
        raise HTTPException(status_code=400, detail="No candidates provided")

    # Log candidates for debugging
    for i, c in enumerate(request.candidates):
        logger.debug(f"  Candidate {i}: '{c[:60]}...'")

    # Create query-candidate pairs
    pairs = [[request.query, candidate] for candidate in request.candidates]

    # Get raw scores and convert to probabilities via sigmoid
    logger.debug("Running CrossEncoder prediction...")
    raw_scores = model.predict(pairs)
    probs = 1 / (1 + np.exp(-raw_scores))

    scores = probs.tolist()
    best_idx = int(np.argmax(probs))

    # Log detailed results
    logger.info(f"Rerank results: best_idx={best_idx}, best_score={probs[best_idx]:.4f}")
    for i, (score, candidate) in enumerate(zip(scores, request.candidates)):
        marker = " <-- BEST" if i == best_idx else ""
        logger.debug(f"  [{i}] score={score:.4f}: '{candidate[:40]}'{marker}")

    return RerankResponse(
        scores=scores,
        best_index=best_idx,
        best_score=float(probs[best_idx]),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9876, log_level="debug")
