"""
Fake Review Detector - FastAPI Backend
======================================
Serves the trained model via REST API with explainability features.

Endpoints:
    POST /analyze  - Analyze a review text
    GET  /health   - Health check

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import re
from contextlib import asynccontextmanager
from typing import Dict, List, Optional

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = os.environ.get("MODEL_DIR", "model")
DEFAULT_MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
ID2LABEL = {0: "fake", 1: "real"}
LABEL2ID = {"fake": 0, "real": 1}


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Review text to analyze")


class AnalyzeResponse(BaseModel):
    prediction: str = Field(..., description="'fake' or 'real'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    suspicious_phrases: List[str] = Field(
        default_factory=list, description="Suspicious phrases detected by the model"
    )
    processing_time_ms: float = Field(..., description="Inference time in milliseconds")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


# ---------------------------------------------------------------------------
# Model manager (loaded on startup)
# ---------------------------------------------------------------------------

class ModelManager:
    """Manages the lifecycle of the transformer model and tokenizer."""

    def __init__(self):
        self.tokenizer: Optional[DistilBertTokenizer] = None
        self.model: Optional[DistilBertForSequenceClassification] = None
        self.device: torch.device = DEVICE

    def load(self) -> None:
        """Load tokenizer and model from disk or download from HuggingFace."""
        if os.path.isdir(MODEL_DIR) and any(os.scandir(MODEL_DIR)):
            print(f"[INFO] Loading model from local directory: {MODEL_DIR}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                MODEL_DIR,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        else:
            print(f"[INFO] No local model found in '{MODEL_DIR}'.")
            print(f"[INFO] Downloading base model: {DEFAULT_MODEL_NAME}")
            self.tokenizer = DistilBertTokenizer.from_pretrained(DEFAULT_MODEL_NAME)
            self.model = DistilBertForSequenceClassification.from_pretrained(
                DEFAULT_MODEL_NAME,
                num_labels=2,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
            # Save for future use
            os.makedirs(MODEL_DIR, exist_ok=True)
            self.model.save_pretrained(MODEL_DIR)
            self.tokenizer.save_pretrained(MODEL_DIR)

        self.model.to(self.device)
        self.model.eval()
        print(f"[INFO] Model loaded on device: {self.device}")

    @torch.no_grad()
    def analyze(self, text: str) -> Dict:
        """
        Run inference and extract suspicious phrases using attention weights.
        """
        import time

        start_time = time.perf_counter()

        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Forward pass with attention output
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
        )
        logits = outputs.logits
        attentions = outputs.attentions  # Tuple of (layers,) each [batch, heads, seq, seq]

        # Prediction and confidence
        probs = torch.softmax(logits, dim=-1)
        predicted_id = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0][predicted_id].item())
        prediction = ID2LABEL[predicted_id]

        # Explainability: extract suspicious phrases from attention
        suspicious_phrases = self._extract_suspicious_phrases(
            input_ids[0], attentions, text
        )

        elapsed = (time.perf_counter() - start_time) * 1000  # ms

        return {
            "prediction": prediction,
            "confidence": confidence,
            "suspicious_phrases": suspicious_phrases,
            "processing_time_ms": round(elapsed, 2),
        }

    def _extract_suspicious_phrases(
        self,
        input_ids: torch.Tensor,
        attentions: tuple,
        original_text: str,
        top_k: int = 8,
        threshold_percentile: float = 75.0,
    ) -> List[str]:
        """
        Extract suspicious phrases using attention-based explainability.

        Strategy:
            1. Average attention across all heads in the last transformer layer
            2. Take [CLS] token attention to all other tokens
            3. Normalize and threshold to find important tokens
            4. Group consecutive important tokens into phrases
            5. Clean and deduplicate
        """
        if attentions is None or len(attentions) == 0:
            return []

        # Last layer attention: [batch=1, num_heads, seq_len, seq_len]
        last_layer_attention = attentions[-1][0]  # [heads, seq, seq]

        # Average across heads
        avg_attention = last_layer_attention.mean(dim=0)  # [seq, seq]

        # [CLS] token (index 0) attention to all tokens
        cls_attention = avg_attention[0].cpu().numpy()  # [seq_len]

        # Exclude [CLS] itself and [SEP]/pad tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        valid_indices = []
        for i, tok in enumerate(tokens):
            if tok in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token):
                continue
            valid_indices.append(i)

        if len(valid_indices) == 0:
            return []

        valid_attention = cls_attention[valid_indices]

        # Normalize valid attention scores
        if valid_attention.max() > 0:
            valid_attention = valid_attention / valid_attention.max()

        # Threshold: keep tokens above percentile threshold
        threshold = np.percentile(valid_attention, threshold_percentile)

        # Get important token indices
        important_valid_idx = [i for i, score in enumerate(valid_attention) if score >= threshold]
        important_seq_indices = [valid_indices[i] for i in important_valid_idx]
        important_seq_indices.sort()

        # Group consecutive indices into phrase spans
        spans = []
        if important_seq_indices:
            start = important_seq_indices[0]
            prev = important_seq_indices[0]
            for idx in important_seq_indices[1:]:
                if idx == prev + 1:
                    prev = idx
                else:
                    spans.append((start, prev))
                    start = idx
                    prev = idx
            spans.append((start, prev))

        # Convert spans to phrases using word mapping
        encodings = self.tokenizer(
            original_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )
        word_ids = encodings.word_ids(batch_index=0)

        phrases = []
        for start_idx, end_idx in spans:
            # Collect tokens in this span
            span_tokens = []
            for i in range(start_idx, end_idx + 1):
                if i < len(tokens):
                    tok = tokens[i]
                    if tok.startswith("##"):
                        span_tokens.append(tok[2:])
                    else:
                        if span_tokens:
                            span_tokens.append(" ")
                        span_tokens.append(tok)

            phrase = "".join(span_tokens).strip()
            # Clean up
            phrase = re.sub(r"\s+", " ", phrase)
            phrase = phrase.strip(".,!?;:'\"")

            if len(phrase) >= 2 and phrase not in phrases:
                phrases.append(phrase)

        # Limit to top_k most important phrases by attention score
        phrase_scores = []
        for phrase in phrases:
            # Find average attention score for tokens in this phrase
            score = 0.0
            count = 0
            # Simple matching: find tokens that compose this phrase
            for i, tok in enumerate(tokens):
                if tok in (self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token):
                    continue
                # Check if token is part of any span
                for start_idx, end_idx in spans:
                    if start_idx <= i <= end_idx:
                        # Get the attention score for this token
                        if i in valid_indices:
                            vi = valid_indices.index(i)
                            score += float(valid_attention[vi])
                            count += 1
                        break
            if count > 0:
                phrase_scores.append((phrase, score / count))

        phrase_scores.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in phrase_scores[:top_k]]


# Global model manager
model_manager = ModelManager()


# ---------------------------------------------------------------------------
# Lifespan context manager (replaces deprecated @app.on_event)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    model_manager.load()
    yield
    # Cleanup (if needed)


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Fake Review Detector API",
    description="Detects fake reviews using DistilBERT with attention-based explainability",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS - allow Chrome extension to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model is not None,
        device=str(model_manager.device),
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest) -> AnalyzeResponse:
    """
    Analyze a review text and return prediction with suspicious phrases.
    """
    if model_manager.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    result = model_manager.analyze(request.text)
    return AnalyzeResponse(**result)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
