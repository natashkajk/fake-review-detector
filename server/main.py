"""
Fake Review Detector - FastAPI Backend
AI-powered fake review detection using BERT NLP model
"""

import os
import re
import time
import json
import sqlite3
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ============================================================================
# Configuration
# ============================================================================

MODEL_DIR = os.environ.get("MODEL_DIR", "./model_mixed")
DB_PATH = os.environ.get("DB_PATH", "./reviews.db")
FALLBACK_MODEL = False  # Use rule-based fallback only if model loading fails
MAX_TEXT_LENGTH = 512
MIN_TEXT_LENGTH = 5

PATTERN_WEIGHTS = {
    "fake_indicators": 0.30,
    "unnatural_language": 0.20,
    "excessive_praise": 0.15,
    "generic_phrases": 0.10,
    "repetitive_patterns": 0.10,
}

SIGNAL_LABELS = {
    "fake_indicators": "Disclosure-style wording",
    "unnatural_language": "Marketing-style phrasing",
    "excessive_praise": "Overly enthusiastic language",
    "generic_phrases": "Template-like wording",
    "repetitive_patterns": "Repetitive text pattern",
    "short_review": "Very short review",
    "capitalization": "Heavy capitalization",
    "exclamation_marks": "Excessive exclamation marks",
    "all_caps_words": "All-caps emphasis",
    "fallback_context": "Context summary",
}

SIGNAL_REASONS = {
    "fake_indicators": "Disclosure wording often appears in incentivized or coordinated reviews, so it raises suspicion.",
    "unnatural_language": "Call-to-action language sounds promotional rather than like a natural customer opinion.",
    "excessive_praise": "Extreme praise without concrete detail often pushes the verdict toward fake.",
    "generic_phrases": "Template-like phrases reduce originality and can increase the fake score.",
    "repetitive_patterns": "Repeated characters or words can indicate spammy or low-quality synthetic text.",
    "short_review": "Very short reviews provide too little detail, which makes the decision less trustworthy.",
    "capitalization": "A high ratio of capital letters can look promotional or emotionally exaggerated.",
    "exclamation_marks": "Too many exclamation marks make the review look more like promotion than feedback.",
    "all_caps_words": "All-caps words add aggressive emphasis and slightly increase suspicion.",
    "fallback_context": "No strong suspicious fragment was found, so this is the main text span used for context.",
}

# Suspicious patterns for explainability
SUSPICIOUS_PATTERNS = {
    "excessive_praise": [
        r"\b(best|greatest|amazing|awesome|incredible|fantastic|perfect|excellent)\s+(ever|product|service|experience)\b",
        r"\b(must\s+have|highly\s+recommend|five\s+stars|10/10)\b",
    ],
    "generic_phrases": [
        r"\b(great\s+product|good\s+quality|works\s+well|as\s+described)\b",
        r"\b(very\s+satisfied|happy\s+with|love\s+this)\s+\w+\b",
    ],
    "unnatural_language": [
        r"\b(click\s+here|buy\s+now|order\s+today|limited\s+time)\b",
        r"\b(don'?t\s+miss|act\s+now|special\s+offer)\b",
    ],
    "repetitive_patterns": [
        r"(.)\1{4,}",  # Repeated characters
        r"\b(\w+)\s+\1\b",  # Repeated words
    ],
    "fake_indicators": [
        r"\b(free\s+product|received\s+this|in\s+exchange|honest\s+review)\b",
        r"\b(disclaimer|sponsored|paid\s+review|affiliate)\b",
    ]
}

# ============================================================================
# Pydantic Models
# ============================================================================

class ReviewRequest(BaseModel):
    """Request model for review analysis"""
    text: str = Field(..., min_length=MIN_TEXT_LENGTH, max_length=2000, 
                      description="Review text to analyze")


class ReviewResponse(BaseModel):
    """Response model for review analysis"""
    analysis_id: Optional[int] = Field(
        default=None,
        description="Database id of the stored analysis row"
    )
    prediction: str = Field(..., description="Prediction: 'fake' or 'genuine'")
    confidence: float = Field(..., ge=0.0, le=1.0, 
                              description="Confidence score (0-1)")
    review_text: str = Field(..., description="Analyzed review text")
    evidence_text: str = Field(
        default="",
        description="Primary suspicious text span used as evidence for the verdict"
    )
    evidence_label: str = Field(
        default="",
        description="Short label for the strongest trigger behind the verdict"
    )
    evidence_reason: str = Field(
        default="",
        description="Why the strongest trigger affected the confidence"
    )
    suspicious_phrases: List[str] = Field(default=[], 
                                          description="List of suspicious phrases found")
    explanation: str = Field(..., description="Human-readable explanation")
    processing_time: float = Field(..., description="Processing time in seconds")
    model_used: str = Field(..., description="Model used for prediction")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_name: str
    version: str = "1.0.0"


class FeedbackRequest(BaseModel):
    """Manual label update for a stored analysis row."""
    analysis_id: int = Field(..., ge=1, description="Stored analysis id")
    human_label: str = Field(..., description="'fake', 'genuine', or 'unknown'")
    notes: str = Field(default="", max_length=1000, description="Optional reviewer notes")


class FeedbackResponse(BaseModel):
    """Response after updating manual review feedback."""
    success: bool
    analysis_id: int
    human_label: str


class ReviewRecord(BaseModel):
    """Stored review analysis row."""
    analysis_id: int
    review_text: str
    prediction: str
    confidence: float
    evidence_text: str = ""
    evidence_label: str = ""
    evidence_reason: str = ""
    suspicious_phrases: List[str] = Field(default_factory=list)
    explanation: str = ""
    model_used: str = ""
    source_url: str = ""
    human_label: str = "unknown"
    notes: str = ""
    created_at: str
    updated_at: str


# ============================================================================
# Model Manager
# ============================================================================

class ReviewDatabase:
    """Simple SQLite storage for analyzed reviews and manual feedback."""

    def __init__(self, db_path: str):
        self.db_path = db_path

    def initialize(self) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS review_analyses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    review_text TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    evidence_text TEXT,
                    evidence_label TEXT,
                    evidence_reason TEXT,
                    suspicious_phrases TEXT,
                    explanation TEXT,
                    model_used TEXT,
                    source_url TEXT,
                    human_label TEXT DEFAULT 'unknown',
                    notes TEXT DEFAULT '',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_review_analyses_created_at
                ON review_analyses(created_at DESC)
                """
            )
            conn.commit()

    def insert_analysis(
        self,
        *,
        review_text: str,
        prediction: str,
        confidence: float,
        evidence_text: str,
        evidence_label: str,
        evidence_reason: str,
        suspicious_phrases: List[str],
        explanation: str,
        model_used: str,
        source_url: str = "",
    ) -> int:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                INSERT INTO review_analyses (
                    review_text,
                    prediction,
                    confidence,
                    evidence_text,
                    evidence_label,
                    evidence_reason,
                    suspicious_phrases,
                    explanation,
                    model_used,
                    source_url
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    review_text,
                    prediction,
                    confidence,
                    evidence_text,
                    evidence_label,
                    evidence_reason,
                    json.dumps(suspicious_phrases, ensure_ascii=False),
                    explanation,
                    model_used,
                    source_url,
                ),
            )
            conn.commit()
            return int(cursor.lastrowid)

    def update_feedback(self, analysis_id: int, human_label: str, notes: str = "") -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE review_analyses
                SET human_label = ?, notes = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = ?
                """,
                (human_label, notes, analysis_id),
            )
            conn.commit()
            if cursor.rowcount == 0:
                raise ValueError(f"Analysis id {analysis_id} was not found")

    def get_recent_reviews(self, limit: int = 20) -> List[ReviewRecord]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                """
                SELECT
                    id,
                    review_text,
                    prediction,
                    confidence,
                    evidence_text,
                    evidence_label,
                    evidence_reason,
                    suspicious_phrases,
                    explanation,
                    model_used,
                    source_url,
                    human_label,
                    notes,
                    created_at,
                    updated_at
                FROM review_analyses
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        records: List[ReviewRecord] = []
        for row in rows:
            suspicious_phrases = []
            if row["suspicious_phrases"]:
                try:
                    suspicious_phrases = json.loads(row["suspicious_phrases"])
                except json.JSONDecodeError:
                    suspicious_phrases = []

            records.append(
                ReviewRecord(
                    analysis_id=int(row["id"]),
                    review_text=row["review_text"],
                    prediction=row["prediction"],
                    confidence=float(row["confidence"]),
                    evidence_text=row["evidence_text"] or "",
                    evidence_label=row["evidence_label"] or "",
                    evidence_reason=row["evidence_reason"] or "",
                    suspicious_phrases=suspicious_phrases,
                    explanation=row["explanation"] or "",
                    model_used=row["model_used"] or "",
                    source_url=row["source_url"] or "",
                    human_label=row["human_label"] or "unknown",
                    notes=row["notes"] or "",
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )

        return records

class ModelManager:
    """Manages the BERT model for fake review detection"""
    
    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.model_loaded = False
        self.temperature = 1.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the BERT model and tokenizer"""
        try:
            resolved_model_dir = os.path.abspath(MODEL_DIR)
            print(f"[ModelManager] Loading model: {resolved_model_dir}")
            print(f"[ModelManager] Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
            self.model.to(self.device)
            self.model.eval()
            self.temperature = self._load_temperature(MODEL_DIR)
            self.model_loaded = True
            
            print("[ModelManager] Model loaded successfully")
            
        except Exception as e:
            print(f"[ModelManager] Failed to load model: {e}")
            print("[ModelManager] Will use rule-based fallback")
            self.model_loaded = False
    
    def predict(self, text: str) -> tuple[str, float]:
        """
        Predict if a review is fake or genuine
        
        Returns:
            tuple: (prediction: 'fake' or 'genuine', confidence: float)
        """
        if not self.model_loaded or FALLBACK_MODEL:
            return self._rule_based_predict(text)
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=MAX_TEXT_LENGTH,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                calibrated_logits = outputs.logits / self.temperature
                probabilities = torch.softmax(calibrated_logits, dim=1)
                
            predicted_id = int(torch.argmax(probabilities, dim=1).item())
            prediction = self.model.config.id2label.get(predicted_id, "genuine").lower()
            confidence = float(probabilities[0][predicted_id].item())
            return prediction, round(confidence, 4)
                
        except Exception as e:
            print(f"[ModelManager] Prediction error: {e}")
            return self._rule_based_predict(text)
    
    def _rule_based_predict(self, text: str) -> tuple[str, float]:
        """Rule-based prediction when model is not available"""
        fake_probability = self._calculate_pattern_score(text)
        prediction = "fake" if fake_probability > 0.5 else "genuine"
        return prediction, self._calibrate_confidence(fake_probability, using_fallback=True)

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        """Clamp a value to the given range."""
        return max(minimum, min(value, maximum))

    def _load_temperature(self, model_dir: str) -> float:
        """Load learned temperature scaling value if available."""
        temperature_path = os.path.join(model_dir, "temperature.json")
        if not os.path.exists(temperature_path):
            return 1.0

        try:
            with open(temperature_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            temperature = float(payload.get("temperature", 1.0))
            return self._clamp(temperature, 0.5, 5.0)
        except Exception as error:
            print(f"[ModelManager] Failed to load temperature: {error}")
            return 1.0

    def _calibrate_confidence(self, fake_probability: float, using_fallback: bool = False) -> float:
        """
        Convert the raw fake probability into a softer confidence score.

        The previous implementation pushed many low-pattern reviews straight to 95%.
        Here we instead scale confidence by distance from the decision boundary (0.5),
        which makes middling cases look middling and reserves high confidence for
        clearly separated examples.
        """
        distance_from_boundary = abs(fake_probability - 0.5) * 2  # 0..1
        base = 0.50 + (distance_from_boundary * 0.38)

        if using_fallback:
            # Rule-based mode is less trustworthy than model-assisted mode.
            base -= 0.08

        return round(self._clamp(base, 0.50, 0.88), 4)
    
    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate suspicious pattern score"""
        text_lower = text.lower()
        score = 0.0
        
        # Check each pattern category
        for category, patterns in SUSPICIOUS_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    score += PATTERN_WEIGHTS.get(category, 0.10) * matches
        
        # Additional heuristics
        words = text.split()
        
        # Very short reviews are suspicious
        if len(words) < 10:
            score += 0.1
        
        # Excessive capitalization
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            score += 0.15
        
        # Excessive exclamation marks
        excl_count = text.count('!')
        if excl_count > 3:
            score += 0.1 * min(excl_count - 3, 3)
        
        # All caps words
        all_caps_words = [w for w in words if w.isupper() and len(w) > 2]
        score += 0.05 * len(all_caps_words)
        
        return min(score, 1.0)


# Global model manager
model_manager = ModelManager()
review_db = ReviewDatabase(DB_PATH)


# ============================================================================
# Explainability Engine
# ============================================================================

class ExplainabilityEngine:
    """Generates explanations and highlights suspicious phrases"""

    @staticmethod
    def analyze_signals(text: str) -> List[Dict[str, Any]]:
        """Return ranked signals that contributed to the verdict."""
        signals: List[Dict[str, Any]] = []
        text_lower = text.lower()

        for category, patterns in SUSPICIOUS_PATTERNS.items():
            weight = PATTERN_WEIGHTS.get(category, 0.10)
            for pattern in patterns:
                for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                    start = max(0, match.start() - 10)
                    end = min(len(text), match.end() + 10)
                    fragment = text[start:end].strip()
                    if not fragment:
                        continue
                    signals.append({
                        "category": category,
                        "weight": weight,
                        "text": fragment,
                        "label": SIGNAL_LABELS.get(category, "Suspicious signal"),
                        "reason": SIGNAL_REASONS.get(category, "This pattern increased the suspiciousness score."),
                    })

        words = text.split()
        if len(words) < 10:
            signals.append({
                "category": "short_review",
                "weight": 0.10,
                "text": " ".join(words[: min(len(words), 12)]),
                "label": SIGNAL_LABELS["short_review"],
                "reason": SIGNAL_REASONS["short_review"],
            })

        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
        if caps_ratio > 0.3:
            signals.append({
                "category": "capitalization",
                "weight": 0.15,
                "text": text[:140] + ("..." if len(text) > 140 else ""),
                "label": SIGNAL_LABELS["capitalization"],
                "reason": SIGNAL_REASONS["capitalization"],
            })

        excl_count = text.count('!')
        if excl_count > 3:
            signals.append({
                "category": "exclamation_marks",
                "weight": 0.1 * min(excl_count - 3, 3),
                "text": text[:140] + ("..." if len(text) > 140 else ""),
                "label": SIGNAL_LABELS["exclamation_marks"],
                "reason": SIGNAL_REASONS["exclamation_marks"],
            })

        all_caps_words = [w for w in words if w.isupper() and len(w) > 2]
        if all_caps_words:
            signals.append({
                "category": "all_caps_words",
                "weight": 0.05 * len(all_caps_words),
                "text": " ".join(all_caps_words[:5]),
                "label": SIGNAL_LABELS["all_caps_words"],
                "reason": SIGNAL_REASONS["all_caps_words"],
            })

        signals.sort(key=lambda signal: (signal["weight"], len(signal["text"])), reverse=True)
        return signals
    
    @staticmethod
    def find_suspicious_phrases(text: str) -> List[str]:
        """Find suspicious phrases in the text"""
        phrases = []
        for signal in ExplainabilityEngine.analyze_signals(text):
            phrase = signal["text"]
            if phrase and phrase not in phrases:
                phrases.append(phrase)
        
        return phrases[:5]  # Limit to top 5

    @staticmethod
    def select_primary_signal(text: str, suspicious_phrases: List[str]) -> Dict[str, str]:
        """Pick the strongest signal to show as evidence."""
        signals = ExplainabilityEngine.analyze_signals(text)
        if signals:
            primary_signal = signals[0]
            return {
                "text": primary_signal["text"],
                "label": primary_signal["label"],
                "reason": primary_signal["reason"],
            }

        cleaned_text = " ".join(text.split())
        fallback_text = cleaned_text[:140] + ("..." if len(cleaned_text) > 140 else "") if cleaned_text else ""
        return {
            "text": fallback_text,
            "label": SIGNAL_LABELS["fallback_context"],
            "reason": SIGNAL_REASONS["fallback_context"],
        }
    
    @staticmethod
    def generate_explanation(prediction: str, confidence: float, 
                            suspicious_phrases: List[str]) -> str:
        """Generate human-readable explanation"""
        
        if prediction == "fake":
            if confidence > 0.8:
                base = "This review shows strong indicators of being fake."
            elif confidence > 0.6:
                base = "This review appears suspicious and may be fake."
            else:
                base = "This review has some questionable elements."
            
            if suspicious_phrases:
                base += f" Found {len(suspicious_phrases)} suspicious pattern(s)."
            
        else:
            if confidence > 0.8:
                base = "This review appears to be genuine and authentic."
            elif confidence > 0.6:
                base = "This review seems legitimate with natural language patterns."
            else:
                base = "This review appears mostly genuine, but has minor quirks."
        
        return base


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("[FastAPI] Starting up...")
    review_db.initialize()
    model_manager.load_model()
    print("[FastAPI] Ready to accept requests")
    yield
    # Shutdown
    print("[FastAPI] Shutting down...")


app = FastAPI(
    title="Fake Review Detector API",
    description="AI-powered fake review detection using BERT NLP model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Fake Review Detector API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        model_loaded=model_manager.model_loaded,
        model_name=os.path.abspath(MODEL_DIR) if model_manager.model_loaded else "rule-based-fallback"
    )


@app.post("/analyze", response_model=ReviewResponse)
async def analyze_review(request: ReviewRequest):
    """
    Analyze a review text and determine if it's fake or genuine
    
    - **text**: The review text to analyze (5-2000 characters)
    
    Returns prediction, confidence score, suspicious phrases, and explanation
    """
    start_time = time.time()
    
    try:
        # Validate input
        text = request.text.strip()
        if len(text) < MIN_TEXT_LENGTH:
            raise HTTPException(
                status_code=400, 
                detail=f"Text too short. Minimum {MIN_TEXT_LENGTH} characters required."
            )
        
        # Get prediction
        prediction, confidence = model_manager.predict(text)
        
        # Find suspicious phrases
        suspicious_phrases = ExplainabilityEngine.find_suspicious_phrases(text)
        primary_signal = ExplainabilityEngine.select_primary_signal(text, suspicious_phrases)
        
        # Generate explanation
        explanation = ExplainabilityEngine.generate_explanation(
            prediction, confidence, suspicious_phrases
        )
        
        processing_time = time.time() - start_time
        model_used = os.path.abspath(MODEL_DIR) if model_manager.model_loaded else "rule-based"
        analysis_id = review_db.insert_analysis(
            review_text=text,
            prediction=prediction,
            confidence=round(confidence, 4),
            evidence_text=primary_signal["text"],
            evidence_label=primary_signal["label"],
            evidence_reason=primary_signal["reason"],
            suspicious_phrases=suspicious_phrases,
            explanation=explanation,
            model_used=model_used,
        )
        
        return ReviewResponse(
            analysis_id=analysis_id,
            prediction=prediction,
            confidence=round(confidence, 4),
            review_text=text[:500] + "..." if len(text) > 500 else text,
            evidence_text=primary_signal["text"],
            evidence_label=primary_signal["label"],
            evidence_reason=primary_signal["reason"],
            suspicious_phrases=suspicious_phrases,
            explanation=explanation,
            processing_time=round(processing_time, 3),
            model_used=model_used
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error analyzing review: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/batch-analyze")
async def batch_analyze(reviews: List[ReviewRequest]):
    """
    Analyze multiple reviews in batch
    
    Returns list of analysis results
    """
    results = []
    
    for review in reviews:
        try:
            result = await analyze_review(review)
            results.append(result)
        except Exception as e:
            results.append({
                "error": str(e),
                "review_text": review.text[:100]
            })
    
    return {"results": results, "count": len(results)}


@app.post("/feedback", response_model=FeedbackResponse)
async def save_feedback(request: FeedbackRequest):
    """
    Save a human-reviewed label for an analyzed review.

    Use this to curate future retraining data from real extension traffic.
    """
    normalized_label = request.human_label.strip().lower()
    if normalized_label not in {"fake", "genuine", "unknown"}:
        raise HTTPException(status_code=400, detail="human_label must be 'fake', 'genuine', or 'unknown'")

    try:
        review_db.update_feedback(
            analysis_id=request.analysis_id,
            human_label=normalized_label,
            notes=request.notes.strip(),
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return FeedbackResponse(
        success=True,
        analysis_id=request.analysis_id,
        human_label=normalized_label,
    )


@app.get("/reviews/recent", response_model=List[ReviewRecord])
async def get_recent_reviews(limit: int = 20):
    """
    Return the most recent stored review analyses.

    Useful for manual review, labeling, and building future retraining data.
    """
    safe_limit = max(1, min(limit, 100))
    return review_db.get_recent_reviews(limit=safe_limit)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"[Main] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
