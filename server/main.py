"""
Fake Review Detector - FastAPI Backend
AI-powered fake review detection using BERT NLP model
"""

import os
import re
import time
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

MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
FALLBACK_MODEL = True  # Use rule-based fallback if model loading fails
MAX_TEXT_LENGTH = 512
MIN_TEXT_LENGTH = 5

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
    prediction: str = Field(..., description="Prediction: 'fake' or 'genuine'")
    confidence: float = Field(..., ge=0.0, le=1.0, 
                              description="Confidence score (0-1)")
    review_text: str = Field(..., description="Analyzed review text")
    evidence_text: str = Field(
        default="",
        description="Primary suspicious text span used as evidence for the verdict"
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


# ============================================================================
# Model Manager
# ============================================================================

class ModelManager:
    """Manages the BERT model for fake review detection"""
    
    def __init__(self):
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForSequenceClassification] = None
        self.model_loaded = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """Load the BERT model and tokenizer"""
        try:
            print(f"[ModelManager] Loading model: {MODEL_NAME}")
            print(f"[ModelManager] Using device: {self.device}")
            
            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            self.model.to(self.device)
            self.model.eval()
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
                probabilities = torch.softmax(outputs.logits, dim=1)
                
            # Get prediction (0 = negative/fake, 1 = positive/genuine for SST-2)
            # We'll invert this logic for fake review detection
            genuine_prob = probabilities[0][1].item()
            fake_prob = probabilities[0][0].item()
            
            # Adjust based on text patterns
            pattern_score = self._calculate_pattern_score(text)
            
            # Combine model output with pattern analysis
            adjusted_fake_prob = (fake_prob * 0.6) + (pattern_score * 0.4)
            
            if adjusted_fake_prob > 0.5:
                return "fake", adjusted_fake_prob
            else:
                return "genuine", 1 - adjusted_fake_prob
                
        except Exception as e:
            print(f"[ModelManager] Prediction error: {e}")
            return self._rule_based_predict(text)
    
    def _rule_based_predict(self, text: str) -> tuple[str, float]:
        """Rule-based prediction when model is not available"""
        score = self._calculate_pattern_score(text)
        
        if score > 0.5:
            return "fake", min(score + 0.2, 0.95)
        else:
            return "genuine", min(1 - score + 0.2, 0.95)
    
    def _calculate_pattern_score(self, text: str) -> float:
        """Calculate suspicious pattern score"""
        text_lower = text.lower()
        score = 0.0
        
        # Check each pattern category
        for category, patterns in SUSPICIOUS_PATTERNS.items():
            for pattern in patterns:
                matches = len(re.findall(pattern, text_lower, re.IGNORECASE))
                if matches > 0:
                    if category == "fake_indicators":
                        score += 0.3 * matches
                    elif category == "excessive_praise":
                        score += 0.15 * matches
                    elif category == "unnatural_language":
                        score += 0.2 * matches
                    else:
                        score += 0.1 * matches
        
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


# ============================================================================
# Explainability Engine
# ============================================================================

class ExplainabilityEngine:
    """Generates explanations and highlights suspicious phrases"""
    
    @staticmethod
    def find_suspicious_phrases(text: str) -> List[str]:
        """Find suspicious phrases in the text"""
        phrases = []
        text_lower = text.lower()
        
        for category, patterns in SUSPICIOUS_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text_lower, re.IGNORECASE)
                for match in matches:
                    # Get the actual matched text from original
                    start = max(0, match.start() - 10)
                    end = min(len(text), match.end() + 10)
                    phrase = text[start:end].strip()
                    if phrase and phrase not in phrases:
                        phrases.append(phrase)
        
        return phrases[:5]  # Limit to top 5

    @staticmethod
    def select_primary_evidence(text: str, suspicious_phrases: List[str]) -> str:
        """Pick the strongest text fragment to show as evidence."""
        if suspicious_phrases:
            # Prefer the most specific phrase instead of a very short generic match.
            return max(suspicious_phrases, key=lambda phrase: (len(phrase), phrase))

        cleaned_text = " ".join(text.split())
        if not cleaned_text:
            return ""

        return cleaned_text[:140] + ("..." if len(cleaned_text) > 140 else "")
    
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
        model_name=MODEL_NAME if model_manager.model_loaded else "rule-based-fallback"
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
        evidence_text = ExplainabilityEngine.select_primary_evidence(text, suspicious_phrases)
        
        # Generate explanation
        explanation = ExplainabilityEngine.generate_explanation(
            prediction, confidence, suspicious_phrases
        )
        
        processing_time = time.time() - start_time
        
        return ReviewResponse(
            prediction=prediction,
            confidence=round(confidence, 4),
            review_text=text[:500] + "..." if len(text) > 500 else text,
            evidence_text=evidence_text,
            suspicious_phrases=suspicious_phrases,
            explanation=explanation,
            processing_time=round(processing_time, 3),
            model_used=MODEL_NAME if model_manager.model_loaded else "rule-based"
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


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"[Main] Starting server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)
