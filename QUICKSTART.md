# Fake Review Detector - Quick Start Guide

## For Diploma Defense Demo

This guide will help you quickly set up and demonstrate the Fake Review Detection system.

---

## Prerequisites

- Python 3.8+ installed
- Google Chrome browser
- Terminal/Command Prompt access

---

## Step 1: Start the Backend (Terminal 1)

```bash
# Navigate to project folder
cd fake-review-detector

# Run setup (first time only)
./setup.sh          # macOS/Linux
# OR
setup.bat           # Windows

# Start the server
cd server
source venv/bin/activate    # macOS/Linux
# OR
venv\Scripts\activate.bat   # Windows

python main.py
```

**Expected output:**
```
[FastAPI] Starting up...
[ModelManager] Loading model: distilbert-base-uncased-finetuned-sst-2-english
[ModelManager] Model loaded successfully
[FastAPI] Ready to accept requests
INFO:     Uvicorn running on http://0.0.0.0:8000
```

**Verify it's working:**
Open http://localhost:8000/docs in your browser

---

## Step 2: Install Chrome Extension

1. Open Chrome → Go to `chrome://extensions/`
2. Toggle ON "Developer mode" (top right)
3. Click "Load unpacked"
4. Select the `fake-review-detector/client` folder
5. Extension icon appears in toolbar

---

## Step 3: Test the System

### Option A: Live Demo on Real Website

1. Go to any website with reviews (Amazon, Yelp, etc.)
2. Highlight a review text
3. Click "Check Review" button
4. See results in popup

### Option B: API Test (Terminal 2)

```bash
cd fake-review-detector/server
source venv/bin/activate
python test_api.py
```

**Expected output:**
```
Testing LIKELY FAKE Reviews
  Prediction: FAKE (confidence: 85%)
  Suspicious phrases: 3 found

Testing LIKELY GENUINE Reviews
  Prediction: GENUINE (confidence: 78%)
```

### Option C: Interactive Test

```bash
python test_api.py --interactive
```

Then type your own reviews to test!

---

## Demo Script for Defense

### 1. Introduction (1 minute)

"This is my diploma project: an AI-powered fake review detection system. It consists of a Chrome browser extension frontend and a Python FastAPI backend using BERT NLP model."

### 2. Show Architecture (1 minute)

```
User selects text → Chrome Extension → FastAPI API → BERT Model → Results
```

**Key files:**
- `client/content.js` - Text selection & floating button
- `client/popup.js` - Results display
- `server/main.py` - API & model inference

### 3. Live Demo (3 minutes)

1. **Show backend running**
   - Terminal with server logs
   - API docs at `/docs`

2. **Show extension installed**
   - chrome://extensions page
   - Click extension icon

3. **Test on real website**
   - Go to Amazon/Yelp
   - Highlight a suspicious review
   - Click "Check Review"
   - Explain the results:
     - Prediction (fake/genuine)
     - Confidence score
     - Suspicious phrases highlighted
     - AI explanation

### 4. Technical Highlights (2 minutes)

**Explainability features:**
- Suspicious pattern detection (regex rules)
- Keyword highlighting in popup
- Human-readable explanations

**Performance:**
- Response time < 2 seconds
- Model caching for speed
- Rule-based fallback if ML fails

**Privacy:**
- No data storage
- HTTPS-ready structure
- Minimal permissions

### 5. Code Walkthrough (2 minutes)

Show key code sections:

**Backend - `/analyze` endpoint:**
```python
@app.post("/analyze")
async def analyze_review(request: ReviewRequest):
    prediction, confidence = model_manager.predict(text)
    suspicious_phrases = ExplainabilityEngine.find_suspicious_phrases(text)
    return ReviewResponse(...)
```

**Frontend - Text selection:**
```javascript
document.addEventListener('mouseup', handleTextSelection);
function showFloatingButton(selection) {
    // Creates button near selected text
}
```

---

## Common Questions & Answers

**Q: What model do you use?**
A: DistilBERT (lightweight BERT) fine-tuned on sentiment analysis. Falls back to rule-based if model fails.

**Q: How accurate is it?**
A: For demo purposes, ~70-80% accuracy. Real-world systems use larger datasets and custom fine-tuning.

**Q: How does explainability work?**
A: We use regex patterns to detect suspicious phrases (excessive praise, generic templates, marketing language) and highlight them.

**Q: Is it production-ready?**
A: This is an MVP. Production would need HTTPS, authentication, rate limiting, and a larger training dataset.

**Q: What about privacy?**
A: No data is stored. Text is processed in-memory and immediately discarded.

---

## Troubleshooting During Demo

### Issue: Server won't start
**Fix:** Check Python version (need 3.8+)
```bash
python3 --version
```

### Issue: Extension not working
**Fix:** Refresh the webpage after installing extension

### Issue: "Cannot connect to server"
**Fix:** Check if server is running on port 8000
```bash
curl http://localhost:8000/health
```

### Issue: Model loading slow
**Fix:** First request is slow. Pre-warm by running test_api.py first.

---

## Quick Commands Reference

```bash
# Start server
cd server && source venv/bin/activate && python main.py

# Test API
cd server && python test_api.py

# Interactive test
cd server && python test_api.py --interactive

# Health check
curl http://localhost:8000/health

# Manual API call
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing!"}'
```

---

## Project Files Summary

| File | Purpose |
|------|---------|
| `client/manifest.json` | Extension config (Manifest V3) |
| `client/content.js` | Text selection & floating button |
| `client/popup.js` | Results display logic |
| `client/background.js` | Service worker |
| `server/main.py` | FastAPI app & BERT model |
| `server/test_api.py` | API testing script |

---

## Success Criteria for Defense

✅ Backend starts without errors  
✅ Extension installs successfully  
✅ Text selection triggers floating button  
✅ API returns prediction & confidence  
✅ Suspicious phrases are highlighted  
✅ Response time < 2 seconds  
✅ Code is clean and well-documented  

---

**Good luck with your defense! 🎓**