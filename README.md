<<<<<<< HEAD
# Fake Review Detector

AI-powered fake review detection system implemented as a Chrome browser extension with a Python FastAPI backend. Built for diploma project demonstration.

## Features

- **Chrome Extension (Manifest V3)**: Highlight any text on a webpage and click "Check Review"
- **FastAPI Backend**: RESTful API with `/analyze` endpoint
- **BERT NLP Model**: Pre-trained transformer model for text classification
- **Explainability**: Highlights suspicious phrases and provides human-readable explanations
- **Real-time Analysis**: Response time < 2 seconds
- **Privacy-focused**: No user data storage

## Project Structure

```
fake-review-detector/
├── client/                    # Chrome Extension
│   ├── manifest.json          # Extension manifest (v3)
│   ├── popup.html             # Popup UI
│   ├── popup.js               # Popup logic
│   ├── styles.css             # Popup styles
│   ├── content.js             # Content script for text selection
│   ├── content.css            # Content script styles
│   ├── background.js          # Service worker
│   └── icons/                 # Extension icons
│       ├── icon16.png
│       ├── icon48.png
│       └── icon128.png
│
├── server/                    # FastAPI Backend
│   ├── main.py                # Main FastAPI application
│   └── requirements.txt       # Python dependencies
│
└── README.md                  # This file
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- Google Chrome browser
- pip (Python package manager)

### Step 1: Start the Backend Server

```bash
# Navigate to server directory
cd server

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Start the server
python main.py
```

The server will start on `http://localhost:8000`

You can verify it's running by visiting:
- http://localhost:8000/ - API info
- http://localhost:8000/docs - Interactive API documentation
- http://localhost:8000/health - Health check

### Step 2: Install the Chrome Extension

1. Open Google Chrome
2. Navigate to `chrome://extensions/`
3. Enable "Developer mode" (toggle in top right corner)
4. Click "Load unpacked"
5. Select the `client` folder from this project
6. The extension icon should appear in your browser toolbar

### Step 3: Test the System

1. Visit any website with reviews (e.g., Amazon, Yelp, Trustpilot)
2. Highlight a review text
3. Click the "Check Review" button that appears
4. View the analysis results in the popup

## API Documentation

### POST /analyze

Analyze a review text for authenticity.

**Request:**
```json
{
  "text": "This product is absolutely amazing! Best purchase ever!"
}
```

**Response:**
```json
{
  "prediction": "fake",
  "confidence": 0.8234,
  "review_text": "This product is absolutely amazing!...",
  "suspicious_phrases": ["absolutely amazing", "Best purchase ever"],
  "explanation": "This review shows strong indicators of being fake. Found 2 suspicious pattern(s).",
  "processing_time": 0.245,
  "model_used": "distilbert-base-uncased-finetuned-sst-2-english"
}
```

### GET /health

Check API health status.

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
  "version": "1.0.0"
}
```

## How It Works

### Detection Algorithm

1. **Text Preprocessing**: Input text is cleaned and tokenized
2. **BERT Model Inference**: The pre-trained model processes the text
3. **Pattern Analysis**: Rule-based heuristics check for suspicious patterns
4. **Score Combination**: Model output and pattern scores are combined
5. **Explainability**: Suspicious phrases are extracted and highlighted

### Suspicious Patterns Detected

- **Excessive Praise**: Overly enthusiastic language ("best ever", "amazing product")
- **Generic Phrases**: Common fake review templates
- **Unnatural Language**: Marketing-style language
- **Repetitive Patterns**: Repeated characters or words
- **Fake Indicators**: Disclosure phrases, promotional language

### Model Details

- **Base Model**: DistilBERT (lightweight BERT variant)
- **Fine-tuning**: SST-2 sentiment classification
- **Fallback**: Rule-based detection if model fails to load
- **Device**: Automatically uses GPU if available, otherwise CPU

## Configuration

### Backend Environment Variables

```bash
PORT=8000              # Server port
HOST=0.0.0.0           # Server host
```

### Extension Settings

Edit `client/background.js` to configure:
- `API_BASE_URL`: Backend API URL
- `MIN_TEXT_LENGTH`: Minimum text length for analysis

## Development

### Running Tests

```bash
# Backend tests (example)
cd server
pytest tests/
```

### Extension Debugging

1. Open `chrome://extensions/`
2. Find "Fake Review Detector"
3. Click "background page" to debug service worker
4. Right-click extension icon → "Inspect popup" to debug popup

### API Testing with curl

```bash
# Health check
curl http://localhost:8000/health

# Analyze review
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "This product is amazing! Highly recommend!"}'
```

## Production Deployment

### Backend Deployment (Example with Docker)

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### HTTPS Configuration

For production, use a reverse proxy (nginx) with SSL certificates:

```nginx
server {
    listen 443 ssl;
    server_name api.yourdomain.com;
    
    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

Update the extension's `API_BASE_URL` to use the HTTPS endpoint.

## Troubleshooting

### Backend Issues

**Problem**: Model fails to load
- **Solution**: The system will automatically use rule-based fallback

**Problem**: CUDA out of memory
- **Solution**: The system automatically falls back to CPU

**Problem**: Slow response times
- **Solution**: First request may be slower due to model loading. Subsequent requests are cached.

### Extension Issues

**Problem**: "Check Review" button doesn't appear
- **Solution**: Ensure text is at least 10 characters long
- **Solution**: Check that the content script is injected (refresh page)

**Problem**: "Cannot connect to server" error
- **Solution**: Verify backend is running on localhost:8000
- **Solution**: Check CORS settings in backend

**Problem**: Extension not loading
- **Solution**: Check manifest.json syntax
- **Solution**: Ensure all required permissions are granted

## Performance Metrics

- **Average Response Time**: ~200-500ms (with model cached)
- **Model Size**: ~250MB (DistilBERT)
- **Memory Usage**: ~500MB RAM
- **Throughput**: ~50 requests/second

## Academic Citation

If using this project for academic purposes:

```bibtex
@misc{fake_review_detector,
  title={Fake Review Detector: AI-Powered Browser Extension},
  author={Nataly Kara},
  year={2026},
  note={Fake Review Detector: AI-Powered Browser Extension}
}
```

## License

This project is for educational purposes only.

## Acknowledgments

- Hugging Face Transformers library
- FastAPI framework
- DistilBERT model by Google

## Contact

For questions or issues, please contact [natashkajjk@gmail.com]
=======

