/**
 * Fake Review Detector - Popup Script
 * Controls the extension toggle and shows the latest analysis result.
 */

const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINT = `${API_BASE_URL}/analyze`;

const elements = {
  instructions: document.getElementById('instructions'),
  loadingState: document.getElementById('loadingState'),
  errorState: document.getElementById('errorState'),
  resultState: document.getElementById('resultState'),
  statusIndicator: document.getElementById('statusIndicator'),
  extensionToggle: document.getElementById('extensionToggle'),
  errorMessage: document.getElementById('errorMessage'),
  retryBtn: document.getElementById('retryBtn'),
  verdictBadge: document.getElementById('verdictBadge'),
  verdictIcon: document.getElementById('verdictIcon'),
  verdictText: document.getElementById('verdictText'),
  confidenceValue: document.getElementById('confidenceValue'),
  progressBar: document.getElementById('progressBar'),
  reviewText: document.getElementById('reviewText'),
  suspiciousSection: document.getElementById('suspiciousSection'),
  suspiciousList: document.getElementById('suspiciousList'),
  evidenceText: document.getElementById('evidenceText'),
  explanationText: document.getElementById('explanationText'),
  newCheckBtn: document.getElementById('newCheckBtn'),
};

let currentReviewText = '';

document.addEventListener('DOMContentLoaded', async () => {
  const stored = await chrome.storage.local.get({
    analysisResult: null,
    pendingReview: '',
    extensionEnabled: true,
  });

  elements.extensionToggle.checked = stored.extensionEnabled !== false;
  setupEventListeners();
  checkApiHealth();

  if (stored.analysisResult) {
    showResult(stored.analysisResult);
    return;
  }

  if (stored.pendingReview) {
    currentReviewText = stored.pendingReview;
    await analyzeReview(currentReviewText);
    await chrome.storage.local.remove('pendingReview');
    return;
  }

  const inlineResult = await getCurrentInlineResult();
  if (inlineResult?.result) {
    showResult(inlineResult.result);
    return;
  }

  showInstructions();
});

function setupEventListeners() {
  elements.extensionToggle.addEventListener('change', async () => {
    const enabled = elements.extensionToggle.checked;
    await chrome.storage.local.set({ extensionEnabled: enabled });

    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, {
        action: enabled ? 'ping' : 'resetSelection',
      });
    }

    if (!enabled) {
      showInstructions();
    }
  });

  elements.retryBtn.addEventListener('click', () => {
    if (currentReviewText) {
      analyzeReview(currentReviewText);
    } else {
      showInstructions();
    }
  });

  elements.newCheckBtn.addEventListener('click', async () => {
    currentReviewText = '';
    await chrome.storage.local.remove(['analysisResult', 'pendingReview']);
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    if (tabs[0]?.id) {
      chrome.tabs.sendMessage(tabs[0].id, { action: 'resetSelection' });
    }
    showInstructions();
  });
}

async function getCurrentInlineResult() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const activeTab = tabs[0];
    if (!activeTab?.id) return null;
    return await chrome.tabs.sendMessage(activeTab.id, { action: 'getInlineResult' });
  } catch (error) {
    console.warn('[Fake Review Detector] Could not get inline result:', error);
    return null;
  }
}

async function getCurrentTabSelection() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const activeTab = tabs[0];
    if (!activeTab?.id) return '';
    const response = await chrome.tabs.sendMessage(activeTab.id, { action: 'getSelection' });
    return response?.text?.trim() || '';
  } catch (error) {
    return '';
  }
}

async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' },
    });
    updateStatus(response.ok ? 'online' : 'offline');
  } catch (error) {
    updateStatus('offline');
  }
}

function updateStatus(status) {
  const dot = elements.statusIndicator.querySelector('.status-dot');
  const text = elements.statusIndicator.querySelector('.status-text');

  if (status === 'online') {
    dot.style.background = '#4ade80';
    text.textContent = 'Online';
  } else {
    dot.style.background = '#ef4444';
    text.textContent = 'Offline';
  }
}

function showInstructions() {
  hideAllStates();
  elements.instructions.style.display = 'block';
}

function showLoading() {
  hideAllStates();
  elements.loadingState.style.display = 'flex';
}

function showError(message) {
  hideAllStates();
  elements.errorState.style.display = 'flex';
  elements.errorMessage.textContent = message || 'Something went wrong. Please try again.';
}

function showResult(result) {
  hideAllStates();
  elements.resultState.style.display = 'block';

  currentReviewText = result.review_text || '';

  const isFake = result.prediction === 'fake';
  const confidence = Math.round((result.confidence || 0) * 100);
  const confidenceLabel = getConfidenceLabel(confidence);

  elements.verdictBadge.className = `verdict-badge ${isFake ? 'fake' : 'genuine'}`;
  elements.verdictIcon.textContent = isFake ? '!' : 'OK';
  elements.verdictText.textContent = isFake ? 'Potentially Fake' : 'Likely Original';
  elements.confidenceValue.textContent = `${confidenceLabel} (${confidence}%)`;
  elements.progressBar.className = `progress-bar ${isFake ? 'fake' : 'genuine'}`;
  elements.progressBar.style.width = `${confidence}%`;
  elements.reviewText.textContent = result.review_text || 'No text provided';
  elements.evidenceText.textContent = result.evidence_text || 'No evidence text provided.';
  elements.explanationText.textContent =
    result.explanation ||
    (isFake
      ? 'This review contains patterns commonly found in fake reviews.'
      : 'This review appears to be genuine based on the text patterns.');

  if (result.suspicious_phrases && result.suspicious_phrases.length > 0) {
    elements.suspiciousSection.style.display = 'block';
    elements.suspiciousList.innerHTML = result.suspicious_phrases
      .map((phrase) => `<li>${escapeHtml(phrase)}</li>`)
      .join('');
  } else {
    elements.suspiciousSection.style.display = 'none';
    elements.suspiciousList.innerHTML = '';
  }
}

function getConfidenceLabel(confidence) {
  if (confidence >= 85) return 'High';
  if (confidence >= 65) return 'Moderate';
  return 'Low';
}

function hideAllStates() {
  elements.instructions.style.display = 'none';
  elements.loadingState.style.display = 'none';
  elements.errorState.style.display = 'none';
  elements.resultState.style.display = 'none';
}

async function analyzeReview(text) {
  const reviewText = (text || '').trim() || (await getCurrentTabSelection());
  if (!reviewText) {
    showError('No review text selected. Please highlight a review and try again.');
    return;
  }

  currentReviewText = reviewText;
  showLoading();

  try {
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: reviewText }),
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }

    const result = await response.json();
    await chrome.storage.local.set({ analysisResult: result });
    showResult(result);
  } catch (error) {
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      showError('Cannot connect to the server. Make sure the backend is running on localhost:8000.');
    } else {
      showError(error.message || 'Failed to analyze review.');
    }
  }
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message.action === 'analyzeText') {
    currentReviewText = message.text;
    analyzeReview(currentReviewText);
    sendResponse({ success: true });
  }

  return true;
});
