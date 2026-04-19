/**
 * Fake Review Detector - Popup Script
 * Handles UI state and communication with background script
 */

// API Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_ENDPOINT = `${API_BASE_URL}/analyze`;

// DOM Elements
const elements = {
  instructions: document.getElementById('instructions'),
  loadingState: document.getElementById('loadingState'),
  errorState: document.getElementById('errorState'),
  resultState: document.getElementById('resultState'),
  statusIndicator: document.getElementById('statusIndicator'),
  
  // Error elements
  errorMessage: document.getElementById('errorMessage'),
  retryBtn: document.getElementById('retryBtn'),
  
  // Result elements
  verdictBadge: document.getElementById('verdictBadge'),
  verdictIcon: document.getElementById('verdictIcon'),
  verdictText: document.getElementById('verdictText'),
  confidenceValue: document.getElementById('confidenceValue'),
  progressBar: document.getElementById('progressBar'),
  reviewText: document.getElementById('reviewText'),
  suspiciousSection: document.getElementById('suspiciousSection'),
  suspiciousList: document.getElementById('suspiciousList'),
  explanationText: document.getElementById('explanationText'),
  newCheckBtn: document.getElementById('newCheckBtn')
};

// State management
let currentReviewText = '';

/**
 * Initialize popup
 */
document.addEventListener('DOMContentLoaded', async () => {
  console.log('[Fake Review Detector] Popup opened');
  
  // Check if we have pending analysis results
  const result = await chrome.storage.local.get(['analysisResult', 'pendingReview']);
  
  if (result.analysisResult) {
    // Show results
    showResult(result.analysisResult);
    // Clear the stored result
    await chrome.storage.local.remove('analysisResult');
  } else if (result.pendingReview) {
    // Analyze the pending review
    currentReviewText = result.pendingReview;
    await analyzeReview(currentReviewText);
    await chrome.storage.local.remove('pendingReview');
  } else {
    // Try to get the current selection directly from the content script
    const selectedText = await getCurrentTabSelection();
    if (selectedText) {
      currentReviewText = selectedText;
      await analyzeReview(currentReviewText);
    } else {
      // Show instructions
      showInstructions();
    }
  }
  
  // Setup event listeners
  setupEventListeners();
  
  // Check API health
  checkApiHealth();
});

/**
 * Setup event listeners
 */
function setupEventListeners() {
  elements.retryBtn.addEventListener('click', () => {
    if (currentReviewText) {
      analyzeReview(currentReviewText);
    } else {
      showInstructions();
    }
  });
  
  elements.newCheckBtn.addEventListener('click', () => {
    showInstructions();
    // Notify content script to reset
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0]) {
        chrome.tabs.sendMessage(tabs[0].id, { action: 'resetSelection' });
      }
    });
  });
}

/**
 * Request the current selection from the active tab.
 */
async function getCurrentTabSelection() {
  try {
    const tabs = await chrome.tabs.query({ active: true, currentWindow: true });
    const activeTab = tabs[0];

    if (!activeTab || !activeTab.id) {
      return '';
    }

    const response = await chrome.tabs.sendMessage(activeTab.id, { action: 'getSelection' });
    return response?.text?.trim() || '';
  } catch (error) {
    console.warn('[Fake Review Detector] Could not get selection from content script:', error);
    return '';
  }
}

/**
 * Check API health status
 */
async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    if (response.ok) {
      updateStatus('online');
    } else {
      updateStatus('offline');
    }
  } catch (error) {
    console.warn('[Fake Review Detector] API health check failed:', error);
    updateStatus('offline');
  }
}

/**
 * Update status indicator
 */
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

/**
 * Show instructions view
 */
function showInstructions() {
  hideAllStates();
  elements.instructions.style.display = 'block';
}

/**
 * Show loading state
 */
function showLoading() {
  hideAllStates();
  elements.loadingState.style.display = 'flex';
}

/**
 * Show error state
 */
function showError(message) {
  hideAllStates();
  elements.errorState.style.display = 'flex';
  elements.errorMessage.textContent = message || 'Something went wrong. Please try again.';
}

/**
 * Show result state
 */
function showResult(result) {
  hideAllStates();
  elements.resultState.style.display = 'block';
  
  // Store current review text
  currentReviewText = result.review_text || '';
  
  // Update verdict
  const isFake = result.prediction === 'fake';
  const confidence = Math.round(result.confidence * 100);
  
  elements.verdictBadge.className = `verdict-badge ${isFake ? 'fake' : 'genuine'}`;
  elements.verdictIcon.textContent = isFake ? '⚠' : '✓';
  elements.verdictText.textContent = isFake ? 'Potentially Fake' : 'Genuine Review';
  
  // Update confidence
  elements.confidenceValue.textContent = `${confidence}%`;
  
  // Update progress bar
  elements.progressBar.className = `progress-bar ${isFake ? 'fake' : 'genuine'}`;
  elements.progressBar.style.width = `${confidence}%`;
  
  // Update review text
  elements.reviewText.textContent = result.review_text || 'No text provided';
  
  // Update suspicious phrases
  if (result.suspicious_phrases && result.suspicious_phrases.length > 0) {
    elements.suspiciousSection.style.display = 'block';
    elements.suspiciousList.innerHTML = result.suspicious_phrases
      .map(phrase => `<li>${escapeHtml(phrase)}</li>`)
      .join('');
  } else {
    elements.suspiciousSection.style.display = 'none';
  }
  
  // Update explanation
  elements.explanationText.textContent = result.explanation || 
    (isFake 
      ? 'This review contains patterns commonly found in fake reviews, such as overly promotional language or suspicious phrasing.' 
      : 'This review appears to be genuine based on its natural language patterns and authentic tone.');
}

/**
 * Hide all state views
 */
function hideAllStates() {
  elements.instructions.style.display = 'none';
  elements.loadingState.style.display = 'none';
  elements.errorState.style.display = 'none';
  elements.resultState.style.display = 'none';
}

/**
 * Analyze review text via API
 */
async function analyzeReview(text) {
  if (!text || text.trim().length === 0) {
    showError('No review text selected. Please highlight a review and try again.');
    return;
  }
  
  showLoading();
  
  try {
    console.log('[Fake Review Detector] Analyzing review:', text.substring(0, 50) + '...');
    
    const response = await fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: text.trim() })
    });
    
    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      throw new Error(errorData.detail || `Server error: ${response.status}`);
    }
    
    const result = await response.json();
    console.log('[Fake Review Detector] Analysis result:', result);
    
    // Show results
    showResult(result);
    
  } catch (error) {
    console.error('[Fake Review Detector] Analysis error:', error);
    
    if (error.message.includes('Failed to fetch') || error.message.includes('NetworkError')) {
      showError('Cannot connect to the server. Please make sure the backend is running on localhost:8000');
    } else {
      showError(error.message || 'Failed to analyze review. Please try again.');
    }
  }
}

/**
 * Escape HTML to prevent XSS
 */
function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// Listen for messages from content script
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[Fake Review Detector] Message received in popup:', message);
  
  if (message.action === 'analyzeText') {
    currentReviewText = message.text;
    analyzeReview(currentReviewText);
    sendResponse({ success: true });
  }
  
  return true;
});
