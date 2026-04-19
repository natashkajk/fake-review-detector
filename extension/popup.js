/**
 * Fake Review Detector - Popup Script
 * ====================================
 * Handles API communication, UI state transitions, and result display.
 *
 * Architecture:
 *   - Receives: { action: "analyzeText", text: "..." } from content.js (via runtime)
 *   - Sends:    { action: "highlightSuspicious", phrases: [...] } to background.js
 *
 * Constraints:
 *   - Does NOT use chrome.storage
 *   - Does NOT depend on background.js (only uses it as a relay)
 *   - Works only through runtime messages
 */

(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // Configuration
  // ---------------------------------------------------------------------------

  const API_BASE_URL = 'http://localhost:8000';
  const API_ANALYZE = `${API_BASE_URL}/analyze`;
  const API_HEALTH = `${API_BASE_URL}/health`;

  // ---------------------------------------------------------------------------
  // DOM References
  // ---------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);

  const els = {
    // States
    stateInitial: $('state-initial'),
    stateLoading: $('state-loading'),
    stateError: $('state-error'),
    stateResult: $('state-result'),

    // Initial state
    manualInput: $('manual-input'),
    btnAnalyze: $('btn-analyze'),

    // Error state
    errorMessage: $('error-message'),
    btnRetry: $('btn-retry'),

    // Result state
    resultBadge: $('result-badge'),
    resultIcon: $('result-icon'),
    resultLabel: $('result-label'),
    confidenceValue: $('confidence-value'),
    confidenceBar: $('confidence-bar'),
    phrasesSection: $('phrases-section'),
    phrasesList: $('phrases-list'),
    btnClear: $('btn-clear'),
    btnNew: $('btn-new'),

    // Footer
    apiStatus: $('api-status'),
  };

  // ---------------------------------------------------------------------------
  // State Machine
  // ---------------------------------------------------------------------------

  function showState(stateName) {
    // Hide all states
    els.stateInitial.classList.add('hidden');
    els.stateLoading.classList.add('hidden');
    els.stateError.classList.add('hidden');
    els.stateResult.classList.add('hidden');

    // Show requested state
    switch (stateName) {
      case 'initial':
        els.stateInitial.classList.remove('hidden');
        break;
      case 'loading':
        els.stateLoading.classList.remove('hidden');
        break;
      case 'error':
        els.stateError.classList.remove('hidden');
        break;
      case 'result':
        els.stateResult.classList.remove('hidden');
        break;
    }
  }

  // ---------------------------------------------------------------------------
  // API Communication
  // ---------------------------------------------------------------------------

  /**
   * Check if the API server is reachable.
   */
  async function checkApiHealth() {
    try {
      const response = await fetch(API_HEALTH, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      if (response.ok) {
        const data = await response.json();
        els.apiStatus.textContent = `API: ${data.status} (${data.device})`;
        els.apiStatus.className = 'footer-status status-online';
        return true;
      }
    } catch {
      // Network error
    }
    els.apiStatus.textContent = 'API: offline';
    els.apiStatus.className = 'footer-status status-offline';
    return false;
  }

  /**
   * Send text to the analysis API.
   */
  async function callAnalyzeApi(text) {
    const response = await fetch(API_ANALYZE, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/json',
      },
      body: JSON.stringify({ text }),
    });

    if (!response.ok) {
      const errorData = await response
        .json()
        .catch(() => ({ detail: `HTTP ${response.status}` }));
      throw new Error(errorData.detail || `HTTP ${response.status}`);
    }

    return response.json();
  }

  // ---------------------------------------------------------------------------
  // UI Rendering
  // ---------------------------------------------------------------------------

  /**
   * Render the analysis result in the popup.
   */
  function renderResult(data) {
    const isReal = data.prediction === 'real';
    const confidencePercent = Math.round(data.confidence * 100);

    // Badge styling
    els.resultBadge.className = `result-badge ${isReal ? 'badge-real' : 'badge-fake'}`;
    els.resultIcon.innerHTML = isReal
      ? '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M20 6L9 17l-5-5"/></svg>'
      : '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M18 6L6 18M6 6l12 12"/></svg>';
    els.resultLabel.textContent = isReal ? 'Real Review' : 'Fake Review';

    // Confidence
    els.confidenceValue.textContent = `${confidencePercent}%`;
    els.confidenceBar.style.width = `${confidencePercent}%`;
    els.confidenceBar.className = `confidence-bar-fill ${isReal ? 'bar-real' : 'bar-fake'}`;

    // Suspicious phrases
    const phrases = data.suspicious_phrases || [];
    if (phrases.length > 0) {
      els.phrasesSection.classList.remove('hidden');
      els.phrasesList.innerHTML = phrases
        .map(
          (phrase) =>
            `<li class="phrase-item"><span class="phrase-dot"></span>${escapeHtml(phrase)}</li>`
        )
        .join('');
    } else {
      els.phrasesSection.classList.add('hidden');
      els.phrasesList.innerHTML = '';
    }
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  // ---------------------------------------------------------------------------
  // Highlight Communication (popup.js -> background.js -> content.js)
  // ---------------------------------------------------------------------------

  /**
   * Send phrases to background.js for highlighting in the page.
   */
  function sendHighlightMessage(phrases) {
    chrome.runtime.sendMessage({
      action: 'highlightSuspicious',
      phrases: phrases,
    });
  }

  /**
   * Clear all highlights from the page.
   */
  function sendClearHighlights() {
    chrome.runtime.sendMessage({ action: 'clearHighlights' });
  }

  // ---------------------------------------------------------------------------
  // Analysis Workflow
  // ---------------------------------------------------------------------------

  /**
   * Main analysis flow:
   *   1. Show loading state
   *   2. Call API
   *   3. Show result
   *   4. Send phrases to content.js for highlighting
   */
  async function analyzeText(text) {
    if (!text || text.trim().length === 0) return;

    showState('loading');

    try {
      const result = await callAnalyzeApi(text.trim());
      renderResult(result);
      showState('result');

      // Send phrases to content.js via background.js relay
      if (result.suspicious_phrases && result.suspicious_phrases.length > 0) {
        sendHighlightMessage(result.suspicious_phrases);
      }
    } catch (error) {
      els.errorMessage.textContent = error.message || 'Failed to analyze review';
      showState('error');
    }
  }

  // ---------------------------------------------------------------------------
  // Event Handlers
  // ---------------------------------------------------------------------------

  // Analyze button (manual input)
  els.btnAnalyze.addEventListener('click', () => {
    const text = els.manualInput.value;
    if (text.trim().length >= 10) {
      analyzeText(text);
    } else {
      els.errorMessage.textContent = 'Please enter at least 10 characters';
      showState('error');
    }
  });

  // Retry button
  els.btnRetry.addEventListener('click', () => {
    showState('initial');
  });

  // Clear highlights button
  els.btnClear.addEventListener('click', () => {
    sendClearHighlights();
  });

  // New analysis button
  els.btnNew.addEventListener('click', () => {
    sendClearHighlights();
    els.manualInput.value = '';
    showState('initial');
  });

  // ---------------------------------------------------------------------------
  // Message Listener (from content.js)
  // ---------------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analyzeText') {
      // Received text from content.js floating button
      if (request.text && request.text.trim().length >= 10) {
        els.manualInput.value = request.text.trim();
        analyzeText(request.text);
      }
      // No response needed
    }
    return false;
  });

  // ---------------------------------------------------------------------------
  // Initialization
  // ---------------------------------------------------------------------------

  // Query content script for any pending selected text when popup opens
  chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
    if (tabs[0]) {
      chrome.tabs.sendMessage(
        tabs[0].id,
        { action: 'getSelection' },
        (response) => {
          if (chrome.runtime.lastError) {
            // Content script not injected on this page
            return;
          }
          if (response && response.text) {
            els.manualInput.value = response.text;
            analyzeText(response.text);
          }
        }
      );
    }
  });

  // Check API health on popup open
  checkApiHealth();
})();
