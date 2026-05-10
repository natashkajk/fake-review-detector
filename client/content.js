/**
 * Fake Review Detector - Content Script
 * Shows an action near selected text and renders the result inline on the page.
 */

(function () {
  'use strict';

  const API_BASE_URL = 'http://localhost:8000';
  const API_ENDPOINT = `${API_BASE_URL}/analyze`;
  const MIN_TEXT_LENGTH = 10;
  const MAX_TEXT_LENGTH = 2000;

  let currentSelection = '';
  let currentRange = null;
  let currentRect = null;
  let floatingAction = null;
  let reportCard = null;
  let highlightOverlay = null;
  let isAnalyzing = false;
  let extensionEnabled = true;
  let lastResult = null;
  let uiInteractionLock = false;

  function init() {
    chrome.storage.local.get({ extensionEnabled: true }, (result) => {
      extensionEnabled = result.extensionEnabled !== false;
      if (!extensionEnabled) {
        clearInlineUi();
      }
    });

    document.addEventListener('mouseup', handleTextSelection, true);
    document.addEventListener('keyup', handleTextSelection, true);
    document.addEventListener('touchend', handleTextSelection, true);
    document.addEventListener('selectionchange', throttle(handleTextSelection, 60), true);
    document.addEventListener('mousedown', handleOutsideClick, true);
    document.addEventListener('scroll', throttle(repositionUi, 80), true);
    window.addEventListener('resize', throttle(repositionUi, 80), true);

    chrome.runtime.onMessage.addListener(handleMessage);
    chrome.storage.onChanged.addListener(handleStorageChange);
  }

  function handleStorageChange(changes, areaName) {
    if (areaName !== 'local' || !changes.extensionEnabled) return;

    extensionEnabled = changes.extensionEnabled.newValue !== false;
    if (!extensionEnabled) {
      currentSelection = '';
      currentRange = null;
      currentRect = null;
      clearInlineUi();
    }
  }

  function handleTextSelection() {
    if (!extensionEnabled) return;

    setTimeout(() => {
      const selectionState = getSelectionState();
      const text = selectionState.text;

      if (text && text.length >= MIN_TEXT_LENGTH && text.length <= MAX_TEXT_LENGTH) {
        currentSelection = text;
        currentRange = selectionState.range;
        currentRect = selectionState.rect;
        showFloatingAction(selectionState.rect);
        createHighlightOverlay(selectionState.rect);
      } else if (!isSelectionInsideUi() && !uiInteractionLock && !isAnalyzing) {
        currentSelection = '';
        currentRange = null;
        currentRect = null;
        hideFloatingAction();
        removeHighlightOverlay();
      }
    }, 10);
  }

  function getSelectionState() {
    const activeElement = document.activeElement;
    if (
      activeElement &&
      (activeElement.tagName === 'TEXTAREA' ||
        (activeElement.tagName === 'INPUT' && /^(text|search|url|tel|password)$/i.test(activeElement.type)))
    ) {
      const start = activeElement.selectionStart ?? 0;
      const end = activeElement.selectionEnd ?? 0;
      const text = (activeElement.value || '').slice(start, end).trim();
      return {
        text,
        range: null,
        rect: activeElement.getBoundingClientRect(),
      };
    }

    const selection = window.getSelection();
    if (!selection || selection.rangeCount === 0) {
      return { text: '', range: null, rect: null };
    }

    const range = selection.getRangeAt(0).cloneRange();
    const rect = getRangeRect(range);
    return {
      text: selection.toString().trim(),
      range,
      rect,
    };
  }

  function getRangeRect(range) {
    if (!range) return null;

    const rect = range.getBoundingClientRect();
    if (rect && (rect.width || rect.height)) {
      return rect;
    }

    const rects = range.getClientRects();
    if (rects && rects.length > 0) {
      return rects[0];
    }

    return null;
  }

  function showFloatingAction(rect) {
    if (!rect) return;

    hideFloatingAction();
    floatingAction = document.createElement('button');
    floatingAction.type = 'button';
    floatingAction.className = 'frd-floating-action';
    floatingAction.textContent = 'Check review';
    floatingAction.addEventListener('mousedown', lockUiInteraction, true);
    floatingAction.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      analyzeSelection();
    });

    const { left, top } = getActionPosition(rect, 132, 36, 10);
    floatingAction.style.left = `${left}px`;
    floatingAction.style.top = `${top}px`;

    document.body.appendChild(floatingAction);
  }

  function createHighlightOverlay(rect) {
    removeHighlightOverlay();
    if (!rect) return;

    highlightOverlay = document.createElement('div');
    highlightOverlay.className = 'frd-highlight-overlay';
    highlightOverlay.style.left = `${rect.left + window.scrollX}px`;
    highlightOverlay.style.top = `${rect.top + window.scrollY}px`;
    highlightOverlay.style.width = `${rect.width}px`;
    highlightOverlay.style.height = `${rect.height}px`;
    document.body.appendChild(highlightOverlay);
  }

  function removeHighlightOverlay() {
    if (highlightOverlay) {
      highlightOverlay.remove();
      highlightOverlay = null;
    }
  }

  function hideFloatingAction() {
    if (floatingAction) {
      floatingAction.remove();
      floatingAction = null;
    }
  }

  function analyzeSelection() {
    if (!currentSelection || isAnalyzing || !extensionEnabled) return;

    isAnalyzing = true;
    showLoadingCard();

    fetch(API_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ text: currentSelection.trim() }),
    })
      .then(async (response) => {
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.detail || `Server error: ${response.status}`);
        }
        return response.json();
      })
      .then((result) => {
        lastResult = result;
        chrome.storage.local.set({ analysisResult: result });
        renderResultCard(result, false);
      })
      .catch((error) => {
        renderErrorCard(error.message || 'Failed to analyze review.');
      })
      .finally(() => {
        isAnalyzing = false;
      });
  }

  function showLoadingCard() {
    const rect = currentRect || getRangeRect(currentRange);
    if (!rect) return;

    hideFloatingAction();
    removeReportCard();

    reportCard = document.createElement('div');
    reportCard.className = 'frd-report-card frd-loading-card';
    reportCard.innerHTML = `
      <div class="frd-report-head">
        <span class="frd-pill neutral">Analyzing...</span>
      </div>
      <div class="frd-loading-row">
        <span class="frd-loader"></span>
        <span>Checking the selected review with AI</span>
      </div>
    `;

    positionReportCard(rect);
    document.body.appendChild(reportCard);
  }

  function renderResultCard(result) {
    const rect = currentRect || getRangeRect(currentRange);
    if (!rect) return;

    removeReportCard();
    hideFloatingAction();

    const isFake = result.prediction === 'fake';
    const confidence = Math.round((result.confidence || 0) * 100);
    const confidenceLabel = getConfidenceLabel(confidence);

    reportCard = document.createElement('div');
    reportCard.className = `frd-report-card ${isFake ? 'fake' : 'genuine'}`;
    reportCard.innerHTML = `
      <div class="frd-report-head">
        <div class="frd-inline-result">
          <span class="frd-pill ${isFake ? 'fake' : 'genuine'}">${isFake ? 'Fake' : 'Original'}</span>
          <span class="frd-confidence">${confidenceLabel} ${confidence}% confidence</span>
        </div>
        <button type="button" class="frd-report-button" data-action="open-report">Full report</button>
      </div>
      <div class="frd-summary">
        <div class="frd-detail-row">
          <div class="frd-detail-value">${escapeHtml(result.evidence_reason || result.explanation || 'No explanation')}</div>
        </div>
      </div>
    `;

    positionReportCard(rect);
    document.body.appendChild(reportCard);
    reportCard.addEventListener('mousedown', lockUiInteraction, true);

    const reportButton = reportCard.querySelector('[data-action="open-report"]');
    reportButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      openFullReport(result);
    });

    if (result.suspicious_phrases && result.suspicious_phrases.length > 0) {
      highlightSuspiciousPhrases(result.suspicious_phrases);
    }
  }

  function renderErrorCard(message) {
    const rect = currentRect || getRangeRect(currentRange);
    if (!rect) return;

    removeReportCard();
    hideFloatingAction();

    reportCard = document.createElement('div');
    reportCard.className = 'frd-report-card fake';
    reportCard.innerHTML = `
      <div class="frd-report-head">
        <div class="frd-inline-result">
          <span class="frd-pill fake">Error</span>
        </div>
        <button type="button" class="frd-report-button" data-action="retry-analysis">Retry</button>
      </div>
      <div class="frd-summary">
        <div class="frd-detail-row">
          <div class="frd-detail-value">${escapeHtml(message)}</div>
        </div>
      </div>
    `;

    positionReportCard(rect);
    document.body.appendChild(reportCard);
    reportCard.addEventListener('mousedown', lockUiInteraction, true);

    const retryButton = reportCard.querySelector('[data-action="retry-analysis"]');
    retryButton.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      analyzeSelection();
    });
  }

  function positionReportCard(rect) {
    if (!reportCard || !rect) return;

    const width = 340;
    const height = reportCard.offsetHeight || 120;
    const { left, top } = getUiPosition(rect, width, height, 14);

    reportCard.style.left = `${left}px`;
    reportCard.style.top = `${top}px`;
  }

  function getUiPosition(rect, width, height, offset) {
    let left = rect.left + rect.width / 2 - width / 2;
    let top = rect.bottom + offset;

    if (left < 12) left = 12;
    if (left + width > window.innerWidth - 12) {
      left = window.innerWidth - width - 12;
    }

    if (top + height > window.innerHeight - 12) {
      top = rect.top - height - offset;
    }

    if (top < 12) top = 12;
    return { left, top };
  }

  function getActionPosition(rect, width, height, offset) {
    let left = rect.right - width;
    let top = rect.bottom + offset;

    if (left < 12) left = 12;
    if (left + width > window.innerWidth - 12) {
      left = window.innerWidth - width - 12;
    }
    if (top + height > window.innerHeight - 12) {
      top = rect.top - height - offset;
    }
    if (top < 12) {
      top = rect.bottom + offset;
    }
    return { left, top };
  }

  function repositionUi() {
    const rect = currentRange ? getRangeRect(currentRange) : currentRect;
    if (!rect) return;

    currentRect = rect;

    if (floatingAction) {
      const pos = getActionPosition(rect, 132, 36, 10);
      floatingAction.style.left = `${pos.left}px`;
      floatingAction.style.top = `${pos.top}px`;
    }

    if (reportCard) {
      positionReportCard(rect);
    }

    if (highlightOverlay) {
      highlightOverlay.style.left = `${rect.left + window.scrollX}px`;
      highlightOverlay.style.top = `${rect.top + window.scrollY}px`;
      highlightOverlay.style.width = `${rect.width}px`;
      highlightOverlay.style.height = `${rect.height}px`;
    }
  }

  function handleOutsideClick(event) {
    if (
      (floatingAction && floatingAction.contains(event.target)) ||
      (reportCard && reportCard.contains(event.target))
    ) {
      return;
    }

    const selectionText = window.getSelection ? window.getSelection().toString().trim() : '';
    if (!selectionText) {
      clearInlineUi();
    }
  }

  function isSelectionInsideUi() {
    const active = document.activeElement;
    return Boolean(
      active &&
        ((floatingAction && floatingAction.contains(active)) ||
          (reportCard && reportCard.contains(active)))
    );
  }

  function removeReportCard() {
    if (reportCard) {
      reportCard.remove();
      reportCard = null;
    }
  }

  function clearInlineUi() {
    hideFloatingAction();
    removeReportCard();
    removeHighlightOverlay();
    removeExistingHighlights();
  }

  function lockUiInteraction() {
    uiInteractionLock = true;
    window.setTimeout(() => {
      uiInteractionLock = false;
    }, 250);
  }

  function openFullReport(result) {
    chrome.runtime.sendMessage(
      {
        action: 'openPopup',
        text: currentSelection,
        result,
      },
      () => {
        if (chrome.runtime.lastError) {
          console.warn('[Fake Review Detector] Could not open popup:', chrome.runtime.lastError.message);
        }
      }
    );
  }

  function handleMessage(message, sender, sendResponse) {
    switch (message.action) {
      case 'resetSelection':
        currentSelection = '';
        currentRange = null;
        currentRect = null;
        lastResult = null;
        clearInlineUi();
        sendResponse({ success: true });
        break;

      case 'highlightSuspicious':
        highlightSuspiciousPhrases(message.phrases);
        sendResponse({ success: true });
        break;

      case 'getSelection':
        sendResponse({ success: true, text: currentSelection || getSelectionState().text });
        break;

      case 'getInlineResult':
        sendResponse({ success: true, result: lastResult, enabled: extensionEnabled });
        break;

      case 'ping':
        sendResponse({ success: true, status: 'alive' });
        break;

      default:
        sendResponse({ success: false, error: 'Unknown action' });
    }

    return true;
  }

  function highlightSuspiciousPhrases(phrases) {
    if (!phrases || phrases.length === 0) return;

    removeExistingHighlights();
    phrases.forEach((phrase) => highlightPhrase(phrase));
  }

  function highlightPhrase(phrase) {
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT, null, false);
    const nodesToReplace = [];
    let node;

    while ((node = walker.nextNode())) {
      const parent = node.parentElement;
      if (!parent || parent.closest('.frd-report-card')) continue;

      const text = node.textContent || '';
      const index = text.toLowerCase().indexOf(phrase.toLowerCase());
      if (index !== -1 && !parent.classList.contains('frd-suspicious-highlight')) {
        nodesToReplace.push({ node, phrase, index });
      }
    }

    nodesToReplace.forEach(({ node, phrase, index }) => {
      const span = document.createElement('span');
      span.className = 'frd-suspicious-highlight';
      span.title = 'Suspicious phrase detected by Fake Review Detector';

      const beforeText = document.createTextNode(node.textContent.substring(0, index));
      const highlightedText = document.createTextNode(node.textContent.substring(index, index + phrase.length));
      const afterText = document.createTextNode(node.textContent.substring(index + phrase.length));

      span.appendChild(highlightedText);
      const parent = node.parentNode;
      parent.insertBefore(beforeText, node);
      parent.insertBefore(span, node);
      parent.insertBefore(afterText, node);
      parent.removeChild(node);
    });
  }

  function removeExistingHighlights() {
    document.querySelectorAll('.frd-suspicious-highlight').forEach((highlight) => {
      const parent = highlight.parentNode;
      while (highlight.firstChild) {
        parent.insertBefore(highlight.firstChild, highlight);
      }
      parent.removeChild(highlight);
    });
  }

  function getConfidenceLabel(confidence) {
    if (confidence >= 85) return 'High';
    if (confidence >= 65) return 'Moderate';
    return 'Low';
  }

  function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = String(text ?? '');
    return div.innerHTML;
  }

  function throttle(func, limit) {
    let inThrottle;
    return function throttled(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => {
          inThrottle = false;
        }, limit);
      }
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
