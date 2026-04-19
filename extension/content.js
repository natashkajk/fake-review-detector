/**
 * Fake Review Detector - Content Script
 * =====================================
 * Injects a floating "Check Review" button when text is selected.
 * Handles text highlighting for suspicious phrases.
 *
 * Architecture:
 *   - content.js -> popup.js:    { action: "analyzeText", text: "..." }
 *   - content.js <- background.js: { action: "highlightSuspicious", phrases: [...] }
 *
 * Constraints:
 *   - Does NOT use chrome.storage
 *   - Does NOT open popup
 *   - Does NOT call API directly
 */

(function () {
  'use strict';

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  /** Currently selected review text ( persisted until replaced ) */
  let pendingText = null;

  /** Reference to the floating button DOM element */
  let floatButton = null;

  /** Unique attribute used to tag our highlight spans */
  const HIGHLIGHT_ATTR = 'data-fake-review-highlight';

  // ---------------------------------------------------------------------------
  // Selection handling
  // ---------------------------------------------------------------------------

  /**
   * Extract the plain-text content of the current selection.
   * Returns null if the selection is outside the allowed length range.
   */
  function getSelectedText() {
    const selection = window.getSelection();
    const text = selection.toString().trim();
    if (text.length >= 10 && text.length <= 2000) {
      return text;
    }
    return null;
  }

  /**
   * Calculate the viewport position where the floating button should appear.
   */
  function getButtonPosition() {
    const selection = window.getSelection();
    if (!selection.rangeCount) return { x: 0, y: 0 };

    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();

    // Position button below the selection, slightly to the right
    return {
      x: rect.left + window.scrollX + rect.width / 2,
      y: rect.bottom + window.scrollY + 8,
    };
  }

  // ---------------------------------------------------------------------------
  // Floating button UI
  // ---------------------------------------------------------------------------

  /**
   * Create and inject the floating "Check Review" button.
   */
  function showFloatingButton() {
    removeFloatingButton();

    const pos = getButtonPosition();

    floatButton = document.createElement('button');
    floatButton.textContent = 'Check Review';
    floatButton.id = 'fake-review-floating-btn';
    floatButton.setAttribute(HIGHLIGHT_ATTR, 'button');

    // Positioning
    floatButton.style.position = 'absolute';
    floatButton.style.left = `${pos.x}px`;
    floatButton.style.top = `${pos.y}px`;
    floatButton.style.transform = 'translateX(-50%)';
    floatButton.style.zIndex = '999999';

    // Click handler
    floatButton.addEventListener('click', onButtonClick);

    document.body.appendChild(floatButton);
  }

  /**
   * Remove the floating button from the DOM.
   */
  function removeFloatingButton() {
    if (floatButton && floatButton.parentNode) {
      floatButton.parentNode.removeChild(floatButton);
      floatButton = null;
    }
  }

  /**
   * Handle floating button click:
   *   1. Send the selected text to popup.js via runtime message
   *   2. Hide the button
   */
  function onButtonClick() {
    if (!pendingText) return;

    chrome.runtime.sendMessage({
      action: 'analyzeText',
      text: pendingText,
    });

    removeFloatingButton();
  }

  // ---------------------------------------------------------------------------
  // Text highlighting
  // ---------------------------------------------------------------------------

  /**
   * Remove all previously injected highlight spans.
   */
  function clearHighlights() {
    const spans = document.querySelectorAll(`[${HIGHLIGHT_ATTR}="phrase"]`);
    spans.forEach((span) => {
      // Replace span with its text content to restore original DOM
      const parent = span.parentNode;
      if (parent) {
        const textNode = document.createTextNode(span.textContent);
        parent.replaceChild(textNode, span);
        // Normalize adjacent text nodes
        parent.normalize();
      }
    });
  }

  /**
   * Recursively search for text nodes inside an element.
   */
  function getTextNodes(element) {
    const textNodes = [];
    const walker = document.createTreeWalker(
      element,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );
    let node;
    while ((node = walker.nextNode())) {
      // Ignore empty or whitespace-only nodes
      if (node.textContent.trim().length > 0) {
        textNodes.push(node);
      }
    }
    return textNodes;
  }

  /**
   * Highlight all occurrences of a phrase in the page.
   * Wraps matching text in a styled span.
   */
  function highlightPhrase(phrase) {
    if (!phrase || phrase.length < 2) return;

    // Escape special regex characters
    const escaped = phrase.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    const regex = new RegExp(`(${escaped})`, 'gi');

    // Get all text nodes in body (excluding our own elements)
    const textNodes = getTextNodes(document.body);

    for (const textNode of textNodes) {
      // Skip nodes inside our highlight spans or button
      let parent = textNode.parentNode;
      let skip = false;
      while (parent) {
        if (
          parent.hasAttribute &&
          (parent.hasAttribute(HIGHLIGHT_ATTR) ||
            parent.id === 'fake-review-floating-btn')
        ) {
          skip = true;
          break;
        }
        parent = parent.parentNode;
      }
      if (skip) continue;

      const text = textNode.textContent;
      if (!regex.test(text)) continue;
      regex.lastIndex = 0; // Reset regex

      const parts = text.split(regex);
      if (parts.length <= 1) continue;

      // Build replacement fragment
      const fragment = document.createDocumentFragment();
      for (let i = 0; i < parts.length; i++) {
        if (regex.test(parts[i])) {
          regex.lastIndex = 0;
          const span = document.createElement('span');
          span.setAttribute(HIGHLIGHT_ATTR, 'phrase');
          span.className = 'fake-review-highlight';
          span.textContent = parts[i];
          fragment.appendChild(span);
        } else {
          fragment.appendChild(document.createTextNode(parts[i]));
        }
      }

      // Replace original node with fragment
      textNode.parentNode.replaceChild(fragment, textNode);
    }
  }

  /**
   * Highlight all suspicious phrases on the page.
   * Clears previous highlights first.
   */
  function highlightSuspiciousPhrases(phrases) {
    clearHighlights();

    if (!phrases || phrases.length === 0) return;

    // Deduplicate and sort by length (longest first to avoid nested matches)
    const uniquePhrases = [...new Set(phrases)].sort(
      (a, b) => b.length - a.length
    );

    for (const phrase of uniquePhrases) {
      highlightPhrase(phrase);
    }
  }

  // ---------------------------------------------------------------------------
  // Event listeners
  // ---------------------------------------------------------------------------

  /**
   * On mouseup: check if user selected valid text and show the button.
   */
  document.addEventListener('mouseup', () => {
    // Small delay to let the selection finalize
    setTimeout(() => {
      const selected = getSelectedText();
      if (selected) {
        pendingText = selected;
        showFloatingButton();
      } else {
        pendingText = null;
        removeFloatingButton();
      }
    }, 10);
  });

  /**
   * Hide button when clicking elsewhere.
   */
  document.addEventListener('mousedown', (event) => {
    if (floatButton && !floatButton.contains(event.target)) {
      removeFloatingButton();
    }
  });

  // ---------------------------------------------------------------------------
  // Message handling (from background.js)
  // ---------------------------------------------------------------------------

  chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'getSelection') {
      // popup.js is asking for the currently selected text
      sendResponse({ text: pendingText });
      return true;
    }

    if (request.action === 'highlightSuspicious') {
      // Highlight phrases on the page
      highlightSuspiciousPhrases(request.phrases);
      sendResponse({ success: true });
      return true;
    }

    if (request.action === 'clearHighlights') {
      clearHighlights();
      sendResponse({ success: true });
      return true;
    }

    return false;
  });
})();
