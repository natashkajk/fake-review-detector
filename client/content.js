/**
 * Fake Review Detector - Content Script
 * Handles text selection, floating button, and page interactions
 */

(function() {
  'use strict';
  
  console.log('[Fake Review Detector] Content script loaded');
  
  // State
  let floatingButton = null;
  let highlightOverlay = null;
  let currentSelection = '';
  let isAnalyzing = false;
  
  // Configuration
  const MIN_TEXT_LENGTH = 10;
  const MAX_TEXT_LENGTH = 2000;
  
  /**
   * Initialize content script
   */
  function init() {
    console.log('[Fake Review Detector] Initializing...');
    
    // Listen for text selection
    document.addEventListener('mouseup', handleTextSelection);
    document.addEventListener('keyup', handleTextSelection);
    
    // Listen for clicks outside to hide button
    document.addEventListener('mousedown', handleOutsideClick);
    
    // Listen for messages from popup/background
    chrome.runtime.onMessage.addListener(handleMessage);
    
    // Listen for scroll to reposition button
    document.addEventListener('scroll', throttle(repositionButton, 100));
    
    console.log('[Fake Review Detector] Initialized successfully');
  }
  
  /**
   * Handle text selection
   */
  function handleTextSelection(event) {
    // Small delay to ensure selection is complete
    setTimeout(() => {
      const selection = window.getSelection();
      const text = selection.toString().trim();
      
      if (text && text.length >= MIN_TEXT_LENGTH && text.length <= MAX_TEXT_LENGTH) {
        currentSelection = text;
        showFloatingButton(selection);
      } else if (text.length < MIN_TEXT_LENGTH) {
        hideFloatingButton();
      }
    }, 10);
  }
  
  /**
   * Show floating button near selection
   */
  function showFloatingButton(selection) {
    if (!selection.rangeCount) return;
    
    const range = selection.getRangeAt(0);
    const rect = range.getBoundingClientRect();
    
    // Remove existing button
    hideFloatingButton();
    
    // Create button
    floatingButton = document.createElement('div');
    floatingButton.className = 'frd-floating-button';
    floatingButton.innerHTML = `
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M2 17L12 22L22 17" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        <path d="M2 12L12 17L22 12" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
      <span>Check Review</span>
    `;
    
    // Position button
    const buttonWidth = 120;
    const buttonHeight = 36;
    const offset = 10;
    
    let left = rect.left + (rect.width / 2) - (buttonWidth / 2);
    let top = rect.top - buttonHeight - offset;
    
    // Adjust if off-screen
    if (left < 10) left = 10;
    if (left + buttonWidth > window.innerWidth - 10) {
      left = window.innerWidth - buttonWidth - 10;
    }
    if (top < 10) {
      top = rect.bottom + offset;
    }
    
    floatingButton.style.cssText = `
      position: fixed;
      left: ${left}px;
      top: ${top}px;
      z-index: 2147483647;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
      color: white;
      padding: 8px 14px;
      border-radius: 20px;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 6px;
      box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
      transition: all 0.2s ease;
      user-select: none;
      animation: frd-fadeIn 0.2s ease;
    `;
    
    // Add hover effect
    floatingButton.addEventListener('mouseenter', () => {
      floatingButton.style.transform = 'translateY(-2px) scale(1.02)';
      floatingButton.style.boxShadow = '0 6px 16px rgba(102, 126, 234, 0.5)';
    });
    
    floatingButton.addEventListener('mouseleave', () => {
      floatingButton.style.transform = 'translateY(0) scale(1)';
      floatingButton.style.boxShadow = '0 4px 12px rgba(102, 126, 234, 0.4)';
    });
    
    // Add click handler
    floatingButton.addEventListener('click', (e) => {
      e.preventDefault();
      e.stopPropagation();
      analyzeSelection();
    });
    
    document.body.appendChild(floatingButton);
    
    // Create highlight overlay
    createHighlightOverlay(rect);
  }
  
  /**
   * Create highlight overlay for selected text
   */
  function createHighlightOverlay(rect) {
    highlightOverlay = document.createElement('div');
    highlightOverlay.className = 'frd-highlight-overlay';
    highlightOverlay.style.cssText = `
      position: absolute;
      left: ${rect.left + window.scrollX}px;
      top: ${rect.top + window.scrollY}px;
      width: ${rect.width}px;
      height: ${rect.height}px;
      background: rgba(102, 126, 234, 0.15);
      border: 2px solid #667eea;
      border-radius: 4px;
      pointer-events: none;
      z-index: 2147483646;
      animation: frd-highlightPulse 2s infinite;
    `;
    
    document.body.appendChild(highlightOverlay);
  }
  
  /**
   * Hide floating button
   */
  function hideFloatingButton() {
    if (floatingButton) {
      floatingButton.remove();
      floatingButton = null;
    }
    if (highlightOverlay) {
      highlightOverlay.remove();
      highlightOverlay = null;
    }
  }
  
  /**
   * Reposition button on scroll
   */
  function repositionButton() {
    if (floatingButton && currentSelection) {
      const selection = window.getSelection();
      if (selection.toString().trim() === currentSelection) {
        const range = selection.getRangeAt(0);
        const rect = range.getBoundingClientRect();
        
        const buttonWidth = 120;
        const buttonHeight = 36;
        const offset = 10;
        
        let left = rect.left + (rect.width / 2) - (buttonWidth / 2);
        let top = rect.top - buttonHeight - offset;
        
        if (left < 10) left = 10;
        if (left + buttonWidth > window.innerWidth - 10) {
          left = window.innerWidth - buttonWidth - 10;
        }
        if (top < 10) {
          top = rect.bottom + offset;
        }
        
        floatingButton.style.left = `${left}px`;
        floatingButton.style.top = `${top}px`;
      }
    }
  }
  
  /**
   * Handle clicks outside the button
   */
  function handleOutsideClick(event) {
    if (floatingButton && !floatingButton.contains(event.target)) {
      hideFloatingButton();
    }
  }
  
  /**
   * Analyze the current selection
   */
  function analyzeSelection() {
    if (!currentSelection || isAnalyzing) return;
    
    isAnalyzing = true;
    console.log('[Fake Review Detector] Analyzing selection:', currentSelection.substring(0, 50) + '...');
    
    // Store the selection for the popup
    chrome.storage.local.set({ pendingReview: currentSelection }, () => {
      if (chrome.runtime.lastError) {
        console.error('[Fake Review Detector] Failed to save pending review:', chrome.runtime.lastError.message);
      }

      // Open the popup
      chrome.runtime.sendMessage({ 
        action: 'openPopup',
        text: currentSelection 
      }, (response) => {
        if (chrome.runtime.lastError) {
          console.error('[Fake Review Detector] Failed to notify background:', chrome.runtime.lastError.message);
        } else {
          console.log('[Fake Review Detector] Popup request sent:', response);
        }
      });
      
      // Hide button
      hideFloatingButton();
      
      isAnalyzing = false;
    });
  }
  
  /**
   * Handle messages from popup/background
   */
  function handleMessage(message, sender, sendResponse) {
    console.log('[Fake Review Detector] Message received:', message);
    
    switch (message.action) {
      case 'resetSelection':
        currentSelection = '';
        hideFloatingButton();
        sendResponse({ success: true });
        break;
        
      case 'highlightSuspicious':
        highlightSuspiciousPhrases(message.phrases);
        sendResponse({ success: true });
        break;

      case 'getSelection':
        sendResponse({ success: true, text: currentSelection || window.getSelection().toString().trim() });
        break;
        
      case 'ping':
        sendResponse({ success: true, status: 'alive' });
        break;
        
      default:
        sendResponse({ success: false, error: 'Unknown action' });
    }
    
    return true;
  }
  
  /**
   * Highlight suspicious phrases on the page
   */
  function highlightSuspiciousPhrases(phrases) {
    if (!phrases || phrases.length === 0) return;
    
    // Remove existing highlights
    removeExistingHighlights();
    
    // Find and highlight each phrase
    phrases.forEach(phrase => {
      highlightPhrase(phrase);
    });
  }
  
  /**
   * Highlight a specific phrase
   */
  function highlightPhrase(phrase) {
    const walker = document.createTreeWalker(
      document.body,
      NodeFilter.SHOW_TEXT,
      null,
      false
    );
    
    const nodesToReplace = [];
    let node;
    
    while (node = walker.nextNode()) {
      const text = node.textContent;
      const index = text.toLowerCase().indexOf(phrase.toLowerCase());
      
      if (index !== -1 && !node.parentElement.classList.contains('frd-suspicious-highlight')) {
        nodesToReplace.push({
          node: node,
          phrase: phrase,
          index: index
        });
      }
    }
    
    nodesToReplace.forEach(({ node, phrase, index }) => {
      const span = document.createElement('span');
      span.className = 'frd-suspicious-highlight';
      span.style.cssText = `
        background: rgba(239, 68, 68, 0.3) !important;
        border-bottom: 2px solid #ef4444 !important;
        border-radius: 2px !important;
        padding: 0 2px !important;
      `;
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
  
  /**
   * Remove existing highlights
   */
  function removeExistingHighlights() {
    const highlights = document.querySelectorAll('.frd-suspicious-highlight');
    highlights.forEach(highlight => {
      const parent = highlight.parentNode;
      while (highlight.firstChild) {
        parent.insertBefore(highlight.firstChild, highlight);
      }
      parent.removeChild(highlight);
    });
  }
  
  /**
   * Throttle function
   */
  function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
      if (!inThrottle) {
        func.apply(this, args);
        inThrottle = true;
        setTimeout(() => inThrottle = false, limit);
      }
    };
  }
  
  // Add CSS animations
  const style = document.createElement('style');
  style.textContent = `
    @keyframes frd-fadeIn {
      from { opacity: 0; transform: translateY(5px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    @keyframes frd-highlightPulse {
      0%, 100% { opacity: 0.6; }
      50% { opacity: 1; }
    }
  `;
  document.head.appendChild(style);
  
  // Initialize
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
  
})();
