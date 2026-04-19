/**
 * Fake Review Detector - Background Service Worker
 * ================================================
 * Minimal message relay between popup.js and content.js.
 *
 * Architecture:
 *   - Receives: { action: "highlightSuspicious", phrases: [...] } from popup.js
 *   - Sends:    same message to active tab's content.js
 *
 * Constraints:
 *   - Does NOT analyze text
 *   - Does NOT save data
 *   - Does NOT open popup
 */

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  // Only handle highlightSuspicious messages from popup.js
  if (request.action === 'highlightSuspicious' || request.action === 'clearHighlights') {
    // Relay to the active tab's content script
    chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
      if (tabs[0] && tabs[0].id) {
        chrome.tabs.sendMessage(tabs[0].id, request, (response) => {
          // Forward the response back to popup.js if needed
          if (chrome.runtime.lastError) {
            // Content script not available on this tab
            sendResponse({ success: false, error: chrome.runtime.lastError.message });
          } else {
            sendResponse(response);
          }
        });
      } else {
        sendResponse({ success: false, error: 'No active tab found' });
      }
    });

    // Return true to indicate async response
    return true;
  }

  // Ignore all other messages (e.g., analyzeText goes directly to popup.js)
  return false;
});
