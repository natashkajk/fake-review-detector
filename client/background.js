/**
 * Fake Review Detector - Background Service Worker
 * Handles API communication and coordinates between content script and popup
 */

// Configuration
const API_BASE_URL = 'http://localhost:8000';

// Store for pending analysis
let pendingAnalysis = null;

/**
 * Initialize service worker
 */
chrome.runtime.onInstalled.addListener((details) => {
  console.log('[Fake Review Detector] Extension installed/updated:', details.reason);
  
  // Set default settings
  chrome.storage.local.set({
    apiUrl: API_BASE_URL,
    autoHighlight: true,
    minTextLength: 10,
    extensionEnabled: true
  });
});

/**
 * Handle messages from content script and popup
 */
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  console.log('[Fake Review Detector] Background received message:', message);
  
  switch (message.action) {
    case 'openPopup':
      handleOpenPopup(message, sender, sendResponse);
      return true; // Keep channel open for async
      
    case 'analyzeText':
      handleAnalyzeText(message, sender, sendResponse);
      return true;
      
    case 'getPendingAnalysis':
      sendResponse({ pendingAnalysis });
      pendingAnalysis = null;
      break;
      
    case 'checkHealth':
      checkApiHealth().then(sendResponse);
      return true;
      
    default:
      sendResponse({ success: false, error: 'Unknown action' });
  }
  
  return true;
});

/**
 * Handle opening popup with text
 */
async function handleOpenPopup(message, sender, sendResponse) {
  try {
    const payload = {};
    if (message.text) {
      payload.pendingReview = message.text;
    }
    if (message.result) {
      payload.analysisResult = message.result;
    }

    if (Object.keys(payload).length > 0) {
      await chrome.storage.local.set(payload);
    }
    
    // Open the popup programmatically
    chrome.action.openPopup();
    
    sendResponse({ success: true });
  } catch (error) {
    console.error('[Fake Review Detector] Error opening popup:', error);
    sendResponse({ success: false, error: error.message });
  }
}

/**
 * Handle text analysis request
 */
async function handleAnalyzeText(message, sender, sendResponse) {
  try {
    const response = await fetch(`${API_BASE_URL}/analyze`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ text: message.text })
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const result = await response.json();
    
    // Store result for popup
    await chrome.storage.local.set({ analysisResult: result });
    
    // Notify content script to highlight suspicious phrases
    if (sender.tab) {
      chrome.tabs.sendMessage(sender.tab.id, {
        action: 'highlightSuspicious',
        phrases: result.suspicious_phrases || []
      });
    }
    
    sendResponse({ success: true, result });
  } catch (error) {
    console.error('[Fake Review Detector] Analysis error:', error);
    sendResponse({ success: false, error: error.message });
  }
}

/**
 * Check API health
 */
async function checkApiHealth() {
  try {
    const response = await fetch(`${API_BASE_URL}/health`, {
      method: 'GET',
      headers: { 'Content-Type': 'application/json' }
    });
    
    return { 
      online: response.ok,
      status: response.status 
    };
  } catch (error) {
    return { 
      online: false, 
      error: error.message 
    };
  }
}

/**
 * Handle extension icon click
 */
chrome.action.onClicked.addListener((tab) => {
  console.log('[Fake Review Detector] Extension icon clicked');
  
  // The popup will open automatically
  // We can use this for additional logic if needed
});

/**
 * Handle tab updates
 */
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete') {
    // Clear any stored analysis when navigating to a new page
    chrome.storage.local.remove(['analysisResult', 'pendingReview']);
  }
});

console.log('[Fake Review Detector] Background service worker initialized');
