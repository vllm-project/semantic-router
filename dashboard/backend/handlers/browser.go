package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/browser"
	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
)

// BrowserHandler handles browser automation requests
type BrowserHandler struct {
	manager *browser.Manager
}

// NewBrowserHandler creates a new browser handler
func NewBrowserHandler() *BrowserHandler {
	return &BrowserHandler{
		manager: browser.NewManager(),
	}
}

// StartSessionRequest represents a request to start a browser session
type StartSessionRequest struct {
	Headless bool `json:"headless"`
}

// StartSessionResponse represents the response after starting a session
type StartSessionResponse struct {
	SessionID string `json:"session_id"`
	Success   bool   `json:"success"`
	Error     string `json:"error,omitempty"`
}

// ActionRequest represents a browser action request
type ActionRequest struct {
	SessionID string         `json:"session_id"`
	Action    browser.Action `json:"action"`
}

// StartSessionHandler handles POST /api/browser/start
func (h *BrowserHandler) StartSessionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req StartSessionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			// Default to headless
			req.Headless = true
		}

		session, err := h.manager.StartSession(req.Headless)
		if err != nil {
			log.Printf("[Browser] Failed to start session: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(StartSessionResponse{
				Success: false,
				Error:   err.Error(),
			})
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(StartSessionResponse{
			SessionID: session.ID,
			Success:   true,
		})
	}
}

// StopSessionHandler handles DELETE /api/browser/:id
func (h *BrowserHandler) StopSessionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodDelete {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract session ID from path
		parts := strings.Split(strings.TrimPrefix(r.URL.Path, "/api/browser/"), "/")
		if len(parts) == 0 || parts[0] == "" {
			http.Error(w, "Session ID required", http.StatusBadRequest)
			return
		}
		sessionID := parts[0]

		if err := h.manager.StopSession(sessionID); err != nil {
			http.Error(w, err.Error(), http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(map[string]bool{"success": true})
	}
}

// ActionHandler handles POST /api/browser/:id/action
func (h *BrowserHandler) ActionHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract session ID from path
		path := strings.TrimPrefix(r.URL.Path, "/api/browser/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 || parts[0] == "" {
			http.Error(w, "Session ID required", http.StatusBadRequest)
			return
		}
		sessionID := parts[0]

		session, ok := h.manager.GetSession(sessionID)
		if !ok {
			http.Error(w, "Session not found", http.StatusNotFound)
			return
		}

		var action browser.Action
		if err := json.NewDecoder(r.Body).Decode(&action); err != nil {
			http.Error(w, "Invalid action: "+err.Error(), http.StatusBadRequest)
			return
		}

		result, err := session.ExecuteAction(action)
		if err != nil {
			log.Printf("[Browser] Action failed: %v", err)
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusInternalServerError)
			json.NewEncoder(w).Encode(browser.ActionResult{
				Success: false,
				Error:   err.Error(),
			})
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

// ScreenshotHandler handles GET /api/browser/:id/screenshot
func (h *BrowserHandler) ScreenshotHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodGet && r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract session ID from path
		path := strings.TrimPrefix(r.URL.Path, "/api/browser/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 || parts[0] == "" {
			http.Error(w, "Session ID required", http.StatusBadRequest)
			return
		}
		sessionID := parts[0]

		session, ok := h.manager.GetSession(sessionID)
		if !ok {
			http.Error(w, "Session not found", http.StatusNotFound)
			return
		}

		result, err := session.TakeScreenshot()
		if err != nil {
			log.Printf("[Browser] Screenshot failed: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

// NavigateHandler handles POST /api/browser/:id/navigate
func (h *BrowserHandler) NavigateHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}

		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		// Extract session ID from path
		path := strings.TrimPrefix(r.URL.Path, "/api/browser/")
		parts := strings.Split(path, "/")
		if len(parts) < 2 || parts[0] == "" {
			http.Error(w, "Session ID required", http.StatusBadRequest)
			return
		}
		sessionID := parts[0]

		session, ok := h.manager.GetSession(sessionID)
		if !ok {
			http.Error(w, "Session not found", http.StatusNotFound)
			return
		}

		var req struct {
			URL string `json:"url"`
		}
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request: "+err.Error(), http.StatusBadRequest)
			return
		}

		if req.URL == "" {
			http.Error(w, "URL required", http.StatusBadRequest)
			return
		}

		result, err := session.Navigate(req.URL)
		if err != nil {
			log.Printf("[Browser] Navigate failed: %v", err)
			http.Error(w, err.Error(), http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

// Cleanup stops all browser sessions
func (h *BrowserHandler) Cleanup() {
	h.manager.StopAllSessions()
}
