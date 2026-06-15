package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"sync"
)

// BuilderNLGenerateHandler converts natural-language routing requests into
// Builder-compatible DSL plus the current deploy base YAML context.
func BuilderNLGenerateHandler(configPath, envoyURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeBuilderNLError(w, http.StatusMethodNotAllowed, "Method not allowed")
			return
		}

		var req BuilderNLGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeBuilderNLError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request body: %v", err))
			return
		}

		resp, err := generateBuilderNLDraft(r.Context(), configPath, envoyURL, req)
		if err != nil {
			writeBuilderNLError(w, http.StatusBadRequest, err.Error())
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			writeBuilderNLError(w, http.StatusInternalServerError, "Failed to encode response")
		}
	}
}

// BuilderNLGenerateStreamHandler streams live Builder NL progress events and
// the final staged draft payload over an SSE response.
func BuilderNLGenerateStreamHandler(configPath, envoyURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeBuilderNLError(w, http.StatusMethodNotAllowed, "Method not allowed")
			return
		}

		var req BuilderNLGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeBuilderNLError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request body: %v", err))
			return
		}

		flusher, ok := w.(http.Flusher)
		if !ok {
			writeBuilderNLError(w, http.StatusInternalServerError, "Streaming not supported")
			return
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.Header().Set("Content-Encoding", "identity")
		w.WriteHeader(http.StatusOK)
		if _, err := fmt.Fprintf(w, ": %s\n\n", strings.Repeat(" ", 2048)); err != nil {
			return
		}
		flusher.Flush()

		var writeMu sync.Mutex
		writeEvent := func(eventType string, payload any) error {
			writeMu.Lock()
			defer writeMu.Unlock()

			raw, err := json.Marshal(payload)
			if err != nil {
				return err
			}
			if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, raw); err != nil {
				return err
			}
			flusher.Flush()
			return nil
		}

		_ = writeEvent("connected", map[string]string{"status": "started"})
		reporter := func(event BuilderNLProgressEvent) {
			_ = writeEvent("progress", event)
		}

		resp, err := generateBuilderNLDraftWithProgress(r.Context(), configPath, envoyURL, req, reporter)
		if err != nil {
			_ = writeEvent("error", map[string]string{"message": err.Error()})
			return
		}

		_ = writeEvent("result", resp)
	}
}

// BuilderNLVerifyHandler verifies that the selected Builder AI connection can
// complete a minimal request before the user attempts generation.
func BuilderNLVerifyHandler(configPath, envoyURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			writeBuilderNLError(w, http.StatusMethodNotAllowed, "Method not allowed")
			return
		}

		var req BuilderNLVerifyRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeBuilderNLError(w, http.StatusBadRequest, fmt.Sprintf("Invalid request body: %v", err))
			return
		}

		resp, err := verifyBuilderNLConnection(r.Context(), configPath, envoyURL, req)
		if err != nil {
			writeBuilderNLError(w, http.StatusBadRequest, err.Error())
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			writeBuilderNLError(w, http.StatusInternalServerError, "Failed to encode response")
		}
	}
}

func writeBuilderNLError(w http.ResponseWriter, status int, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"error":   "builder_nl_error",
		"message": message,
	})
}
