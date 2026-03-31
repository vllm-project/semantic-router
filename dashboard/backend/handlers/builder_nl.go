package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"
)

// BuilderNLGenerateHandler converts natural-language routing requests into
// Builder-compatible DSL plus an updated deploy base YAML.
func BuilderNLGenerateHandler(configPath, envoyURL string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req BuilderNLGenerateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
			return
		}

		resp, err := generateBuilderNLDraft(r.Context(), configPath, envoyURL, req)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}
