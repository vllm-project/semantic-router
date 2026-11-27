package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/middleware"
	"github.com/vllm-project/semantic-router/dashboard/backend/workflowstore"
)

// WorkflowHealthResponse is a typed control-plane snapshot (not log-derived).
type WorkflowHealthResponse struct {
	Store      string `json:"store"`
	MCPServers int    `json:"mcp_servers"`
}

// WorkflowHealthHandler reports durable workflow store connectivity and entity counts.
func WorkflowHealthHandler(wf *workflowstore.Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet && r.Method != http.MethodHead {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		resp := WorkflowHealthResponse{Store: "ok"}
		count, err := wf.MCPServerCount()
		if err != nil {
			log.Printf("workflow health: mcp servers: %v", err)
			resp.Store = "degraded"
		} else {
			resp.MCPServers = count
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Printf("workflow health encode: %v", err)
		}
	}
}
