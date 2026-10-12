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
	Store  string `json:"store"`
	MLJobs struct {
		Total   int `json:"total"`
		Running int `json:"running"`
	} `json:"ml_pipeline_jobs"`
}

// WorkflowHealthHandler reports durable workflow store connectivity and entity counts.
func WorkflowHealthHandler(wf *workflowstore.Store) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if middleware.HandleCORSPreflight(w, r) {
			return
		}
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		var resp WorkflowHealthResponse
		resp.Store = "ok"
		total, running, err := wf.MLWorkflowStats()
		if err != nil {
			log.Printf("workflow health: ml stats: %v", err)
			resp.Store = "degraded"
		} else {
			resp.MLJobs.Total = total
			resp.MLJobs.Running = running
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			log.Printf("workflow health encode: %v", err)
		}
	}
}
