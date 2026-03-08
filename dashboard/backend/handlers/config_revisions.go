package handlers

import (
	"encoding/json"
	"log"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
)

type ConfigRevisionSummaryResponse struct {
	ID                string     `json:"id"`
	ParentRevisionID  string     `json:"parentRevisionId,omitempty"`
	Status            string     `json:"status"`
	Source            string     `json:"source,omitempty"`
	Summary           string     `json:"summary,omitempty"`
	CreatedBy         string     `json:"createdBy,omitempty"`
	RuntimeTarget     string     `json:"runtimeTarget,omitempty"`
	LastDeployStatus  string     `json:"lastDeployStatus,omitempty"`
	LastDeployMessage string     `json:"lastDeployMessage,omitempty"`
	ActivatedAt       *time.Time `json:"activatedAt,omitempty"`
	LastDeployedAt    *time.Time `json:"lastDeployedAt,omitempty"`
	CreatedAt         time.Time  `json:"createdAt"`
	UpdatedAt         time.Time  `json:"updatedAt"`
}

type ConfigRevisionDetailResponse struct {
	ConfigRevisionSummaryResponse
	Document          interface{}            `json:"document"`
	RuntimeConfigYAML string                 `json:"runtimeConfigYAML,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

type ActivateConfigRevisionRequest struct {
	ID string `json:"id"`
}

type ActivateConfigRevisionResponse struct {
	ConfigRevisionDetailResponse
	Message string `json:"message,omitempty"`
}

func ConfigRevisionsHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		revisions, err := service.ListRevisions(50)
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		response := make([]ConfigRevisionSummaryResponse, 0, len(revisions))
		for _, revision := range revisions {
			response = append(response, revisionSummaryResponse(revision))
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(response); err != nil {
			log.Printf("Error encoding config revision list: %v", err)
		}
	}
}

func CurrentConfigRevisionHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		revision, err := service.CurrentRevision()
		if err != nil {
			writeLifecycleError(w, err)
			return
		}
		if revision == nil {
			http.Error(w, "No active config revision found", http.StatusNotFound)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(revisionDetailResponse(*revision)); err != nil {
			log.Printf("Error encoding current config revision: %v", err)
		}
	}
}

func ActivateConfigRevisionHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Revision activation is disabled.")
			return
		}

		var req ActivateConfigRevisionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		result, err := service.ActivateRevisionAs(req.ID, requestActorID(r))
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(ActivateConfigRevisionResponse{
			ConfigRevisionDetailResponse: revisionDetailResponse(result.RevisionDetail),
			Message:                      result.Message,
		}); err != nil {
			log.Printf("Error encoding activated config revision: %v", err)
		}
	}
}

func revisionSummaryResponse(revision configlifecycle.RevisionSummary) ConfigRevisionSummaryResponse {
	return ConfigRevisionSummaryResponse{
		ID:                revision.ID,
		ParentRevisionID:  revision.ParentRevisionID,
		Status:            string(revision.Status),
		Source:            revision.Source,
		Summary:           revision.Summary,
		CreatedBy:         revision.CreatedBy,
		RuntimeTarget:     revision.RuntimeTarget,
		LastDeployStatus:  string(revision.LastDeployStatus),
		LastDeployMessage: revision.LastDeployMessage,
		ActivatedAt:       revision.ActivatedAt,
		LastDeployedAt:    revision.LastDeployedAt,
		CreatedAt:         revision.CreatedAt,
		UpdatedAt:         revision.UpdatedAt,
	}
}

func revisionDetailResponse(revision configlifecycle.RevisionDetail) ConfigRevisionDetailResponse {
	return ConfigRevisionDetailResponse{
		ConfigRevisionSummaryResponse: revisionSummaryResponse(revision.RevisionSummary),
		Document:                      revision.Document,
		RuntimeConfigYAML:             revision.RuntimeConfigYAML,
		Metadata:                      revision.Metadata,
	}
}
