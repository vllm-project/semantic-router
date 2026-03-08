package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
)

type SaveConfigRevisionDraftRequest struct {
	ID                string                 `json:"id,omitempty"`
	ParentRevisionID  string                 `json:"parentRevisionId,omitempty"`
	Source            string                 `json:"source,omitempty"`
	Summary           string                 `json:"summary,omitempty"`
	Document          interface{}            `json:"document,omitempty"`
	RuntimeConfigYAML string                 `json:"runtimeConfigYAML,omitempty"`
	Metadata          map[string]interface{} `json:"metadata,omitempty"`
}

type SaveConfigRevisionDraftResponse struct {
	ConfigRevisionDetailResponse
	Message string `json:"message,omitempty"`
}

type ValidateConfigRevisionRequest struct {
	ID string `json:"id"`
}

type ValidateConfigRevisionResponse struct {
	ConfigRevisionDetailResponse
	Message string `json:"message,omitempty"`
}

func SaveConfigRevisionDraftHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Revision draft updates are disabled.")
			return
		}

		var req SaveConfigRevisionDraftRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		result, err := service.SaveDraftRevision(configlifecycle.RevisionDraftInput{
			ID:                req.ID,
			ParentRevisionID:  req.ParentRevisionID,
			Source:            req.Source,
			Summary:           req.Summary,
			Document:          req.Document,
			RuntimeConfigYAML: req.RuntimeConfigYAML,
			Metadata:          req.Metadata,
		})
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		writeRevisionMutationResponse(w, SaveConfigRevisionDraftResponse{
			ConfigRevisionDetailResponse: revisionDetailResponse(result.RevisionDetail),
			Message:                      result.Message,
		}, "saved config draft revision")
	}
}

func ValidateConfigRevisionHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Revision validation is disabled.")
			return
		}

		var req ValidateConfigRevisionRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "Invalid request body: "+err.Error(), http.StatusBadRequest)
			return
		}

		result, err := service.ValidateRevision(req.ID)
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		writeRevisionMutationResponse(w, ValidateConfigRevisionResponse{
			ConfigRevisionDetailResponse: revisionDetailResponse(result.RevisionDetail),
			Message:                      result.Message,
		}, "validated config revision")
	}
}

func writeRevisionMutationResponse(w http.ResponseWriter, response interface{}, action string) {
	w.Header().Set("Content-Type", "application/json")
	if err := json.NewEncoder(w).Encode(response); err != nil {
		log.Printf("Error encoding %s response: %v", action, err)
	}
}
