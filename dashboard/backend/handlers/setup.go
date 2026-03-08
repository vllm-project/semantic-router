package handlers

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
)

type SetupStateResponse struct {
	SetupMode    bool `json:"setupMode"`
	ListenerPort int  `json:"listenerPort"`
	Models       int  `json:"models"`
	Decisions    int  `json:"decisions"`
	HasModels    bool `json:"hasModels"`
	HasDecisions bool `json:"hasDecisions"`
	CanActivate  bool `json:"canActivate"`
}

type SetupConfigRequest struct {
	Config map[string]interface{} `json:"config"`
}

type SetupValidateResponse struct {
	Valid       bool                   `json:"valid"`
	Config      map[string]interface{} `json:"config,omitempty"`
	Models      int                    `json:"models"`
	Decisions   int                    `json:"decisions"`
	CanActivate bool                   `json:"canActivate"`
}

type SetupActivateResponse struct {
	Status    string `json:"status"`
	SetupMode bool   `json:"setupMode"`
	Message   string `json:"message,omitempty"`
}

func SetupStateHandler(configPath string) http.HandlerFunc {
	return SetupStateHandlerWithService(configlifecycle.New(configPath, ""))
}

func SetupStateHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		state, err := service.SetupState()
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(SetupStateResponse(state)); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func SetupValidateHandler(configPath string) http.HandlerFunc {
	return SetupValidateHandlerWithService(configlifecycle.New(configPath, ""))
}

func SetupValidateHandlerWithService(service *configlifecycle.Service) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}

		requestConfig, err := decodeSetupRequest(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		validation, err := service.ValidateSetup(requestConfig)
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		resp := SetupValidateResponse{
			Valid:       true,
			Config:      validation.Config,
			Models:      validation.Models,
			Decisions:   validation.Decisions,
			CanActivate: validation.CanActivate,
		}
		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(resp); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func SetupActivateHandler(configPath string, readonlyMode bool, configDir string) http.HandlerFunc {
	return SetupActivateHandlerWithService(configlifecycle.New(configPath, configDir), readonlyMode)
}

func SetupActivateHandlerWithService(service *configlifecycle.Service, readonlyMode bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "Method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if readonlyMode {
			writeReadonlyResponse(w, "Dashboard is in read-only mode. Setup activation is disabled.")
			return
		}

		requestConfig, err := decodeSetupRequest(r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}

		activation, err := service.ActivateSetup(requestConfig)
		if err != nil {
			writeLifecycleError(w, err)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		if err := json.NewEncoder(w).Encode(SetupActivateResponse{
			Status:    "success",
			SetupMode: activation.SetupMode,
			Message:   activation.Message,
		}); err != nil {
			http.Error(w, "Failed to encode response", http.StatusInternalServerError)
		}
	}
}

func decodeSetupRequest(r *http.Request) (map[string]interface{}, error) {
	var req SetupConfigRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		return nil, fmt.Errorf("invalid request body: %w", err)
	}
	return req.Config, nil
}
