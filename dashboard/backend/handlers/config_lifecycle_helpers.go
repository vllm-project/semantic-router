package handlers

import (
	"encoding/json"
	"errors"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/configlifecycle"
)

func writeReadonlyResponse(w http.ResponseWriter, message string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusForbidden)
	if err := json.NewEncoder(w).Encode(map[string]string{
		"error":   "readonly_mode",
		"message": message,
	}); err != nil {
		log.Printf("Error encoding readonly response: %v", err)
	}
}

func writeLifecycleError(w http.ResponseWriter, err error) {
	var lifecycleErr *configlifecycle.Error
	if errors.As(err, &lifecycleErr) {
		if lifecycleErr.Code != "" {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(lifecycleErr.StatusCode)
			_ = json.NewEncoder(w).Encode(map[string]string{
				"error":   lifecycleErr.Code,
				"message": lifecycleErr.Message,
			})
			return
		}
		http.Error(w, lifecycleErr.Message, lifecycleErr.StatusCode)
		return
	}
	http.Error(w, err.Error(), http.StatusInternalServerError)
}
