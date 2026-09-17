package handlers

import (
	"encoding/json"
	"log"
	"net/http"

	"github.com/vllm-project/semantic-router/dashboard/backend/setupmode"
)

// failSetupActivation is called while the ordinary runtime-config mutation
// lock is still held. Restoring the bootstrap document keeps a failed first
// activation retryable without allowing it to overwrite another config save.
// A retry uses the existing restart-or-start lifecycle for any service that
// already started before its peer failed.
func failSetupActivation(w http.ResponseWriter, configPath string, previousData []byte, resolver *setupmode.Resolver, stage string) {
	restored := false
	if err := writeConfigAtomically(configPath, previousData); err == nil {
		resolver.Invalidate()
		_, syncErr := syncRuntimeConfigForCurrentRuntime(configPath)
		restored = syncErr == nil
	} else {
		resolver.Invalidate()
	}

	message := "Activation failed; the setup configuration was restored. Correct the runtime failure and retry activation."
	if !restored {
		message = "Activation failed and the previous runtime configuration could not be fully restored. Inspect the current setup state and service logs before retrying."
	}
	// Python renderer errors may quote arbitrary configuration values, including
	// provider credentials. Expose the bounded phase and restoration outcome,
	// never the raw subprocess output, in either this response or this log.
	log.Printf("Setup activation failed: stage=%s configuration_restored=%t", stage, restored)
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusInternalServerError)
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"error":                  "setup_activation_failed",
		"stage":                  stage,
		"configuration_restored": restored,
		"setupMode":              resolver.Resolve().Active,
		"message":                message,
	})
}
