package handlers

import (
	"log"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

var configProjectionStore *configprojection.Store

// SetConfigProjectionStore wires the dashboard config projection store into handlers.
func SetConfigProjectionStore(store *configprojection.Store) {
	configProjectionStore = store
}

func refreshConfigProjection(input configprojection.RefreshInput) {
	if configProjectionStore == nil {
		log.Printf("Warning: config projection store is not initialized")
		return
	}
	if err := configProjectionStore.RefreshFromCanonical(input); err != nil {
		log.Printf("Warning: failed to refresh config projection: %v", err)
	}
}

func newActivationVersion() string {
	return configprojection.NewActivationVersion()
}
