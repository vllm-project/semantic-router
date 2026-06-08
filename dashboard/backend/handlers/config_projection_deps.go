package handlers

import (
	"log"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

var (
	configProjectionStore        *configprojection.Store
	configProjectionRefreshAsync = true
)

// SetConfigProjectionStore wires the dashboard config projection store into handlers.
func SetConfigProjectionStore(store *configprojection.Store) {
	configProjectionStore = store
}

// SetConfigProjectionRefreshAsync controls whether projection refresh runs in the
// request goroutine. Tests disable async mode for deterministic assertions.
func SetConfigProjectionRefreshAsync(enabled bool) {
	configProjectionRefreshAsync = enabled
}

func refreshConfigProjection(input configprojection.RefreshInput) {
	if configProjectionStore == nil {
		log.Printf("Warning: config projection store is not initialized")
		return
	}

	refresh := func() {
		if err := configProjectionStore.RefreshFromCanonical(input); err != nil {
			log.Printf("Warning: failed to refresh config projection: %v", err)
		}
	}

	if configProjectionRefreshAsync {
		go refresh()
		return
	}
	refresh()
}

func newActivationVersion() string {
	return configprojection.NewActivationVersion()
}
