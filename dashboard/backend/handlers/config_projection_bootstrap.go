package handlers

import (
	"log"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
)

// BootstrapConfigProjectionFromFile seeds the projection store from canonical
// config.yaml when no healthy active projection exists yet.
func BootstrapConfigProjectionFromFile(configPath, configDir string) {
	if configProjectionStore == nil {
		return
	}

	projection, err := configProjectionStore.GetActiveProjection()
	if err == nil &&
		projection.Status == configprojection.StatusOK &&
		strings.TrimSpace(projection.ActiveVersion) != "" {
		return
	}

	data, err := os.ReadFile(configPath)
	if err != nil || len(data) == 0 {
		log.Printf("Warning: config projection bootstrap skipped: %v", err)
		return
	}

	if refreshErr := configProjectionStore.RefreshFromCanonical(configprojection.RefreshInput{
		Version:     newActivationVersion(),
		Source:      configprojection.SourceManual,
		YAMLBytes:   data,
		DSLSnapshot: readArchivedDSL(configDir),
	}); refreshErr != nil {
		log.Printf("Warning: config projection bootstrap failed: %v", refreshErr)
		return
	}

	log.Printf("Config projection bootstrapped from canonical file: %s", configPath)
}
