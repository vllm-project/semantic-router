//go:build !windows && cgo

package apiserver

import (
	"net/http"
	"os"
	"path/filepath"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestKnowledgeBasePersistenceWaitsForWholeGenerationPublication(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	previous, candidate := []byte("version: v0.3\nrouting: {}\n"), []byte("version: v0.3\nrouting: {}\n# new KB candidate\n")
	if err := os.WriteFile(path, previous, 0o600); err != nil {
		t.Fatal(err)
	}
	old := &config.RouterConfig{DocumentHash: "old-generation"}
	service := services.NewClassificationService(nil, old)
	registry := routerruntime.NewRegistry(old)
	registry.SetClassificationService(service)
	server := &ClassificationAPIServer{config: old, configPath: path, runtimeRegistry: registry, classificationSvc: service}
	newConfig := &config.RouterConfig{}
	paths := resolveConfigPersistencePaths(path)
	if err := persistConfigAndSync(server, paths, previous, candidate, newConfig); err != nil {
		t.Fatal(err)
	}
	if registry.CurrentConfig() != old || service.GetConfig() != old || server.currentConfig() != old {
		t.Fatal("KB persistence mutated the borrowed live generation")
	}
	persisted, err := os.ReadFile(path)
	if err != nil || string(persisted) != string(candidate) {
		t.Fatalf("candidate was not persisted: %v", err)
	}
	state, status := server.knowledgeBaseActivationStatus(paths.runtimePath, http.StatusCreated)
	if status != http.StatusAccepted || state.ActivationStatus != "pending" || state.GeneratedRuntimeHash == "" {
		t.Fatalf("pending state=%+v status=%d", state, status)
	}
	// Only the watcher can publish the prepared service/config pair.
	newConfig.DocumentHash = state.GeneratedRuntimeHash
	nextService := services.NewClassificationService(nil, newConfig)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: newConfig, ClassificationService: nextService})
	state, status = server.knowledgeBaseActivationStatus(paths.runtimePath, http.StatusCreated)
	if status != http.StatusCreated || state.ActivationStatus != "active" || registry.ClassificationService() != nextService {
		t.Fatalf("published state=%+v status=%d", state, status)
	}
	if service.GetConfig() != old {
		t.Fatal("retired service snapshot was mutated")
	}
}
