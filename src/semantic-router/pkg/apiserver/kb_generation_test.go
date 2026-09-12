//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
)

func TestKnowledgeBaseCandidatePreservesRetiredAssetsAndRejectsPendingWrites(t *testing.T) {
	server, _, configPath := newTestKnowledgeBaseAPIServer(t)
	withStubbedRuntimeConfigSync(t)
	payload := testKnowledgeBasePayload()
	created := createKnowledgeBaseDocument(t, server, payload)
	old, err := config.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server.runtimeRegistry = routerruntime.NewRegistry(old)
	server.runtimeConfig = newLiveRuntimeConfig(old, server.runtimeRegistry.CurrentConfig, nil)
	baseDir := knowledgeBaseConfigBaseDir(old, configPath)
	oldKB, _ := knowledgeBaseByName(old.KnowledgeBases, payload.Name)
	oldDefinition, err := config.LoadKnowledgeBaseDefinition(baseDir, oldKB.Source)
	if err != nil {
		t.Fatal(err)
	}
	payload.Description = "candidate definition"
	payload.Labels[0].Exemplars = []string{"Candidate-only exemplar"}
	request := httptest.NewRequest(http.MethodPut, "/api/v1/storage/knowledge-bases/"+payload.Name, bytes.NewReader(mustMarshalKnowledgeBasePayload(t, payload)))
	request.SetPathValue("name", payload.Name)
	response := httptest.NewRecorder()
	server.handleUpdateKnowledgeBase(response, request)
	if response.Code != http.StatusAccepted {
		t.Fatalf("update = %d: %s", response.Code, response.Body.String())
	}
	updated := mustDecodeKnowledgeBaseDocument(t, response)
	if updated.Source.Path == created.Source.Path || !strings.Contains(updated.Source.Path, "/revisions/") {
		t.Fatalf("candidate did not use an independent asset revision: %+v", updated.Source)
	}
	// A watcher may reject preparation. Until it publishes, every old reader
	// must retain its complete definition, not merely the old config pointer.
	actualOld, err := config.LoadKnowledgeBaseDefinition(baseDir, oldKB.Source)
	if err != nil || !reflect.DeepEqual(actualOld, oldDefinition) {
		t.Fatalf("retired definition changed before publication: %+v, %v", actualOld, err)
	}
	oldDocument, err := buildKnowledgeBaseDocument(old, baseDir, oldKB)
	if err != nil || oldDocument.LoadError != "" || oldDocument.Description != created.Description || !reflect.DeepEqual(oldDocument.Labels, created.Labels) {
		t.Fatalf("old generation document changed: %+v, %v", oldDocument, err)
	}
	before, err := os.ReadFile(configPath)
	if err != nil {
		t.Fatal(err)
	}
	payload.Name = "must_not_replace_pending"
	second := httptest.NewRecorder()
	server.handleCreateKnowledgeBase(second, httptest.NewRequest(http.MethodPost, "/api/v1/storage/knowledge-bases", bytes.NewReader(mustMarshalKnowledgeBasePayload(t, payload))))
	if second.Code != http.StatusConflict || !strings.Contains(second.Body.String(), "CONFIG_ACTIVATION_PENDING") {
		t.Fatalf("second mutation = %d: %s", second.Code, second.Body.String())
	}
	after, err := os.ReadFile(configPath)
	if err != nil || !bytes.Equal(before, after) {
		t.Fatalf("pending mutation replaced the saved candidate: %v", err)
	}
	if _, statErr := os.Stat(managedKnowledgeBaseDirForSource(baseDir, managedKnowledgeBaseSourcePath(payload.Name), payload.Name)); !os.IsNotExist(statErr) {
		t.Fatalf("rejected mutation staged assets: %v", statErr)
	}
	candidate, err := config.Parse(configPath)
	if err != nil {
		t.Fatal(err)
	}
	server.runtimeRegistry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: candidate})
	deleteRequest := httptest.NewRequest(http.MethodDelete, "/api/v1/storage/knowledge-bases/"+created.Name, nil)
	deleteRequest.SetPathValue("name", created.Name)
	deleted := httptest.NewRecorder()
	server.handleDeleteKnowledgeBase(deleted, deleteRequest)
	if deleted.Code != http.StatusAccepted {
		t.Fatalf("delete = %d: %s", deleted.Code, deleted.Body.String())
	}
	retiredKB, _ := knowledgeBaseByName(candidate.KnowledgeBases, created.Name)
	retired, err := buildKnowledgeBaseDocument(candidate, baseDir, retiredKB)
	if err != nil || retired.LoadError != "" || retired.Description != updated.Description || !reflect.DeepEqual(retired.Labels, updated.Labels) {
		t.Fatalf("delete removed assets still used by a retired generation: %+v, %v", retired, err)
	}
}

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
