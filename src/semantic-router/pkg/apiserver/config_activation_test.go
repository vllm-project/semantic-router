//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestConfigHashReportsExactCandidateFailureAndRecovery(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(path, []byte("version: v0.3\nrouting: {}\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	hash, err := configFileHash(path)
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(&config.RouterConfig{DocumentHash: "old"})
	server := &ClassificationAPIServer{configPath: path, runtimeRegistry: registry}
	mux := server.setupRoutes()
	read := func() configHashResponse {
		t.Helper()
		rr := httptest.NewRecorder()
		mux.ServeHTTP(rr, httptest.NewRequest(http.MethodGet, apiConfigHashPath, nil))
		if rr.Code != http.StatusOK {
			t.Fatalf("hash: %d %s", rr.Code, rr.Body.String())
		}
		var response configHashResponse
		if err := json.Unmarshal(rr.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		return response
	}
	failed := registry.BeginConfigActivation(hash, "file")
	registry.SetConfigActivationStage(failed, "model_prepare")
	registry.FinishConfigActivation(failed, "failed", errors.New("model preparation unavailable"))
	response := read()
	if response.ActivationStatus != "failed" || response.Activation == nil || response.Activation.Error == "" || response.ActiveRuntimeHash != "old" {
		t.Fatalf("failed candidate obscured: %+v", response)
	}
	generatedDocument, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if _, status := server.waitForRuntimeConfigActivation(path, generatedDocument, 0); status != "failed" {
		t.Fatalf("wait status: %s", status)
	}
	registry.BeginConfigActivation("another-document", "file")
	if response = read(); response.ActivationStatus != "pending" || response.Activation != nil {
		t.Fatalf("foreign candidate leaked: %+v", response)
	}
	registry.UpdateConfig(&config.RouterConfig{DocumentHash: hash})
	if response = read(); response.ActivationStatus != "active" {
		t.Fatalf("published generation not active: %+v", response)
	}
}

func TestConfigMutationRetriesSameHashAfterFailedActivation(t *testing.T) {
	for _, operation := range []string{"put", "rollback"} {
		for _, outcome := range []string{"active", "failed"} {
			t.Run(operation+"/"+outcome, func(t *testing.T) {
				path := writeDeployTestBaseConfig(t)
				doc, _, err := readConfigDocument(path)
				if err != nil {
					t.Fatal(err)
				}
				// PUT normalizes the document; persist that exact representation so
				// both operations retry the identical previously failed hash.
				candidate, err := normalizeRouterConfigDocument(doc)
				if err != nil {
					t.Fatal(err)
				}
				if writeErr := os.WriteFile(path, candidate, 0o600); writeErr != nil {
					t.Fatal(writeErr)
				}
				candidateConfig, err := config.Parse(path)
				if err != nil {
					t.Fatal(err)
				}
				previousConfig := *candidateConfig
				previousConfig.DocumentHash = "previous-runtime"
				registry := routerruntime.NewRegistry(&previousConfig)
				failed := registry.BeginConfigActivation(candidateConfig.DocumentHash, "file")
				registry.FinishConfigActivation(failed, "failed", errors.New("previous attempt failed"))
				server := &ClassificationAPIServer{configPath: path, runtimeRegistry: registry}
				method, route := http.MethodPut, apiConfigPath
				var payload any = RouterConfigUpdateRequest{YAML: string(candidate)}
				if operation == "rollback" {
					const version = "20260802-120000"
					backupDir := filepath.Join(filepath.Dir(path), ".vllm-sr", "config-backups")
					recordConfigBackup(backupDir, version, candidate, configVersionSourceAPI)
					method, route = http.MethodPost, apiConfigRollbackPath
					payload = routerConfigRollbackRequest{Version: version}
				}
				body, err := json.Marshal(payload)
				if err != nil {
					t.Fatal(err)
				}
				request := httptest.NewRequest(method, route, bytes.NewReader(body))
				setConfigPrecondition(t, request, path)
				before, err := os.Stat(path)
				if err != nil {
					t.Fatal(err)
				}
				response := httptest.NewRecorder()
				done := make(chan struct{})
				mux := server.setupRoutes()
				go func() {
					mux.ServeHTTP(response, request)
					close(done)
				}()
				// Atomic persistence replaces the inode even though the bytes are
				// identical. Keep watcher preparation delayed until after that.
				persisted := false
				deadline := time.Now().Add(2 * time.Second)
				for time.Now().Before(deadline) {
					after, statErr := os.Stat(path)
					if statErr == nil && !os.SameFile(before, after) {
						persisted = true
						break
					}
					time.Sleep(time.Millisecond)
				}
				returnedBeforeRetry := false
				select {
				case <-done:
					returnedBeforeRetry = true
				case <-time.After(50 * time.Millisecond):
				}
				attempt := registry.BeginConfigActivation(candidateConfig.DocumentHash, "file")
				wantStatus := http.StatusOK
				if outcome == "active" {
					registry.UpdateConfig(candidateConfig)
					registry.FinishConfigActivation(attempt, "active", nil)
				} else {
					wantStatus = http.StatusServiceUnavailable
					registry.FinishConfigActivation(attempt, "failed", errors.New("retried attempt failed"))
				}
				select {
				case <-done:
				case <-time.After(2 * time.Second):
					t.Fatal("mutation did not observe retried activation")
				}
				if !persisted || returnedBeforeRetry {
					t.Fatalf("stale failure completed retried mutation: persisted=%t early=%t body=%s", persisted, returnedBeforeRetry, response.Body.String())
				}
				var result RouterConfigUpdateResponse
				if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
					t.Fatal(err)
				}
				if response.Code != wantStatus || result.ActivationStatus != outcome || result.Activation == nil || result.Activation.Attempt != attempt {
					t.Fatalf("retry result: status=%d body=%s", response.Code, response.Body.String())
				}
			})
		}
	}
}
