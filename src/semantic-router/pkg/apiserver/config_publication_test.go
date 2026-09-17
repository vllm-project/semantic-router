//go:build !windows && cgo

package apiserver

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestConfigPutCoordinatesPersistenceAndReleasesBeforeActivation(t *testing.T) {
	path := writeDeployTestBaseConfig(t)
	initial, err := config.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(initial)
	server := &ClassificationAPIServer{configPath: path, runtimeRegistry: registry}
	payload := mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("replacement"))
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(payload)})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPut, "/api/v1/config", bytes.NewReader(body))
	setConfigPrecondition(t, request, path)
	response := httptest.NewRecorder()
	written := make(chan struct{})
	originalRename := atomicRename
	defer func() { atomicRename = originalRename }()
	atomicRename = func(oldPath, newPath string) error {
		if err := originalRename(oldPath, newPath); err != nil {
			return err
		}
		close(written)
		return nil
	}
	// A generation validating its identity and publishing must finish before
	// persistence can replace the source document it just validated.
	release := registry.LockConfigPublication()
	finished := make(chan struct{})
	go func() { server.handleConfigPut(response, request); close(finished) }()
	select {
	case <-written:
		release()
		t.Fatal("config persistence raced the publication critical section")
	case <-time.After(50 * time.Millisecond):
	}
	release()
	select {
	case <-written:
	case <-time.After(5 * time.Second):
		t.Fatal("config was not persisted after publication finished")
	}
	activated := make(chan error, 1)
	go func() {
		unlock := registry.LockConfigPublication()
		defer unlock()
		candidate, parseErr := config.Parse(path)
		if parseErr == nil {
			registry.UpdateConfig(candidate)
		}
		activated <- parseErr
	}()
	select {
	case err := <-activated:
		if err != nil {
			t.Fatal(err)
		}
	case <-time.After(5 * time.Second):
		t.Fatal("API retained publication lock while waiting for activation")
	}
	select {
	case <-finished:
	case <-time.After(5 * time.Second):
		t.Fatal("API did not observe the published generation")
	}
	var result RouterConfigUpdateResponse
	if err := json.Unmarshal(response.Body.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if response.Code != http.StatusOK || result.ActivationStatus != "active" || result.GeneratedRuntimeHash != registry.CurrentConfig().DocumentHash {
		t.Fatalf("config activation response = %d %s", response.Code, response.Body.String())
	}
}
