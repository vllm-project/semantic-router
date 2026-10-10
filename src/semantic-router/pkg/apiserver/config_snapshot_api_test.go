//go:build !windows

package apiserver

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// publishingRuntime plays the Router for the lifecycle: activating a
// candidate publishes it, as the router's generation swap does.
type publishingRuntime struct {
	registry *routerruntime.Registry
	reject   error
}

func (r publishingRuntime) Validate(context.Context, *configsnapshot.Candidate) error { return nil }

func (r publishingRuntime) Warm(_ context.Context, c *configsnapshot.Candidate) (configsnapshot.Warmed, error) {
	if r.reject != nil {
		return nil, r.reject
	}
	return publishingWarmed{registry: r.registry, candidate: c}, nil
}

type publishingWarmed struct {
	registry  *routerruntime.Registry
	candidate *configsnapshot.Candidate
}

func (w publishingWarmed) Activate(context.Context) error {
	snapshot := w.candidate.Snapshot()
	w.registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: snapshot.Config(), ConfigSnapshot: snapshot})
	return nil
}

func (publishingWarmed) Discard() {}

func newSnapshotAPIServer(t *testing.T, reject error) (*ClassificationAPIServer, *configsnapshot.Manager, string) {
	t.Helper()
	path := writeDeployTestBaseConfig(t)
	startup, err := config.Parse(path)
	if err != nil {
		t.Fatal(err)
	}
	document, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(startup)
	manager := configsnapshot.NewManager(configsnapshot.Options{
		Runtime: publishingRuntime{registry: registry, reject: reject}, Reporter: registry,
	})
	snapshot, err := manager.Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: startup, Document: document,
	})
	if err != nil {
		t.Fatal(err)
	}
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: startup, ConfigSnapshot: snapshot})
	return &ClassificationAPIServer{configPath: path, runtimeRegistry: registry}, manager, path
}

// watchAndApply plays the file watcher: once the persisted document changes,
// it hands that document to the lifecycle. Apply writes a history backup, so
// the test waits for that goroutine before TempDir removes the directory.
func watchAndApply(t *testing.T, manager *configsnapshot.Manager, path string) {
	t.Helper()
	before, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		deadline := time.Now().Add(5 * time.Second)
		for time.Now().Before(deadline) {
			data, readErr := os.ReadFile(path)
			if readErr == nil && !bytes.Equal(data, before) {
				cfg, parseErr := config.Parse(path)
				if parseErr != nil {
					t.Errorf("parse persisted candidate: %v", parseErr)
					return
				}
				_, _ = manager.Apply(context.Background(), configsnapshot.Update{
					Origin: configsnapshot.Origin{Source: configsnapshot.SourceFile}, Config: cfg, Document: data,
				})
				return
			}
			time.Sleep(time.Millisecond)
		}
	}()
	t.Cleanup(func() { <-done })
}

func putConfig(t *testing.T, server *ClassificationAPIServer, path string) (*httptest.ResponseRecorder, RouterConfigUpdateResponse) {
	t.Helper()
	body, err := json.Marshal(RouterConfigUpdateRequest{YAML: string(mustMarshalCanonicalConfigYAML(t, minimalDeployTestConfig("new_route")))})
	if err != nil {
		t.Fatal(err)
	}
	request := httptest.NewRequest(http.MethodPut, apiConfigPath, bytes.NewReader(body))
	setConfigPrecondition(t, request, path)
	recorder := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(recorder, request)
	var result RouterConfigUpdateResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &result); err != nil {
		t.Fatalf("decode %s: %v", recorder.Body.String(), err)
	}
	return recorder, result
}

func readConfigHash(t *testing.T, server *ClassificationAPIServer) configHashResponse {
	t.Helper()
	recorder := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, apiConfigHashPath, nil))
	var response configHashResponse
	if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
		t.Fatalf("decode %s: %v", recorder.Body.String(), err)
	}
	return response
}

func TestConfigReadNamesTheServingSnapshot(t *testing.T) {
	server, manager, _ := newSnapshotAPIServer(t, nil)
	recorder := httptest.NewRecorder()
	server.setupRoutes().ServeHTTP(recorder, httptest.NewRequest(http.MethodGet, apiConfigPath, nil))
	if recorder.Code != http.StatusOK {
		t.Fatalf("GET config: %d %s", recorder.Code, recorder.Body.String())
	}
	active := manager.Active()
	if got := recorder.Header().Get(headers.VSRConfigVersion); got != "1" {
		t.Fatalf("%s = %q, want 1", headers.VSRConfigVersion, got)
	}
	if got := recorder.Header().Get(headers.VSRConfigHash); got != active.Hash() || got == "" {
		t.Fatalf("%s = %q, want %q", headers.VSRConfigHash, got, active.Hash())
	}
	if hash := readConfigHash(t, server); hash.ActiveVersion != 1 || hash.LastRejection != nil {
		t.Fatalf("config hash = %+v", hash)
	}
}

func TestConfigMutationACKCarriesTheActivatedVersion(t *testing.T) {
	server, manager, path := newSnapshotAPIServer(t, nil)
	watchAndApply(t, manager, path)
	recorder, result := putConfig(t, server, path)
	if recorder.Code != http.StatusOK || result.ActivationStatus != "active" || result.ConfigVersion != 2 {
		t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
	}
	if result.Activation == nil || result.Activation.Version != 2 || result.Activation.Reasons != nil {
		t.Fatalf("activation = %+v", result.Activation)
	}
	if hash := readConfigHash(t, server); hash.ActiveVersion != 2 {
		t.Fatalf("config hash after ACK = %+v", hash)
	}
}

func TestConfigMutationNACKCarriesRedactedReasons(t *testing.T) {
	rejection := configsnapshot.Reject(configsnapshot.StageWarm, configsnapshot.CodeModelUnavailable,
		errors.New("model download failed: api_key: fake-credential-123"))
	server, manager, path := newSnapshotAPIServer(t, rejection)
	watchAndApply(t, manager, path)
	recorder, result := putConfig(t, server, path)
	if recorder.Code != http.StatusServiceUnavailable || result.Status != "activation_failed" || result.ConfigVersion != 0 {
		t.Fatalf("PUT: %d %s", recorder.Code, recorder.Body.String())
	}
	if strings.Contains(recorder.Body.String(), "fake-credential-123") {
		t.Fatalf("the NACK leaked a credential: %s", recorder.Body.String())
	}
	activation := result.Activation
	if activation == nil || len(activation.Reasons) != 1 {
		t.Fatalf("activation = %+v", activation)
	}
	reason := activation.Reasons[0]
	if reason.Stage != configsnapshot.StageWarm || reason.Code != configsnapshot.CodeModelUnavailable ||
		reason.Message != "model download failed: api_key: "+redactedConfigValue {
		t.Fatalf("reason = %+v", reason)
	}

	hash := readConfigHash(t, server)
	if hash.ActiveVersion != 1 || hash.LastRejection == nil || hash.LastRejection.Reasons[0].Message != reason.Message {
		t.Fatalf("config hash after NACK = %+v", hash)
	}
}
