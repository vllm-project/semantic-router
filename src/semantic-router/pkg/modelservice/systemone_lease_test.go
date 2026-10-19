package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestSystemOneArtifactMismatchDoesNotInvokeRuntime(t *testing.T) {
	calls := 0
	manager := systemOneManager(t, func(w http.ResponseWriter, r *http.Request) {
		calls++
		_, _ = io.WriteString(w, `{"answers":{}}`)
	})
	_, err := manager.SystemOneForArtifact(context.Background(), "test-systemone", "different/new-model", json.RawMessage(`{}`))
	if !errors.Is(err, ErrSystemOneArtifactChanged) || calls != 0 {
		t.Fatalf("mismatch invoked runtime: calls=%d err=%v", calls, err)
	}
}

func TestSystemOneLeaseDoesNotFollowPublishedGeneration(t *testing.T) {
	manager := systemOneManager(t, func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, `{"model":"served","answers":{}}`)
	})
	status := manager.Published().Statuses()[0]
	retained, err := manager.AcquireDeployments(map[string]config.ModelDeployment{"test-systemone": {Provider: config.ModelRuntimeProvider, Endpoint: status.Endpoint, ServedName: "served"}})
	if err != nil {
		t.Fatal(err)
	}
	defer retained.Close()
	if err = manager.Reconcile(&config.RouterConfig{}); err != nil {
		t.Fatal(err)
	}
	result, err := retained.SystemOne(context.Background(), "test-systemone", json.RawMessage(`{"state":"text","questions":{}}`))
	if err != nil || result.Status != 200 {
		t.Fatalf("retained generation lost: %+v %v", result, err)
	}
	if _, err = manager.SystemOne(context.Background(), "test-systemone", json.RawMessage(`{}`)); err == nil {
		t.Fatal("test must remove deployment from published generation")
	}
	_ = retained.Close()
	if _, err = retained.SystemOne(context.Background(), "test-systemone", json.RawMessage(`{}`)); err == nil {
		t.Fatal("closed generation remained callable")
	}
}
