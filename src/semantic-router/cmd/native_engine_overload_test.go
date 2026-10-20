//go:build !windows

package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/extproc"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/gateway"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestEngineFrontendAnswersAFullWorkerPoolAsOverloaded(t *testing.T) {
	fake := runtimetest.New(runtimetest.Model{ID: "served", Repo: "example/vela"})
	var held atomic.Int32
	release := make(chan struct{})
	worker := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/v1/decisions" {
			held.Add(1)
			<-release
		}
		fake.Handler().ServeHTTP(w, r)
	}))
	defer worker.Close()
	var releaseOnce sync.Once
	releaseAll := func() { releaseOnce.Do(func() { close(release) }) }
	// Runs before worker.Close, so a failed admission does not leave handlers blocked.
	defer releaseAll()
	manager := modelservice.NewManager()
	previous := modelservice.DefaultManager()
	modelservice.SetDefault(manager)
	t.Cleanup(func() { modelservice.SetDefault(previous); _ = manager.Shutdown(context.Background()) })
	disabled := false
	cfg := &config.RouterConfig{RouterOptions: config.RouterOptions{RouterEnabled: &disabled}}
	cfg.DecisionModel = "primary"
	cfg.ModelDeployments = map[string]config.ModelDeployment{"primary": {Provider: config.ModelRuntimeProvider, Artifact: "example/vela", Endpoint: worker.URL, ServedName: "served"}}
	cfg.Listeners = []config.Listener{{Name: "native", Port: 8899, APIKeys: []string{"key"}, SystemOne: &config.ListenerSystemOne{Models: []string{"example/vela"}}}}
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if _, err := manager.Published().Card(ctx, "primary"); err != nil {
		t.Fatal(err)
	}
	server, err := extproc.NewServer(t.TempDir()+"/config.yaml", 0, false, "", routerruntime.NewRegistry(cfg), extproc.WithGatewayMode(config.GatewayStandalone))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.Shutdown(context.Background()) })
	handler, err := gateway.NewHandler(gateway.Options{Listener: "native", Serving: nativeServing{pin: server.Pin}})
	if err != nil {
		t.Fatal(err)
	}
	ask := func() *httptest.ResponseRecorder {
		request := httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(`{"model":"example/vela","state":"A question","questions":{"q":{"type":"noul","instructions":"Is this a question?"}}}`))
		request.Header.Set("Authorization", "Bearer key")
		response := httptest.NewRecorder()
		handler.ServeHTTP(response, request)
		return response
	}
	// One worker admits 32 requests (replicaAdmissionLimit in pkg/modelservice); hold them all in the worker.
	var admitted sync.WaitGroup
	for range 32 {
		admitted.Add(1)
		go func() { defer admitted.Done(); ask() }()
	}
	for deadline := time.Now().Add(5 * time.Second); held.Load() < 32; time.Sleep(5 * time.Millisecond) {
		if time.Now().After(deadline) {
			t.Fatalf("worker holds %d requests", held.Load())
		}
	}
	refused := ask()
	releaseAll()
	drained := make(chan struct{})
	go func() { admitted.Wait(); close(drained) }()
	select {
	case <-drained:
	case <-time.After(5 * time.Second):
		t.Fatal("held requests did not finish after release")
	}
	if refused.Code != http.StatusTooManyRequests || !strings.Contains(refused.Body.String(), `"overloaded"`) || refused.Header().Get("Retry-After") != "1" {
		t.Fatalf("full pool: %d %q Retry-After %q", refused.Code, refused.Body.String(), refused.Header().Get("Retry-After"))
	}
	if answered := ask(); answered.Code != http.StatusOK {
		t.Fatalf("after the pool drained: %d %s", answered.Code, answered.Body.String())
	}
}
