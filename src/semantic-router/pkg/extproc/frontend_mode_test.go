package extproc

import (
	"errors"
	"net/http"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

func TestFrontendEngineStartsWithoutRoutingOrProviders(t *testing.T) {
	cfg, err := config.ParseYAMLBytes([]byte("version: v0.3\nglobal:\n  router:\n    enabled: false\n"))
	if err != nil {
		t.Fatal(err)
	}
	registry := routerruntime.NewRegistry(cfg)
	server, err := NewServer(t.TempDir()+"/config.yaml", 0, false, "", registry)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = server.service.Close() })
	if server.GetRouter() != nil || registry.ClassificationService() != nil {
		t.Fatal("Engine initialized a routing pipeline")
	}
	if server.CurrentConfig().RoutingEnabled() || registry.ConfigSnapshot() == nil {
		t.Fatal("Engine did not publish its configuration")
	}
	lease, err := server.Pin()
	if err != nil {
		t.Fatal(err)
	}
	lease.Release()
	if _, _, err = server.service.lease(); err == nil {
		t.Fatal("ExtProc must reject disabled routing")
	}
}

func TestFrontendModePublicationAndFailedEnablePreserveServingGeneration(t *testing.T) {
	server, registry, path := newLifecycleTestServer(t)
	stubPassingReload(t)
	disabled := false
	engine := &config.RouterConfig{RouterOptions: config.RouterOptions{RouterEnabled: &disabled}}
	writeReloadTestDocument(t, path, "engine", engine)
	parseReloadConfig = func(string) (*config.RouterConfig, error) { return engine, nil }
	ensureReloadConfigModels = func(*config.RouterConfig) error { t.Fatal("Engine downloaded routing models"); return nil }
	buildReloadRouter = func(*config.RouterConfig, ...*binding.Pool) (*OpenAIRouter, error) {
		t.Fatal("Engine built classifiers")
		return nil, nil
	}
	if err := server.reloadRouterFromFile(path); err != nil {
		t.Fatal(err)
	}
	if server.GetRouter() != nil || registry.CurrentConfig() != engine {
		t.Fatal("Engine generation not active")
	}
	engineSnapshot := server.service.Snapshot()
	lease, err := server.Pin()
	if err != nil {
		t.Fatal(err)
	}
	defer lease.Release()
	// Enabling routing may fail preparation. Native requests retain the valid
	// Engine generation and its model part while the candidate is discarded.
	candidate := &config.RouterConfig{}
	writeReloadTestDocument(t, path, "router", candidate)
	parseReloadConfig = func(string) (*config.RouterConfig, error) { return candidate, nil }
	ensureReloadConfigModels = func(*config.RouterConfig) error { return nil }
	buildReloadRouter = func(*config.RouterConfig, ...*binding.Pool) (*OpenAIRouter, error) {
		return nil, errors.New("invalid routing candidate")
	}
	if err := server.reloadRouterFromFile(path); err == nil {
		t.Fatal("invalid Router unexpectedly activated")
	}
	if server.service.Snapshot() != engineSnapshot || registry.CurrentConfig() != engine {
		t.Fatal("failed enable replaced the serving Engine")
	}
	buildReloadRouter = func(cfg *config.RouterConfig, _ ...*binding.Pool) (*OpenAIRouter, error) {
		return &OpenAIRouter{Config: cfg, resources: newResourceScope()}, nil
	}
	if err := server.reloadRouterFromFile(path); err != nil {
		t.Fatal(err)
	}
	if server.GetRouter() == nil || !registry.CurrentConfig().RoutingEnabled() {
		t.Fatal("Router did not reactivate")
	}
	if lease.Snapshot != engineSnapshot {
		t.Fatal("in-flight native request changed generations")
	}
}

func TestFrontendEngineExtProcDeclinesChatWithoutFailOpenTransportError(t *testing.T) {
	service := NewRouterService(nil)
	defer func() { _ = service.Close() }()
	stream := NewMockStream([]*ext_proc.ProcessingRequest{{Request: &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{}}}})
	if err := service.Process(stream); err != nil {
		t.Fatalf("transport error could fail open: %v", err)
	}
	if len(stream.Responses) != 1 || int(stream.Responses[0].GetImmediateResponse().GetStatus().GetCode()) != http.StatusNotFound {
		t.Fatalf("disabled routing response=%v", stream.Responses)
	}
}
