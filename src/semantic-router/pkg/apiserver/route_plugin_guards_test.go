//go:build !windows && cgo

package apiserver

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/pluginruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

type blockingPluginGuard struct {
	started chan struct{}
	finish  chan struct{}
}

func (g *blockingPluginGuard) PreviewResponseJailbreak(context.Context, pluginruntime.ResponseJailbreakPreviewRequest) (pluginruntime.GuardPreviewResponse, error) {
	close(g.started)
	<-g.finish
	return pluginruntime.GuardPreviewResponse{}, nil
}

func (g *blockingPluginGuard) PreviewHallucination(context.Context, pluginruntime.HallucinationPreviewRequest) (pluginruntime.GuardPreviewResponse, error) {
	return pluginruntime.GuardPreviewResponse{}, nil
}

func TestPluginPreviewCancellationRetainsRuntimeLease(t *testing.T) {
	guard := &blockingPluginGuard{started: make(chan struct{}), finish: make(chan struct{})}
	registry := routerruntime.NewRegistry(&config.RouterConfig{})
	var held atomic.Int32
	released := make(chan struct{})
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: &config.RouterConfig{}, Plugins: pluginruntime.Capabilities{Guards: guard}, AcquireClassification: func() (func(), bool) { held.Add(1); return func() { held.Add(-1); close(released) }, true }})
	server := &ClassificationAPIServer{runtimeRegistry: registry}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	req := httptest.NewRequest(http.MethodPost, apiPluginsPath+"/response_jailbreak/preview", strings.NewReader(`{"binding":{"recipe":"default","decision":"guard"},"response":"text","mode":"probe"}`)).WithContext(ctx)
	done := make(chan struct{})
	go func() { server.handleResponseJailbreakPreview(httptest.NewRecorder(), req); close(done) }()
	select {
	case <-guard.started:
	case <-time.After(time.Second):
		t.Fatal("preview worker did not start")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("HTTP handler did not stop after cancellation")
	}
	if held.Load() != 1 {
		t.Fatal("handler released generation while its classifier was still running")
	}
	close(guard.finish)
	select {
	case <-released:
	case <-time.After(time.Second):
		t.Fatal("completed preview worker leaked generation lease")
	}
	if held.Load() != 0 {
		t.Fatal("completed worker retained generation")
	}
}

func TestPluginPreviewDeadlineReturnsResponseAndRetainsAdmission(t *testing.T) {
	guard := &blockingPluginGuard{started: make(chan struct{}), finish: make(chan struct{})}
	timeout, concurrency := 1, 1
	cfg := &config.RouterConfig{}
	cfg.API.RoutingPreview = config.RoutingPreviewConfig{RequestTimeoutSeconds: &timeout, MaxConcurrency: &concurrency}
	registry := routerruntime.NewRegistry(cfg)
	var held atomic.Int32
	released := make(chan struct{}, 2)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, Plugins: pluginruntime.Capabilities{Guards: guard}, AcquireClassification: func() (func(), bool) {
		held.Add(1)
		return func() { held.Add(-1); released <- struct{}{} }, true
	}})
	api := &ClassificationAPIServer{runtimeRegistry: registry}
	server := httptest.NewUnstartedServer(api.setupRoutes())
	server.Config.WriteTimeout = 20 * time.Millisecond
	server.Start()
	t.Cleanup(func() { close(guard.finish); server.Close() })
	client := &http.Client{Timeout: 3 * time.Second}
	request := func() (int, string) {
		t.Helper()
		response, err := client.Post(server.URL+apiPluginsPath+"/response_jailbreak/preview", "application/json", strings.NewReader(`{"binding":{"recipe":"default","decision":"guard"},"response":"text","mode":"probe"}`))
		if err != nil {
			t.Fatal(err)
		}
		defer response.Body.Close()
		body, err := io.ReadAll(response.Body)
		if err != nil {
			t.Fatal(err)
		}
		return response.StatusCode, string(body)
	}
	status, body := request()
	if status != http.StatusGatewayTimeout || !strings.Contains(body, "REQUEST_TIMEOUT") || held.Load() != 1 {
		t.Fatalf("deadline failed or released native work: %d %s held=%d", status, body, held.Load())
	}
	status, body = request()
	if status != http.StatusTooManyRequests || held.Load() != 1 {
		t.Fatalf("deadline released admission: %d %s held=%d", status, body, held.Load())
	}
}
