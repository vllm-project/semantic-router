// Package servingtest prepares a serving.Runtime whose deployments are served
// by a fake runtime process (runtimetest), for consumer tests that exercise
// the real lease, client, bundles and typed bindings.
package servingtest

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// Runtime serves every named deployment (configured or implicit, like
// "@domain_classifier") with its fake model from one fake runtime process.
func Runtime(t testing.TB, deployments map[string]runtimetest.Model) (*serving.Runtime, *runtimetest.Runtime) {
	t.Helper()
	models := make([]runtimetest.Model, 0, len(deployments))
	attached := make(map[string]config.ModelDeployment, len(deployments))
	for name, model := range deployments {
		if model.ID == "" {
			model.ID = name
		}
		models = append(models, model)
		attached[name] = config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "", Model: model.ID}
	}
	fake := runtimetest.New(models...)
	server := httptest.NewServer(fake.Handler())
	t.Cleanup(server.Close)
	for name, deployment := range attached {
		deployment.Endpoint = server.URL
		attached[name] = deployment
	}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(attached)
	if err != nil {
		t.Fatal(err)
	}
	// Preparation checks an attached deployment against the card the lease's
	// watcher has observed, and skips one it has not observed yet. Return
	// once every card is observed, as from a runtime that is already serving.
	ctx, cancel := context.WithTimeout(context.Background(), modelservice.ReadyTimeout())
	defer cancel()
	for name := range attached {
		if _, err := lease.Card(ctx, name); err != nil {
			t.Fatal(err)
		}
	}
	return serving.New(lease, nil), fake
}
