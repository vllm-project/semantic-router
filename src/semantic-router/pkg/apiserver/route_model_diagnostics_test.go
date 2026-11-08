//go:build !windows

package apiserver

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
)

// A fake model runtime verifies HTTP->live generation->prepared inference.
// These are not quality or hardware tests of the published Vela artifacts.

func diagnosticRequest(t *testing.T, server *httptest.Server, path, body string) (int, []byte) {
	t.Helper()
	response, err := server.Client().Post(server.URL+apiDiagnosticsPath+"/models/"+path, "application/json", strings.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatal(err)
	}
	return response.StatusCode, data
}

// preparedEmbedding prepares the embedding binding of a recipe on a fake
// runtime whose embedder answers vectors of width dimension.
func preparedEmbedding(t *testing.T, recipe string, dimension int) (*serving.Runtime, *runtimetest.Runtime) {
	t.Helper()
	runtime, fake := servingtest.Runtime(t, map[string]runtimetest.Model{
		"embed": {Embedding: &runtimetest.Embedder{Dimensions: []int{dimension, 2 * dimension}, Layers: []int{7}, Modalities: []string{"text", "audio"}}},
	})
	provider, err := runtime.Embedding(context.Background(), config.ResolvedModelBinding{
		Recipe: config.RecipeName(recipe), Name: "embedding", Contract: "embedding.v1", Binding: config.ModelBinding{Deployment: "embed"},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime:8100"},
	}, dimension, 7)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = provider.Close() })
	return runtime, fake
}

func TestModelDiagnosticsHTTPRunsThePreparedEmbeddingOfAnExplicitRecipe(t *testing.T) {
	runtime, fake := preparedEmbedding(t, string(config.DefaultRecipeName), 2)
	cfg, service := preparedInventoryServiceWith(t, classification.RecipeRuntimeOptions{Runtime: runtime}, &config.RouterConfig{Recipes: []config.RoutingRecipe{{Name: config.DefaultRecipeName}, {Name: "private"}}})
	registry := routerruntime.NewRegistry(cfg)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: service})
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes())
	t.Cleanup(server.Close)

	calls := fake.Calls("embeddings")
	status, body := diagnosticRequest(t, server, "embeddings", `{"recipe":"default","binding":"embedding","text":"query"}`)
	var embedded ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult]
	if err := json.Unmarshal(body, &embedded); err != nil {
		t.Fatal(err)
	}
	if status != 200 || len(embedded.Result.Embedding) != 2 || fake.Calls("embeddings") != calls+1 || embedded.Binding.Recipe != "default" ||
		!strings.Contains(string(body), `"dimensions":[2,4]`) || strings.Contains(string(body), `"max_seconds"`) {
		t.Fatalf("embedding: %d %s", status, body)
	}
	for _, bad := range []struct {
		body   string
		status int
	}{
		{`{"binding":"embedding","text":"query"}`, 400},
		{`{"recipe":"foreign","binding":"embedding","text":"query"}`, 404},
		{`{"recipe":"private","binding":"embedding","text":"query"}`, 404},
		{`{"recipe":"default","binding":"unloaded","text":"query"}`, 404},
		{`{"recipe":"default","binding":"embedding","text":"query","artifact":"arbitrary"}`, 400},
	} {
		badStatus, badBody := diagnosticRequest(t, server, "embeddings", bad.body)
		if badStatus != bad.status {
			t.Fatalf("invalid scope/input %s: %d %s", bad.body, badStatus, badBody)
		}
	}
	if fake.Calls("embeddings") != calls+1 {
		t.Fatal("invalid request executed a model")
	}
	response, err := server.Client().Get(server.URL + apiDiagnosticsPath + "/models?recipe=default")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var inventory ModelDiagnosticInventory
	if err = json.NewDecoder(response.Body).Decode(&inventory); err != nil {
		t.Fatal(err)
	}
	if response.StatusCode != 200 || len(inventory.Bindings) != 1 || inventory.Bindings[0].Name != "embedding" || inventory.Bindings[0].Embedding == nil {
		t.Fatalf("inventory = %+v", inventory)
	}
}

func TestModelDiagnosticsHoldsLiveGenerationThroughInferenceAndUsesReplacement(t *testing.T) {
	oldRuntime, oldFake := preparedEmbedding(t, string(config.DefaultRecipeName), 2)
	nextRuntime, _ := preparedEmbedding(t, string(config.DefaultRecipeName), 4)
	cfg, oldService := preparedInventoryService(t, oldRuntime)
	_, nextService := preparedInventoryService(t, nextRuntime)
	var leases atomic.Int32
	registry := routerruntime.NewRegistry(cfg)
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: oldService, AcquireClassification: func() (func(), bool) {
		leases.Add(1)
		return func() { leases.Add(-1) }, true
	}})
	server := httptest.NewServer((&ClassificationAPIServer{config: cfg, runtimeRegistry: registry}).setupRoutes())
	t.Cleanup(server.Close)
	oldFake.SetDelay(300 * time.Millisecond)
	result := make(chan []byte, 1)
	go func() {
		_, body := diagnosticRequest(t, server, "embeddings", `{"recipe":"default","binding":"embedding","text":"query"}`)
		result <- body
	}()
	until := time.Now().Add(2 * time.Second)
	for leases.Load() == 0 && time.Now().Before(until) {
		time.Sleep(time.Millisecond)
	}
	if leases.Load() != 1 {
		t.Fatal("the old generation was never leased")
	}
	registry.PublishRouterRuntimeSnapshot(routerruntime.RouterRuntimeSnapshot{Config: cfg, ClassificationService: nextService})
	var answered ModelDiagnosticResponse[ModelDiagnosticEmbeddingResult]
	select {
	case body := <-result:
		if err := json.Unmarshal(body, &answered); err != nil || len(answered.Result.Embedding) != 2 {
			t.Fatalf("mixed generation: %s", body)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("old inference did not finish")
	}
	if leases.Load() != 0 {
		t.Fatal("generation lease leaked")
	}
	status, body := diagnosticRequest(t, server, "embeddings", `{"recipe":"default","binding":"embedding","text":"query"}`)
	if err := json.Unmarshal(body, &answered); err != nil || status != 200 || len(answered.Result.Embedding) != 4 {
		t.Fatalf("replacement not used: %d %s", status, body)
	}
}

func TestModelDiagnosticsDeadlineRetainsGenerationUntilTheCallReturns(t *testing.T) {
	var leases atomic.Int32
	leases.Add(1)
	entered, finish := make(chan struct{}), make(chan struct{})
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Millisecond)
	defer cancel()
	request := httptest.NewRequest(http.MethodPost, "/", nil).WithContext(ctx)
	done := make(chan error, 1)
	go func() {
		_, err := executeModelDiagnostic(httptest.NewRecorder(), request, func() { leases.Add(-1) }, func(ctx context.Context) (int, error) {
			if deadline, ok := ctx.Deadline(); !ok || time.Until(deadline) > apiWriteTimeout {
				t.Error("the API did not bound the diagnostic inference context")
			}
			close(entered)
			<-ctx.Done()
			<-finish
			return 0, nil
		})
		done <- err
	}()
	select {
	case <-entered:
	case <-time.After(time.Second):
		close(finish)
		t.Fatal("inference did not begin")
	}
	select {
	case err := <-done:
		if err == nil {
			close(finish)
			t.Fatal("a call past its deadline succeeded")
		}
	case <-time.After(time.Second):
		close(finish)
		t.Fatal("the diagnostic did not respect its deadline")
	}
	if leases.Load() != 1 {
		close(finish)
		t.Fatal("the deadline released the generation while the call ran")
	}
	close(finish)
	until := time.Now().Add(time.Second)
	for leases.Load() != 0 && time.Now().Before(until) {
		time.Sleep(time.Millisecond)
	}
	if leases.Load() != 0 {
		t.Fatal("the completed call retained its generation")
	}
}
