package modelruntime

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"slices"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// The shared fixture normally supplies a digest. Remove the optional identity
// fields on the wire to exercise discovery of an unhashed attached runtime.
func newUnhashedEmbeddingRuntime(t *testing.T, id string, dimension int) (*runtimetest.Runtime, *httptest.Server) {
	t.Helper()
	fake := runtimetest.New(runtimetest.Model{ID: id, Embedding: &runtimetest.Embedder{Dimensions: []int{dimension}}})
	handler := fake.Handler()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			handler.ServeHTTP(w, r)
			return
		}
		recorder := httptest.NewRecorder()
		handler.ServeHTTP(recorder, r)
		var list map[string]json.RawMessage
		if err := json.Unmarshal(recorder.Body.Bytes(), &list); err != nil {
			t.Errorf("decode model list: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		var cards []map[string]json.RawMessage
		if err := json.Unmarshal(list["data"], &cards); err != nil {
			t.Errorf("decode model cards: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		for _, card := range cards {
			delete(card, "model_sha256")
			delete(card, "revision")
		}
		data, err := json.Marshal(cards)
		if err != nil {
			t.Errorf("encode model cards: %v", err)
			w.WriteHeader(http.StatusInternalServerError)
			return
		}
		list["data"] = data
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(recorder.Code)
		if err := json.NewEncoder(w).Encode(list); err != nil {
			t.Errorf("encode model list: %v", err)
		}
	}))
	t.Cleanup(server.Close)
	return fake, server
}

func TestOwnedEmbeddingAPICacheSeparatesUnhashedAttachedModels(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()
	a, serverA := newUnhashedEmbeddingRuntime(t, "model-a", 8)
	b, serverB := newUnhashedEmbeddingRuntime(t, "model-b", 4)
	manager := modelservice.NewManager()
	defer func() { _ = manager.Shutdown(context.Background()) }()
	text := t.Name() + " identical text"
	for _, model := range []struct {
		id        string
		dimension int
		fake      *runtimetest.Runtime
		server    *httptest.Server
	}{
		{id: "model-a", dimension: 8, fake: a, server: serverA},
		{id: "model-b", dimension: 4, fake: b, server: serverB},
	} {
		cfg := &config.RouterConfig{}
		cfg.API.Embeddings.Enabled = true
		cfg.EmbeddingConfig.ModelType = "qwen3"
		cfg.ModelDeployments = map[string]config.ModelDeployment{
			model.id: {Provider: config.ModelRuntimeProvider, Endpoint: model.server.URL, ServedName: model.id},
		}
		cfg.GlobalModelBindings = map[string]config.ModelBinding{"embedding": {Deployment: model.id, Contract: "embedding.v1"}}
		lease, err := manager.Acquire(cfg)
		if err != nil {
			t.Fatal(err)
		}
		defer lease.Close()
		prepared, err := PrepareOwnedEmbeddingAPI(ctx, cfg, serving.New(lease, nil))
		if err != nil {
			t.Fatal(err)
		}
		defer prepared.Close()
		card, err := lease.Card(ctx, model.id)
		if err != nil {
			t.Fatal(err)
		}
		if card.ID != model.id || !card.Ready || card.ModelSHA256 != "" || card.Revision != "" {
			t.Fatalf("expected a discovered ready unhashed %s card, got %+v", model.id, card)
		}
		provider, err := prepared.Default()
		if err != nil {
			t.Fatal(err)
		}
		// Preparation warms up the backend; count only calls for the shared text.
		before := model.fake.Calls("embeddings")
		vector, err := provider.Embed(ctx, text)
		if err != nil {
			t.Fatal(err)
		}
		calls := model.fake.Calls("embeddings") - before
		t.Logf("%s: advertised width=%d, returned width=%d, backend calls=%d", model.id, provider.Dimension(), len(vector), calls)
		if provider.Dimension() != model.dimension || len(vector) != model.dimension {
			t.Errorf("%s: advertised width=%d, returned width=%d, want %d", model.id, provider.Dimension(), len(vector), model.dimension)
		}
		if calls != 1 {
			t.Errorf("%s: made %d backend calls for the shared text, want 1", model.id, calls)
		}
		repeated, err := provider.Embed(ctx, text)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(vector, repeated) || model.fake.Calls("embeddings") != before+1 {
			t.Errorf("%s: repeated embedding did not reuse its own cached vector", model.id)
		}
	}
}
