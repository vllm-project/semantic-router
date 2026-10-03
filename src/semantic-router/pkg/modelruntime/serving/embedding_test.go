package serving

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// embedServices serves embedding and rerank cards. A vector is [len(input),
// dimension, layer, 1, 1, ...] truncated to the requested dimension; a rerank
// logit is the document's length.
type embedServices struct {
	mu      sync.Mutex
	cards   map[string]modelservice.ModelCard
	embeds  []modelservice.EmbedRequest
	reranks []modelservice.RerankRequest
	fail    string
}

func (f *embedServices) Card(_ context.Context, deployment string) (modelservice.ModelCard, error) {
	card, ok := f.cards[deployment]
	if !ok {
		return modelservice.ModelCard{}, modelservice.ErrUnknownDeployment
	}
	return card, nil
}

func (f *embedServices) Classify(context.Context, string, modelservice.ClassifyRequest) (modelservice.ClassifyResponse, error) {
	return modelservice.ClassifyResponse{}, errors.New("not used")
}

func (f *embedServices) Embed(_ context.Context, deployment string, request modelservice.EmbedRequest) (modelservice.EmbedResponse, error) {
	f.mu.Lock()
	f.embeds = append(f.embeds, request)
	fail := f.fail
	f.mu.Unlock()
	full := f.cards[deployment].Embedding.Dimensions[0]
	dimension := full
	if request.Dimensions > 0 {
		dimension = request.Dimensions
	}
	response := modelservice.EmbedResponse{Embeddings: make([][]float32, len(request.Inputs)), Inputs: make([]*modelservice.InputUsage, len(request.Inputs)), Errors: make([]string, len(request.Inputs))}
	for i, input := range request.Inputs {
		if fail != "" {
			response.Errors[i] = fail
			continue
		}
		vector := make([]float32, dimension)
		for j := range vector {
			vector[j] = 1
		}
		vector[0] = float32(len(input.Text + input.ImageURL + input.AudioWAV))
		vector[1] = float32(dimension)
		vector[2] = float32(request.Layer)
		response.Embeddings[i] = vector
		response.Inputs[i] = &modelservice.InputUsage{Tokens: 4, ProcessedTokens: 4}
	}
	return response, nil
}

func (f *embedServices) Rerank(_ context.Context, _ string, request modelservice.RerankRequest) (modelservice.RerankResponse, error) {
	f.mu.Lock()
	f.reranks = append(f.reranks, request)
	f.mu.Unlock()
	response := modelservice.RerankResponse{Results: make([]modelservice.RerankResult, len(request.Documents))}
	for i, document := range request.Documents {
		tokens := len(strings.Fields(request.Query + " " + document))
		response.Results[i] = modelservice.RerankResult{Index: i, Logit: float64(len(document)), Input: &modelservice.InputUsage{Tokens: tokens, ProcessedTokens: tokens}}
	}
	return response, nil
}

func (f *embedServices) embedCalls() []modelservice.EmbedRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	return append([]modelservice.EmbedRequest(nil), f.embeds...)
}

func embeddingCard(id string, modalities ...string) modelservice.ModelCard {
	return modelservice.ModelCard{
		ID: id, Repo: "vllm-sr/" + id, ModelSHA256: strings.Repeat("ab", 32), Revision: strings.Repeat("1", 40),
		Surfaces: []string{"embeddings"}, MaxInputTokens: 512, MaxInputs: 3, Device: "cpu", Dtype: "float32", Profile: "exact", Engine: "native",
		Embedding: &modelservice.EmbeddingCard{Dimensions: []int{8, 4}, Layers: []int{6, 22}, Modalities: modalities, Normalized: true, Pooling: "mean"},
	}
}

func embeddingSpec(deployment string) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: "embedding",
		Binding:    config.ModelBinding{Deployment: deployment, Contract: embeddingContract},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: "http://runtime:8100", Precision: "native", Device: "cpu", Profile: "exact", Input: config.ModelInputBudget{Overflow: "truncate"}},
	}
}

func TestEmbeddingServesViewsBatchesAndCaches(t *testing.T) {
	services := &embedServices{cards: map[string]modelservice.ModelCard{"emb-views": embeddingCard("emb-views", "text")}}
	runtime := New(services, nil)
	ctx := context.Background()
	provider, err := runtime.Embedding(ctx, embeddingSpec("emb-views"), 4, 6)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	if provider.Dimension() != 4 || provider.Backend() != Provider {
		t.Fatalf("dimension %d backend %s", provider.Dimension(), provider.Backend())
	}
	vector, err := provider.Embed(ctx, "hello there")
	if err != nil || len(vector) != 4 || vector[0] != 11 || vector[2] != 6 {
		t.Fatalf("default view: %v, %v", vector, err)
	}
	full, err := provider.EmbedWithOptions(ctx, "hello there", embedding.Options{})
	if err != nil || len(full) != 8 || full[2] != 0 {
		t.Fatalf("full view: %v, %v", full, err)
	}
	before := len(services.embedCalls())
	texts := []string{"a", "bb", "hello there", "ccc", "dddd", "a"}
	vectors, err := provider.EmbedBatch(ctx, texts)
	if err != nil || len(vectors) != len(texts) {
		t.Fatalf("batch: %v, %v", vectors, err)
	}
	for i, text := range texts {
		if vectors[i][0] != float32(len(text)) {
			t.Fatalf("batch vector %d = %v for %q", i, vectors[i], text)
		}
	}
	calls := services.embedCalls()[before:]
	embedded := 0
	for _, call := range calls {
		if len(call.Inputs) > 3 || call.Dimensions != 4 || call.Layer != 6 || call.Overflow != "truncate" {
			t.Fatalf("call %+v", call)
		}
		embedded += len(call.Inputs)
	}
	// "hello there" is cached from the first call; the repeated "a" is embedded once.
	if embedded != 4 || len(calls) != 2 {
		t.Fatalf("%d inputs in %d calls, want 4 in 2", embedded, len(calls))
	}
	vectors[0][0] = -1
	again, _ := provider.Embed(ctx, "a")
	if again[0] != 1 || len(services.embedCalls()) != before+2 {
		t.Fatalf("cached vector changed or recomputed: %v", again)
	}
}

func TestEmbeddingRejectsUndeclaredViewsAndSurfaces(t *testing.T) {
	classify := modelservice.ModelCard{ID: "classify-only", Surfaces: []string{"classify"}, Device: "cpu"}
	services := &embedServices{cards: map[string]modelservice.ModelCard{"emb": embeddingCard("emb"), "cls": classify}}
	runtime := New(services, nil)
	for _, tc := range []struct {
		deployment       string
		dimension, layer int
	}{{"emb", 5, 0}, {"emb", 0, 7}, {"cls", 0, 0}, {"missing", 0, 0}} {
		if provider, err := runtime.Embedding(context.Background(), embeddingSpec(tc.deployment), tc.dimension, tc.layer); err == nil {
			_ = provider.Close()
			t.Fatalf("%+v must be rejected", tc)
		}
	}
	if _, err := New(nil, nil).Embedding(context.Background(), embeddingSpec("emb"), 0, 0); !errors.Is(err, ErrNotConfigured) {
		t.Fatalf("no services: %v", err)
	}
}

func TestEmbeddingMapsItemErrors(t *testing.T) {
	services := &embedServices{cards: map[string]modelservice.ModelCard{"emb-errors": embeddingCard("emb-errors")}}
	provider, err := New(services, nil).Embedding(context.Background(), embeddingSpec("emb-errors"), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	services.mu.Lock()
	services.fail = "max_length_exceeded"
	services.mu.Unlock()
	if _, err := provider.Embed(context.Background(), "a text too long"); !errors.Is(err, binding.ErrInputLimit) {
		t.Fatalf("item error: %v", err)
	}
	if _, err := provider.Embed(context.Background(), "   "); !errors.Is(err, binding.ErrInvalidInput) {
		t.Fatalf("blank text: %v", err)
	}
}

func TestEmbeddingMediaInputs(t *testing.T) {
	services := &embedServices{cards: map[string]modelservice.ModelCard{"omni": embeddingCard("omni", "text", "image", "audio")}}
	provider, err := New(services, nil).Embedding(context.Background(), embeddingSpec("omni"), 0, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	if info := provider.EmbeddingInfo(); info.Audio == nil || len(info.Modalities) != 3 {
		t.Fatalf("info %+v", info)
	}
	png := []byte("\x89PNG\r\n\x1a\nrest")
	if _, err = provider.EmbedImage(context.Background(), png, 4); err != nil {
		t.Fatal(err)
	}
	pcm := []float32{0.5, -0.25, 0.125, 1}
	if _, err = provider.EmbedAudio(context.Background(), embedding.AudioRequest{PCM: pcm, SampleRate: 16000, Channels: 2}); err != nil {
		t.Fatal(err)
	}
	calls := services.embedCalls()
	image, audio := calls[len(calls)-2].Inputs[0], calls[len(calls)-1].Inputs[0]
	if !strings.HasPrefix(image.ImageURL, "data:image/png;base64,") || image.Text != "" {
		t.Fatalf("image input %+v", image)
	}
	wav, err := base64.StdEncoding.DecodeString(audio.AudioWAV)
	if err != nil {
		t.Fatal(err)
	}
	decoded, err := embedding.DecodeAudio(base64.StdEncoding.EncodeToString(wav))
	if err != nil || decoded.SampleRate != 16000 || decoded.Channels != 2 || len(decoded.PCM) != 4 {
		t.Fatalf("audio round trip: %+v, %v", decoded, err)
	}
	for i := range pcm {
		if decoded.PCM[i] != pcm[i] {
			t.Fatalf("sample %d = %v, want %v", i, decoded.PCM[i], pcm[i])
		}
	}
}

func TestEmbeddingRepresentationIdentity(t *testing.T) {
	services := &embedServices{cards: map[string]modelservice.ModelCard{"a": embeddingCard("a"), "b": embeddingCard("b")}}
	services.cards["b"] = func(card modelservice.ModelCard) modelservice.ModelCard {
		card.ModelSHA256 = strings.Repeat("cd", 32)
		return card
	}(services.cards["b"])
	runtime := New(services, nil)
	a, err := runtime.Embedding(context.Background(), embeddingSpec("a"), 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer a.Close()
	b, err := runtime.Embedding(context.Background(), embeddingSpec("b"), 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer b.Close()
	view, _ := a.RepresentationIdentity(embedding.Options{Dimension: 4}, "memory-v1")
	same, _ := a.RepresentationIdentity(embedding.Options{Dimension: 4}, "memory-v1")
	full, _ := a.RepresentationIdentity(embedding.Options{}, "memory-v1")
	policy, _ := a.RepresentationIdentity(embedding.Options{Dimension: 4}, "cache-v1")
	other, _ := b.RepresentationIdentity(embedding.Options{Dimension: 4}, "memory-v1")
	if view.Fingerprint == "" || view.Fingerprint != same.Fingerprint || view.Descriptor.Dimension != 4 || full.Descriptor.Dimension != 8 {
		t.Fatalf("view %+v full %+v", view, full)
	}
	for _, distinct := range []embedding.ContentIdentity{full, policy, other} {
		if distinct.Fingerprint == view.Fingerprint {
			t.Fatalf("distinct representations share %s", view.Fingerprint)
		}
	}
}

func TestDiagnoseEmbedding(t *testing.T) {
	services := &embedServices{cards: map[string]modelservice.ModelCard{"emb-diag": embeddingCard("emb-diag")}}
	runtime := New(services, nil)
	provider, err := runtime.Embedding(context.Background(), embeddingSpec("emb-diag"), 4, 0)
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	result, err := runtime.DiagnoseEmbedding(context.Background(), string(config.DefaultRecipeName), "embedding", "diagnose me")
	if err != nil || len(result.Result.Embedding) != 4 || result.Binding.Identity.Deployment != "emb-diag" {
		t.Fatalf("diagnostic %+v, %v", result, err)
	}
}

func rerankSpec(deployment, overflow string, selection *config.PairScorerSelection) config.ResolvedModelBinding {
	spec := embeddingSpec(deployment)
	spec.Name = config.RAGRerankerConsumer
	spec.Binding.Contract = config.RelevanceScoresContract
	spec.Binding.PairScorer = selection
	spec.Deployment.Input.Overflow = overflow
	return spec
}

func TestRelevanceScoresPairsPerQueryAtTheSelectedExit(t *testing.T) {
	card := modelservice.ModelCard{
		ID: "reranker", ModelSHA256: strings.Repeat("ef", 32), Surfaces: []string{"rerank"}, MaxInputTokens: 512, Device: "cpu",
		Rerank: &modelservice.RerankCard{Default: modelservice.RerankExit{Layer: 22, Dimension: 768}, Exits: []modelservice.RerankExit{{Layer: 6, Dimension: 256}}},
	}
	services := &embedServices{cards: map[string]modelservice.ModelCard{"reranker": card}}
	runtime := New(services, nil)
	if _, err := runtime.Relevance(context.Background(), rerankSpec("reranker", "truncate", nil)); err == nil {
		t.Fatal("a truncating reranker must be rejected")
	}
	if _, err := runtime.Relevance(context.Background(), rerankSpec("reranker", "reject", &config.PairScorerSelection{Layer: 6, Dimension: 768})); err == nil {
		t.Fatal("an undeclared exit must be rejected")
	}
	scorer, err := runtime.Relevance(context.Background(), rerankSpec("reranker", "reject", &config.PairScorerSelection{Layer: 6}))
	if err != nil {
		t.Fatal(err)
	}
	defer scorer.Close()
	if scorer.Selection() != (config.PairScorerSelection{Layer: 6, Dimension: 256}) || scorer.CacheIdentity() == "" {
		t.Fatalf("selection %+v identity %q", scorer.Selection(), scorer.CacheIdentity())
	}
	pairs := []tasks.QueryDocument{{Query: "q1", Document: "aa"}, {Query: "q1", Document: "bbbb"}, {Query: "q2", Document: "c"}}
	scores, err := scorer.ScorePairs(context.Background(), string(config.DefaultRecipeName), pairs)
	if err != nil {
		t.Fatal(err)
	}
	if scores.Scores[0] != 2 || scores.Scores[1] != 4 || scores.Scores[2] != 1 || scores.Inputs[2].OriginalTokens != 2 {
		t.Fatalf("scores %+v", scores)
	}
	services.mu.Lock()
	reranks := services.reranks[1:]
	services.mu.Unlock()
	if len(reranks) != 2 || len(reranks[0].Documents) != 2 || reranks[0].Layer != 6 || reranks[0].Dimensions != 256 || reranks[0].Overflow != "reject" {
		t.Fatalf("rerank calls %+v", reranks)
	}
	diagnostic, err := runtime.DiagnoseRerank(context.Background(), string(config.DefaultRecipeName), config.RAGRerankerConsumer, pairs[:1])
	if err != nil || len(diagnostic.Result.Scores) != 1 {
		t.Fatalf("diagnostic %+v, %v", diagnostic, err)
	}
}

func TestRemoteEmbeddingBatchesAndCaches(t *testing.T) {
	var mu sync.Mutex
	var requests [][]string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var body struct {
			Input []string `json:"input"`
		}
		_ = json.NewDecoder(r.Body).Decode(&body)
		mu.Lock()
		requests = append(requests, body.Input)
		mu.Unlock()
		data := make([]map[string]interface{}, len(body.Input))
		for i, text := range body.Input {
			data[i] = map[string]interface{}{"index": i, "embedding": []float64{float64(len(text)), 1, 1}}
		}
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"data": data})
	}))
	defer server.Close()
	spec := embeddingSpec("embedding:remote")
	spec.Deployment = config.ModelDeployment{Provider: "http"}
	provider, err := New(nil, nil).RemoteEmbedding(context.Background(), spec, embedding.OpenAICompatibleConfig{BaseURL: server.URL, Model: "remote-model"})
	if err != nil {
		t.Fatal(err)
	}
	defer provider.Close()
	vectors, err := provider.EmbedBatch(context.Background(), []string{"x", "yy", "x"})
	if err != nil || len(vectors) != 3 || vectors[1][0] != 2 || vectors[2][0] != 1 {
		t.Fatalf("vectors %v, %v", vectors, err)
	}
	if _, err := provider.Embed(context.Background(), "yy"); err != nil {
		t.Fatal(err)
	}
	mu.Lock()
	defer mu.Unlock()
	if len(requests) != 2 || len(requests[1]) != 2 || provider.Dimension() != 3 {
		t.Fatalf("endpoint requests %v (warmup, then one batch of distinct texts)", requests)
	}
}
