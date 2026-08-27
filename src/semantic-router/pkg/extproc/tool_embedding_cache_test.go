package extproc

import (
	"context"
	"fmt"
	"hash/fnv"
	"reflect"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/packages/param"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// countingEmbeddingProvider is a deterministic fake provider that records how it
// was called, so tests can assert on batching and cache behaviour rather than on
// embedding values.
type countingEmbeddingProvider struct {
	mu         sync.Mutex
	embedCalls int
	batchCalls int
	batchSizes []int
	embedded   []string
}

// vectorFor derives a stable unit-ish 4-dim vector from the text, so identical
// text always yields an identical vector and different text almost never does.
func vectorFor(text string) []float32 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(text))
	sum := h.Sum64()
	vec := make([]float32, 4)
	for i := range vec {
		vec[i] = float32((sum>>(i*8))&0xff) / 255.0
	}
	return vec
}

func (p *countingEmbeddingProvider) Embed(_ context.Context, text string) ([]float32, error) {
	p.mu.Lock()
	p.embedCalls++
	p.embedded = append(p.embedded, text)
	p.mu.Unlock()
	return vectorFor(text), nil
}

func (p *countingEmbeddingProvider) EmbedBatch(_ context.Context, texts []string) ([][]float32, error) {
	p.mu.Lock()
	p.batchCalls++
	p.batchSizes = append(p.batchSizes, len(texts))
	p.embedded = append(p.embedded, texts...)
	p.mu.Unlock()

	out := make([][]float32, len(texts))
	for i, text := range texts {
		out[i] = vectorFor(text)
	}
	return out, nil
}

func (p *countingEmbeddingProvider) Dimension() int { return 4 }

func (p *countingEmbeddingProvider) Backend() string {
	return config.EmbeddingBackendOpenAICompatible
}

func (p *countingEmbeddingProvider) snapshot() (batchCalls int, batchSizes []int, embedded []string) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.batchCalls, append([]int(nil), p.batchSizes...), append([]string(nil), p.embedded...)
}

func (p *countingEmbeddingProvider) reset() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.embedCalls = 0
	p.batchCalls = 0
	p.batchSizes = nil
	p.embedded = nil
}

func newTestToolEmbedder(provider *countingEmbeddingProvider) *cachedToolEmbedder {
	return newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 4, "")
}

func toolTextsFor(n int) []string {
	texts := make([]string, n)
	for i := range texts {
		texts[i] = fmt.Sprintf("tool_%d does job number %d", i, i)
	}
	return texts
}

func TestToolEmbedderColdCallBatchesOnceAndAligns(t *testing.T) {
	provider := &countingEmbeddingProvider{}
	emb := newTestToolEmbedder(provider)
	toolTexts := toolTextsFor(10)

	queryVec, toolVecs, err := emb.embedQueryAndTools(context.Background(), "the query", toolTexts)
	if err != nil {
		t.Fatalf("embedQueryAndTools failed: %v", err)
	}

	batchCalls, batchSizes, _ := provider.snapshot()
	if batchCalls != 1 {
		t.Fatalf("EmbedBatch called %d times, want exactly 1 for %d unique tools + query", batchCalls, len(toolTexts))
	}
	if batchSizes[0] != len(toolTexts)+1 {
		t.Fatalf("batch size = %d, want %d (tools + query)", batchSizes[0], len(toolTexts)+1)
	}
	if provider.embedCalls != 0 {
		t.Fatalf("single-text Embed called %d times, want 0 (batching only)", provider.embedCalls)
	}

	if !reflect.DeepEqual(queryVec, vectorFor("the query")) {
		t.Fatalf("query embedding = %v, want %v", queryVec, vectorFor("the query"))
	}
	for i, text := range toolTexts {
		if !reflect.DeepEqual(toolVecs[i], vectorFor(text)) {
			t.Fatalf("tool %d embedding = %v, want the vector for %q", i, toolVecs[i], text)
		}
	}
}

func TestToolEmbedderWarmCallEmbedsOnlyTheQuery(t *testing.T) {
	provider := &countingEmbeddingProvider{}
	emb := newTestToolEmbedder(provider)
	toolTexts := toolTextsFor(10)

	_, coldVecs, err := emb.embedQueryAndTools(context.Background(), "first query", toolTexts)
	if err != nil {
		t.Fatalf("cold embedQueryAndTools failed: %v", err)
	}
	provider.reset()

	queryVec, warmVecs, err := emb.embedQueryAndTools(context.Background(), "a different second query", toolTexts)
	if err != nil {
		t.Fatalf("warm embedQueryAndTools failed: %v", err)
	}

	batchCalls, batchSizes, embedded := provider.snapshot()
	if batchCalls != 1 || len(batchSizes) != 1 || batchSizes[0] != 1 {
		t.Fatalf("warm call issued batches %v (calls=%d), want exactly one batch of 1 (the query)", batchSizes, batchCalls)
	}
	if len(embedded) != 1 || embedded[0] != "a different second query" {
		t.Fatalf("warm call embedded %v, want only the query text", embedded)
	}
	if !reflect.DeepEqual(queryVec, vectorFor("a different second query")) {
		t.Fatalf("warm query embedding mismatch")
	}

	if !reflect.DeepEqual(coldVecs, warmVecs) {
		t.Fatalf("cached tool embeddings differ from the cold ones; ranking would not be bit-identical")
	}
}

func TestToolEmbedderDeduplicatesRepeatedToolText(t *testing.T) {
	provider := &countingEmbeddingProvider{}
	emb := newTestToolEmbedder(provider)
	toolTexts := []string{"same tool text", "other tool", "same tool text"}

	_, toolVecs, err := emb.embedQueryAndTools(context.Background(), "query", toolTexts)
	if err != nil {
		t.Fatalf("embedQueryAndTools failed: %v", err)
	}

	_, batchSizes, embedded := provider.snapshot()
	if batchSizes[0] != 3 {
		t.Fatalf("batch size = %d, want 3 (2 unique tools + query)", batchSizes[0])
	}
	occurrences := 0
	for _, text := range embedded {
		if text == "same tool text" {
			occurrences++
		}
	}
	if occurrences != 1 {
		t.Fatalf("duplicate tool text embedded %d times, want 1", occurrences)
	}
	if !reflect.DeepEqual(toolVecs[0], toolVecs[2]) {
		t.Fatalf("duplicate tool indices got different vectors")
	}
	if !reflect.DeepEqual(toolVecs[0], vectorFor("same tool text")) {
		t.Fatalf("duplicate tool vector is not the vector for its text")
	}
}

func TestToolEmbedderChunksLargeMissSets(t *testing.T) {
	provider := &countingEmbeddingProvider{}
	emb := newTestToolEmbedder(provider)
	toolTexts := toolTextsFor(150)

	_, toolVecs, err := emb.embedQueryAndTools(context.Background(), "query", toolTexts)
	if err != nil {
		t.Fatalf("embedQueryAndTools failed: %v", err)
	}

	batchCalls, batchSizes, _ := provider.snapshot()
	if batchCalls < 2 {
		t.Fatalf("EmbedBatch called %d times for %d texts, want chunking into multiple calls", batchCalls, len(toolTexts)+1)
	}
	total := 0
	for _, size := range batchSizes {
		if size > toolEmbedBatchChunkSize {
			t.Fatalf("chunk of %d texts exceeds the %d cap (sizes=%v)", size, toolEmbedBatchChunkSize, batchSizes)
		}
		total += size
	}
	if total != len(toolTexts)+1 {
		t.Fatalf("chunks covered %d texts, want %d", total, len(toolTexts)+1)
	}

	for i, text := range toolTexts {
		if !reflect.DeepEqual(toolVecs[i], vectorFor(text)) {
			t.Fatalf("chunked fill misaligned tool %d: got %v, want the vector for %q", i, toolVecs[i], text)
		}
	}
}

func TestToolEmbedderPropagatesFillErrors(t *testing.T) {
	provider := &failingEmbeddingProvider{}
	emb := newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 4, "")

	if _, _, err := emb.embedQueryAndTools(context.Background(), "query", []string{"tool one"}); err == nil {
		t.Fatalf("embedQueryAndTools succeeded despite a provider error")
	}
	if _, ok := emb.memo.Get(emb.key("tool one")); ok {
		t.Fatalf("a failed fill memoized an entry; the failure would not be retried")
	}
}

// batchRejectingProvider fails every multi-text EmbedBatch (like an endpoint
// with a per-request token cap) while single-text embedding succeeds.
type batchRejectingProvider struct {
	countingEmbeddingProvider
}

func (p *batchRejectingProvider) EmbedBatch(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) > 1 {
		return nil, fmt.Errorf("batch too large")
	}
	return p.countingEmbeddingProvider.EmbedBatch(ctx, texts)
}

func TestToolEmbedderFallsBackToPerTextWhenBatchRejected(t *testing.T) {
	provider := &batchRejectingProvider{}
	emb := newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 4, "")
	toolTexts := toolTextsFor(3)

	queryVec, toolVecs, err := emb.embedQueryAndTools(context.Background(), "the query", toolTexts)
	if err != nil {
		t.Fatalf("embedQueryAndTools failed despite per-text fallback: %v", err)
	}
	if provider.embedCalls != len(toolTexts)+1 {
		t.Fatalf("per-text Embed called %d times, want %d (each tool + query)", provider.embedCalls, len(toolTexts)+1)
	}
	if !reflect.DeepEqual(queryVec, vectorFor("the query")) {
		t.Fatalf("fallback query embedding mismatch")
	}
	for i, text := range toolTexts {
		if !reflect.DeepEqual(toolVecs[i], vectorFor(text)) {
			t.Fatalf("fallback misaligned tool %d: got %v, want the vector for %q", i, toolVecs[i], text)
		}
	}
}

type failingEmbeddingProvider struct{}

func (p *failingEmbeddingProvider) Embed(context.Context, string) ([]float32, error) {
	return nil, fmt.Errorf("embedding backend unavailable")
}

func (p *failingEmbeddingProvider) EmbedBatch(context.Context, []string) ([][]float32, error) {
	return nil, fmt.Errorf("embedding backend unavailable")
}

func (p *failingEmbeddingProvider) Dimension() int { return 4 }

func (p *failingEmbeddingProvider) Backend() string {
	return config.EmbeddingBackendOpenAICompatible
}

func TestToolEmbedderKeyDistinguishesModelIdentity(t *testing.T) {
	provider := &countingEmbeddingProvider{}
	remote := newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 4, "http://a\x00model-a")
	remoteOtherDim := newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 8, "http://a\x00model-a")
	remoteOtherModel := newCachedToolEmbedder(provider, config.EmbeddingModelTypeRemote, 4, "http://a\x00model-b")
	local := newCachedToolEmbedder(nil, "bert", 4, "")
	localOtherModel := newCachedToolEmbedder(nil, "qwen3", 4, "")

	keys := map[string]string{
		"remote":             remote.key("text"),
		"remote-other-dim":   remoteOtherDim.key("text"),
		"remote-other-model": remoteOtherModel.key("text"),
		"local":              local.key("text"),
		"local-other-model":  localOtherModel.key("text"),
	}
	seen := make(map[string]string, len(keys))
	for name, key := range keys {
		if other, dup := seen[key]; dup {
			t.Fatalf("%s and %s share the memo key %q; one model could serve the other's vectors", name, other, key)
		}
		seen[key] = name
	}
	if remote.key("a") == remote.key("b") {
		t.Fatalf("different texts collide on the same memo key")
	}
}

// requestToolsFor builds request tools whose embedding text matches toolTextsFor.
func requestToolsFor(n int) []openai.ChatCompletionToolParam {
	requestTools := make([]openai.ChatCompletionToolParam, n)
	for i := range requestTools {
		requestTools[i] = openai.ChatCompletionToolParam{
			Type: "function",
			Function: openai.FunctionDefinitionParam{
				Name:        fmt.Sprintf("tool_%d", i),
				Description: param.NewOpt(fmt.Sprintf("does job number %d", i)),
			},
		}
	}
	return requestTools
}

// perToolScores recomputes the dot products the filter ranks on, in request order.
func perToolScores(t *testing.T, emb *cachedToolEmbedder, query string, requestTools []openai.ChatCompletionToolParam) []float32 {
	t.Helper()
	toolTexts := make([]string, len(requestTools))
	for i, tool := range requestTools {
		toolTexts[i] = toolEmbeddingText(tool)
	}
	queryVec, toolVecs, err := emb.embedQueryAndTools(context.Background(), query, toolTexts)
	if err != nil {
		t.Fatalf("embedQueryAndTools failed: %v", err)
	}
	scores := make([]float32, len(toolVecs))
	for i, vec := range toolVecs {
		scores[i] = dotProductFloat32(queryVec, vec)
	}
	return scores
}

func TestFilterRankingIsIdenticalColdAndWarm(t *testing.T) {
	provider := &countingEmbeddingProvider{}
	emb := newTestToolEmbedder(provider)
	requestTools := requestToolsFor(25)
	const query = "which tool does job number 7"

	coldKept, coldScore, err := filterRequestToolsAgainstQuerySemantic(context.Background(), query, requestTools, emb, 0.4, 3)
	if err != nil {
		t.Fatalf("cold filter failed: %v", err)
	}
	warmKept, warmScore, err := filterRequestToolsAgainstQuerySemantic(context.Background(), query, requestTools, emb, 0.4, 3)
	if err != nil {
		t.Fatalf("warm filter failed: %v", err)
	}

	if coldScore != warmScore {
		t.Fatalf("max score changed across the cache boundary: cold=%v warm=%v", coldScore, warmScore)
	}
	if len(coldKept) != len(warmKept) {
		t.Fatalf("kept tool count changed: cold=%d warm=%d", len(coldKept), len(warmKept))
	}
	for i := range coldKept {
		if coldKept[i].Function.Name != warmKept[i].Function.Name {
			t.Fatalf("kept tool %d changed: cold=%q warm=%q", i, coldKept[i].Function.Name, warmKept[i].Function.Name)
		}
	}
	if len(coldKept) == 0 {
		t.Fatalf("filter kept no tools, so the equivalence check is vacuous")
	}

	// Kept names and the max score agree; check every per-tool score too, so the
	// equivalence covers the scores the filter ranks on and not just its output.
	coldScores := perToolScores(t, newTestToolEmbedder(&countingEmbeddingProvider{}), query, requestTools)
	warmScores := perToolScores(t, emb, query, requestTools)
	if !reflect.DeepEqual(coldScores, warmScores) {
		t.Fatalf("per-tool scores changed across the cache boundary:\ncold=%v\nwarm=%v", coldScores, warmScores)
	}

	// A fresh embedder with an empty memo must rank the same way, proving the memo
	// only removes work and never changes the outcome.
	freshKept, freshScore, err := filterRequestToolsAgainstQuerySemantic(context.Background(), query, requestTools, newTestToolEmbedder(&countingEmbeddingProvider{}), 0.4, 3)
	if err != nil {
		t.Fatalf("fresh filter failed: %v", err)
	}
	if freshScore != coldScore || len(freshKept) != len(coldKept) {
		t.Fatalf("fresh embedder ranked differently: score %v vs %v, kept %d vs %d",
			freshScore, coldScore, len(freshKept), len(coldKept))
	}
	for i := range coldKept {
		if freshKept[i].Function.Name != coldKept[i].Function.Name {
			t.Fatalf("fresh embedder kept tool %d = %q, want %q", i, freshKept[i].Function.Name, coldKept[i].Function.Name)
		}
	}
}

func TestFilterRequiresInitializedEmbedderOnlyWhenEmbeddingIsNeeded(t *testing.T) {
	requestTools := requestToolsFor(2)

	// An empty query short-circuits before any embedding, so a nil embedder is fine.
	kept, _, err := filterRequestToolsAgainstQuerySemantic(context.Background(), "   ", requestTools, nil, 0.5, 0)
	if err != nil {
		t.Fatalf("empty-query filter with nil embedder failed: %v", err)
	}
	if len(kept) != len(requestTools) {
		t.Fatalf("empty-query filter kept %d tools, want all %d", len(kept), len(requestTools))
	}

	if _, _, err := filterRequestToolsAgainstQuerySemantic(context.Background(), "real query", requestTools, nil, 0.5, 0); err == nil {
		t.Fatalf("filter with a nil embedder succeeded, want an error")
	}
}

func BenchmarkToolSelectionFilterEmbed(b *testing.B) {
	const query = "which tool does job number 7"

	for _, toolCount := range []int{5, 20, 50, 100} {
		requestTools := requestToolsFor(toolCount)

		// Cold: a fresh embedder per iteration, i.e. every tool is embedded again,
		// matching the pre-cache behaviour.
		b.Run(fmt.Sprintf("cold/tools=%d", toolCount), func(b *testing.B) {
			provider := &countingEmbeddingProvider{}
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				emb := newTestToolEmbedder(provider)
				if _, _, err := filterRequestToolsAgainstQuerySemantic(context.Background(), query, requestTools, emb, 0.4, 3); err != nil {
					b.Fatalf("filter failed: %v", err)
				}
			}
			b.StopTimer()
			batchCalls, _, embedded := provider.snapshot()
			b.ReportMetric(float64(len(embedded))/float64(b.N), "texts-embedded/op")
			b.ReportMetric(float64(batchCalls)/float64(b.N), "embed-calls/op")
		})

		// Warm: one embedder reused across iterations, so only the query is embedded.
		b.Run(fmt.Sprintf("warm/tools=%d", toolCount), func(b *testing.B) {
			provider := &countingEmbeddingProvider{}
			emb := newTestToolEmbedder(provider)
			if _, _, err := filterRequestToolsAgainstQuerySemantic(context.Background(), query, requestTools, emb, 0.4, 3); err != nil {
				b.Fatalf("warm-up filter failed: %v", err)
			}
			provider.reset()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, _, err := filterRequestToolsAgainstQuerySemantic(context.Background(), query, requestTools, emb, 0.4, 3); err != nil {
					b.Fatalf("filter failed: %v", err)
				}
			}
			b.StopTimer()
			batchCalls, _, embedded := provider.snapshot()
			b.ReportMetric(float64(len(embedded))/float64(b.N), "texts-embedded/op")
			b.ReportMetric(float64(batchCalls)/float64(b.N), "embed-calls/op")
		})
	}
}
