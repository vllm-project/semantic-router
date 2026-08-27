package extproc

import (
	"context"
	"fmt"
	"slices"
	"strconv"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// newToolEmbedderForConfig builds the tool embedder for cfg with the given
// provider (nil selects the local candle path). The remote endpoint's base URL
// and model name are folded into the memo key identity so cached vectors can
// never be served across differently configured endpoints.
func newToolEmbedderForConfig(cfg *config.RouterConfig, provider embedding.Provider) *cachedToolEmbedder {
	var modelType, remoteIdentity string
	var targetDim int
	if cfg != nil {
		modelType = cfg.EmbeddingModels.EmbeddingConfig.ModelType
		targetDim = cfg.EmbeddingModels.EmbeddingConfig.TargetDimension
		remoteIdentity = cfg.EmbeddingModels.Endpoint.BaseURL + "\x00" + cfg.EmbeddingModels.Endpoint.Model
	}
	return newCachedToolEmbedder(provider, modelType, targetDim, remoteIdentity)
}

// toolEmbedBatchChunkSize caps how many texts are sent to the remote provider in
// a single EmbedBatch call. The provider does not chunk large inputs itself, and
// embedding endpoints commonly reject or badly latency-spike oversized batches.
const toolEmbedBatchChunkSize = 64

// cachedToolEmbedder embeds tool-selection texts, reusing tool embeddings across
// requests through a bounded memo and filling misses in batches.
//
// A nil provider selects the local candle path (per-text FFI calls, unchanged);
// a non-nil provider selects the remote path, where misses are batched.
type cachedToolEmbedder struct {
	provider  embedding.Provider // nil => local candle path
	modelType string
	targetDim int
	// keyPrefix namespaces memoized texts by the identity of the model that
	// produced the vector, so a reconfigured model can never serve a vector
	// computed by a different one. Constant for the embedder's lifetime, so it
	// is computed once here instead of per key() call.
	keyPrefix string
	memo      *embedding.ToolEmbeddingMemo
}

// newCachedToolEmbedder constructs an embedder. remoteIdentity distinguishes
// remote endpoints/models that a bare Backend() string cannot (it is a constant
// per provider type); pass "" for the local candle path.
func newCachedToolEmbedder(provider embedding.Provider, modelType string, targetDim int, remoteIdentity string) *cachedToolEmbedder {
	dim := strconv.Itoa(targetDim)
	keyPrefix := "candle\x00" + modelType + "\x00" + dim + "\x00"
	if provider != nil {
		keyPrefix = "remote\x00" + provider.Backend() + "\x00" + remoteIdentity + "\x00" + dim + "\x00"
	}
	return &cachedToolEmbedder{
		provider:  provider,
		modelType: modelType,
		targetDim: targetDim,
		keyPrefix: keyPrefix,
		memo:      embedding.NewToolEmbeddingMemo(embedding.DefaultToolEmbeddingMemoSize),
	}
}

func (e *cachedToolEmbedder) key(text string) string {
	return e.keyPrefix + text
}

// embedQueryAndTools returns the query embedding and one embedding per toolTexts entry
// (aligned by index). Tool embeddings are served from the memo when possible; misses
// are deduplicated and filled in batches. The query embedding is intentionally NOT
// memoized: query texts are near-unique across requests and would evict the tool
// entries the memo exists to keep resident.
func (e *cachedToolEmbedder) embedQueryAndTools(
	ctx context.Context,
	queryText string,
	toolTexts []string,
) ([]float32, [][]float32, error) {
	toolEmbeddings := make([][]float32, len(toolTexts))

	// Pass 1: serve what the memo already holds, and collect the unique misses in
	// first-seen order so the fill list is deterministic.
	missTexts := make([]string, 0, len(toolTexts))
	missIndex := make(map[string]int, len(toolTexts)) // miss text -> position in missTexts
	hits := 0
	for i, text := range toolTexts {
		if v, ok := e.memo.Get(e.key(text)); ok {
			toolEmbeddings[i] = v
			hits++
			continue
		}
		if _, seen := missIndex[text]; !seen {
			missIndex[text] = len(missTexts)
			missTexts = append(missTexts, text)
		}
	}

	// The query is always a miss (it is never memoized), and goes last so the
	// tool misses keep their positions in the fill list.
	fillTexts := make([]string, 0, len(missTexts)+1)
	fillTexts = append(fillTexts, missTexts...)
	fillTexts = append(fillTexts, queryText)

	filled, err := e.fill(ctx, fillTexts)
	if err != nil {
		return nil, nil, err
	}

	// Metrics describe completed passes only: an aborted fill inflates neither the
	// hit/miss ratio nor the batch-size histogram. hits counts tool *indices*
	// served from the memo, while misses counts the *unique* texts embedded, so
	// duplicate tool texts within one request are not double-counted as misses.
	metrics.RecordToolEmbeddingCacheHits(hits)
	metrics.RecordToolEmbeddingCacheMisses(len(missTexts))
	metrics.RecordToolEmbeddingBatchSize(len(fillTexts))

	// Peel the query off the tail, then memoize only the tool misses. Nothing is
	// memoized when the fill failed, so a transient error is retried next request.
	queryEmbedding := filled[len(filled)-1]
	for i, text := range missTexts {
		e.memo.Put(e.key(text), filled[i])
	}

	// Pass 2: fan the freshly computed vectors back out to every index that asked
	// for them, including duplicate tool texts within this one request.
	for i, text := range toolTexts {
		if toolEmbeddings[i] != nil {
			continue
		}
		toolEmbeddings[i] = filled[missIndex[text]]
	}

	return queryEmbedding, toolEmbeddings, nil
}

// fill computes one embedding per text, in order, batching through the remote
// provider or looping the local candle binding. Both paths return exactly
// len(texts) vectors or an error.
func (e *cachedToolEmbedder) fill(ctx context.Context, texts []string) ([][]float32, error) {
	if e.provider == nil {
		return e.fillLocal(texts)
	}
	return e.fillRemote(ctx, texts)
}

func (e *cachedToolEmbedder) fillLocal(texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for i, text := range texts {
		output, err := candle_binding.GetEmbeddingWithModelType(text, e.modelType, e.targetDim)
		if err != nil {
			return nil, fmt.Errorf("tool_selection filter: embedding text %d/%d: %w", i+1, len(texts), err)
		}
		out = append(out, output.Embedding)
	}
	return out, nil
}

func (e *cachedToolEmbedder) fillRemote(ctx context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, 0, len(texts))
	for chunk := range slices.Chunk(texts, toolEmbedBatchChunkSize) {
		embeddings, err := e.provider.EmbedBatch(ctx, chunk)
		if err != nil && len(chunk) > 1 {
			// An endpoint can reject a combined batch (e.g. a per-request token
			// cap) even though every text embeds fine on its own — the exact
			// requests the pre-batching code used to make. Degrade to per-text
			// embedding for this chunk so batching never fails a workload that
			// previously succeeded; a hard outage still aborts on the first text.
			logging.Warnf("tool_selection filter: batch of %d texts failed (%v), retrying per text", len(chunk), err)
			embeddings, err = e.embedChunkPerText(ctx, chunk)
		}
		if err != nil {
			return nil, fmt.Errorf("tool_selection filter: embedding batch of %d texts: %w", len(chunk), err)
		}
		// EmbedBatch already validates the returned count against the request;
		// re-check here so an alternative Provider implementation cannot silently
		// misalign tool texts with vectors.
		if len(embeddings) != len(chunk) {
			return nil, fmt.Errorf(
				"tool_selection filter: embedding batch returned %d vectors for %d texts",
				len(embeddings), len(chunk),
			)
		}
		out = append(out, embeddings...)
	}
	return out, nil
}

// embedChunkPerText embeds each text of a failed chunk individually, restoring
// the pre-batching request shape. The error names the failing text's position
// so a single bad tool definition is attributable from logs.
func (e *cachedToolEmbedder) embedChunkPerText(ctx context.Context, chunk []string) ([][]float32, error) {
	out := make([][]float32, 0, len(chunk))
	for i, text := range chunk {
		vec, err := e.provider.Embed(ctx, text)
		if err != nil {
			return nil, fmt.Errorf("embedding text %d/%d: %w", i+1, len(chunk), err)
		}
		out = append(out, vec)
	}
	return out, nil
}
