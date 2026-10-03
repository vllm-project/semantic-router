package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// retrieveFromHybrid retrieves context using hybrid backend (multiple backends)
func (r *OpenAIRouter) retrieveFromHybrid(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	hybridConfig, err := ragConfig.HybridBackendConfig()
	if err != nil {
		return "", fmt.Errorf("invalid hybrid RAG config: %w", err)
	}

	if hybridConfig.Primary == "" {
		return "", fmt.Errorf("primary backend is required for hybrid RAG")
	}

	strategy := hybridConfig.Strategy
	if strategy == "" {
		strategy = "sequential" // Default
	}

	switch strategy {
	case "sequential":
		return r.retrieveSequential(traceCtx, ctx, ragConfig, hybridConfig)
	case "parallel":
		return r.retrieveParallel(traceCtx, ctx, ragConfig, hybridConfig)
	default:
		return "", fmt.Errorf("unknown hybrid strategy: %s", strategy)
	}
}

// retrieveSequential tries primary backend first, then fallback
func (r *OpenAIRouter) retrieveSequential(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig, hybridConfig *config.HybridRAGConfig) (string, error) {
	// Try primary backend
	primaryConfig := &config.RAGPluginConfig{
		Enabled:             ragConfig.Enabled,
		Backend:             hybridConfig.Primary,
		SimilarityThreshold: ragConfig.SimilarityThreshold,
		TopK:                ragConfig.TopK,
		MaxContextLength:    ragConfig.MaxContextLength,
		InjectionMode:       ragConfig.InjectionMode,
		BackendConfig:       hybridConfig.PrimaryConfig,
		OnFailure:           "skip", // Don't block on primary failure
		CacheResults:        ragConfig.CacheResults,
		CacheTTLSeconds:     ragConfig.CacheTTLSeconds,
	}

	context, err := r.retrieveFromBackend(traceCtx, ctx, primaryConfig)
	if err == nil && context != "" {
		logging.Infof("Hybrid RAG: primary backend (%s) succeeded", hybridConfig.Primary)
		return context, nil
	}

	logging.Warnf("Hybrid RAG: primary backend (%s) failed: %v", hybridConfig.Primary, err)

	// Try fallback backend
	if hybridConfig.Fallback == "" {
		return "", fmt.Errorf("primary backend failed and no fallback configured: %w", err)
	}

	fallbackConfig := &config.RAGPluginConfig{
		Enabled:             ragConfig.Enabled,
		Backend:             hybridConfig.Fallback,
		SimilarityThreshold: ragConfig.SimilarityThreshold,
		TopK:                ragConfig.TopK,
		MaxContextLength:    ragConfig.MaxContextLength,
		InjectionMode:       ragConfig.InjectionMode,
		BackendConfig:       hybridConfig.FallbackConfig,
		OnFailure:           ragConfig.OnFailure,
		CacheResults:        ragConfig.CacheResults,
		CacheTTLSeconds:     ragConfig.CacheTTLSeconds,
	}

	fallbackContext, fallbackErr := r.retrieveFromBackend(traceCtx, ctx, fallbackConfig)
	if fallbackErr != nil {
		return "", fmt.Errorf("both primary and fallback backends failed: primary=%w, fallback=%w", err, fallbackErr)
	}

	logging.Infof("Hybrid RAG: fallback backend (%s) succeeded", hybridConfig.Fallback)
	return fallbackContext, nil
}

// parallelRAGResult is one backend's retrieval outcome.
// Score and latency are copied off a private RequestContext so the lookup
// goroutine never writes the caller's request.
type parallelRAGResult struct {
	context string
	err     error
	score   float32
	latency float64
}

// measureParallelRAGLookup records the full duration of one child lookup. It
// deliberately does not use a backend's RAGRetrievalLatency: some backends do
// not set it, while others record only their HTTP portion. Parallel hybrid
// metrics must use the same end-to-end scope for every backend.
func measureParallelRAGLookup(ctx *RequestContext, retrieve func() (string, error)) parallelRAGResult {
	start := time.Now()
	retrieved, err := retrieve()
	return parallelRAGResult{
		context: retrieved,
		err:     err,
		score:   ctx.RAGSimilarityScore,
		latency: time.Since(start).Seconds(),
	}
}

// retrieveParallel starts both backends and returns as soon as the primary
// produces context. It does not rank backends: they do not share a score.
// A fallback is used only when the primary fails or returns empty context.
func (r *OpenAIRouter) retrieveParallel(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig, hybridConfig *config.HybridRAGConfig) (string, error) {
	// Buffered so a late backend can send after the caller has returned.
	primaryChan := make(chan parallelRAGResult, 1)
	fallbackChan := make(chan parallelRAGResult, 1)
	childCtx, cancel := context.WithCancel(traceCtx)
	defer cancel()

	// Each lookup gets its own RequestContext. Backends write similarity and
	// latency onto that copy; the caller applies the winner after selection.
	primaryCtx := *ctx
	fallbackCtx := *ctx

	// Try primary backend
	go func() {
		defer func() {
			if r := recover(); r != nil {
				primaryChan <- parallelRAGResult{err: fmt.Errorf("panic in primary backend: %v", r)}
			}
		}()

		// Check context cancellation
		select {
		case <-childCtx.Done():
			primaryChan <- parallelRAGResult{err: childCtx.Err()}
			return
		default:
		}

		primaryConfig := &config.RAGPluginConfig{
			Enabled:             ragConfig.Enabled,
			Backend:             hybridConfig.Primary,
			SimilarityThreshold: ragConfig.SimilarityThreshold,
			TopK:                ragConfig.TopK,
			MaxContextLength:    ragConfig.MaxContextLength,
			InjectionMode:       ragConfig.InjectionMode,
			BackendConfig:       hybridConfig.PrimaryConfig,
			OnFailure:           "skip",
			CacheResults:        ragConfig.CacheResults,
			CacheTTLSeconds:     ragConfig.CacheTTLSeconds,
		}
		primaryChan <- measureParallelRAGLookup(&primaryCtx, func() (string, error) {
			return r.retrieveFromBackend(childCtx, &primaryCtx, primaryConfig)
		})
	}()

	// Try fallback backend
	if hybridConfig.Fallback != "" {
		go func() {
			defer func() {
				if r := recover(); r != nil {
					fallbackChan <- parallelRAGResult{err: fmt.Errorf("panic in fallback backend: %v", r)}
				}
			}()

			// Check context cancellation
			select {
			case <-childCtx.Done():
				fallbackChan <- parallelRAGResult{err: childCtx.Err()}
				return
			default:
			}

			fallbackConfig := &config.RAGPluginConfig{
				Enabled:             ragConfig.Enabled,
				Backend:             hybridConfig.Fallback,
				SimilarityThreshold: ragConfig.SimilarityThreshold,
				TopK:                ragConfig.TopK,
				MaxContextLength:    ragConfig.MaxContextLength,
				InjectionMode:       ragConfig.InjectionMode,
				BackendConfig:       hybridConfig.FallbackConfig,
				OnFailure:           "skip",
				CacheResults:        ragConfig.CacheResults,
				CacheTTLSeconds:     ragConfig.CacheTTLSeconds,
			}
			fallbackChan <- measureParallelRAGLookup(&fallbackCtx, func() (string, error) {
				return r.retrieveFromBackend(childCtx, &fallbackCtx, fallbackConfig)
			})
		}()
	}

	// Primary with context returns immediately. A slow fallback is cancelled.
	// Backends do not publish a comparable score, so this does not rank them.
	chosen, err := selectParallelRAG(primaryChan, hybridConfig.Fallback != "", fallbackChan)
	if err != nil {
		return "", err
	}
	ctx.RAGSimilarityScore = chosen.score
	ctx.RAGRetrievalLatency = chosen.latency
	return chosen.context, nil
}

// selectParallelRAG prefers a non-empty primary result and does not wait for
// the fallback once that result is available. The fallback is used when the
// primary fails or is empty.
func selectParallelRAG(primary <-chan parallelRAGResult, haveFallback bool, fallback <-chan parallelRAGResult) (parallelRAGResult, error) {
	if !haveFallback {
		res := <-primary
		if res.err == nil && res.context != "" {
			return res, nil
		}
		if res.err != nil {
			return parallelRAGResult{}, res.err
		}
		return parallelRAGResult{}, fmt.Errorf("primary backend returned empty context")
	}

	var fallbackRes parallelRAGResult
	fallbackReady := false
	for {
		select {
		case res := <-primary:
			if res.err == nil && res.context != "" {
				return res, nil
			}
			if !fallbackReady {
				fallbackRes = <-fallback
			}
			if fallbackRes.err == nil && fallbackRes.context != "" {
				return fallbackRes, nil
			}
			primaryErr := res.err
			if primaryErr == nil {
				primaryErr = fmt.Errorf("primary backend returned empty context")
			}
			fallbackErr := fallbackRes.err
			if fallbackErr == nil {
				fallbackErr = fmt.Errorf("fallback backend returned empty context")
			}
			return parallelRAGResult{}, fmt.Errorf("both backends failed: primary=%w, fallback=%w", primaryErr, fallbackErr)
		case res := <-fallback:
			if !fallbackReady {
				fallbackRes = res
				fallbackReady = true
			}
		}
	}
}

// retrieveFromBackend is a helper to retrieve from a specific backend
func (r *OpenAIRouter) retrieveFromBackend(traceCtx context.Context, ctx *RequestContext, backendConfig *config.RAGPluginConfig) (string, error) {
	switch backendConfig.Backend {
	case "milvus":
		return r.retrieveFromMilvus(traceCtx, ctx, backendConfig)
	case "qdrant":
		return r.retrieveFromQdrant(traceCtx, ctx, backendConfig)
	case "external_api":
		return r.retrieveFromExternalAPI(traceCtx, ctx, backendConfig)
	case "mcp":
		return r.retrieveFromMCP(traceCtx, ctx, backendConfig)
	case "openai":
		return r.retrieveFromOpenAI(traceCtx, ctx, backendConfig)
	default:
		return "", fmt.Errorf("unknown backend: %s", backendConfig.Backend)
	}
}
