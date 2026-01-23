package extproc

import (
	"context"
	"fmt"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// retrieveFromHybrid retrieves context using hybrid backend (multiple backends)
func (r *OpenAIRouter) retrieveFromHybrid(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig) (string, error) {
	hybridConfig, ok := ragConfig.BackendConfig.(*config.HybridRAGConfig)
	if !ok {
		return "", fmt.Errorf("invalid hybrid RAG config")
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

// retrieveParallel tries both backends in parallel and uses the best result
// Uses channels for proper synchronization to avoid race conditions when
// multiple goroutines write results. The WaitGroup ensures all goroutines
// complete before we read from channels, and buffered channels (size 1) allow
// goroutines to send without blocking.
func (r *OpenAIRouter) retrieveParallel(traceCtx context.Context, ctx *RequestContext, ragConfig *config.RAGPluginConfig, hybridConfig *config.HybridRAGConfig) (string, error) {
	type result struct {
		context string
		err     error
	}

	var wg sync.WaitGroup
	// Buffered channels (size 1) allow goroutines to send results without blocking
	// and provide proper synchronization according to Go's memory model
	primaryChan := make(chan result, 1)
	fallbackChan := make(chan result, 1)

	// Try primary backend
	wg.Add(1)
	go func() {
		defer wg.Done()
		defer func() {
			if r := recover(); r != nil {
				primaryChan <- result{"", fmt.Errorf("panic in primary backend: %v", r)}
			}
		}()

		// Check context cancellation
		select {
		case <-traceCtx.Done():
			primaryChan <- result{"", traceCtx.Err()}
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
		context, err := r.retrieveFromBackend(traceCtx, ctx, primaryConfig)
		primaryChan <- result{context, err}
	}()

	// Try fallback backend
	if hybridConfig.Fallback != "" {
		wg.Add(1)
		go func() {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					fallbackChan <- result{"", fmt.Errorf("panic in fallback backend: %v", r)}
				}
			}()

			// Check context cancellation
			select {
			case <-traceCtx.Done():
				fallbackChan <- result{"", traceCtx.Err()}
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
			context, err := r.retrieveFromBackend(traceCtx, ctx, fallbackConfig)
			fallbackChan <- result{context, err}
		}()
	}

	// Wait for goroutines and collect results
	wg.Wait()
	close(primaryChan)
	if hybridConfig.Fallback != "" {
		close(fallbackChan)
	}

	primaryRes := <-primaryChan
	primaryContext, primaryErr := primaryRes.context, primaryRes.err
	var fallbackContext string
	var fallbackErr error
	if hybridConfig.Fallback != "" {
		fallbackRes := <-fallbackChan
		fallbackContext, fallbackErr = fallbackRes.context, fallbackRes.err
	}

	// Use the result with highest similarity score or first successful result
	if primaryErr == nil && primaryContext != "" {
		if fallbackErr == nil && fallbackContext != "" {
			// Both succeeded, use the one with better similarity (if available)
			// For now, prefer primary
			logging.Infof("Hybrid RAG: both backends succeeded, using primary (%s)", hybridConfig.Primary)
			return primaryContext, nil
		}
		logging.Infof("Hybrid RAG: primary backend (%s) succeeded", hybridConfig.Primary)
		return primaryContext, nil
	}

	if fallbackErr == nil && fallbackContext != "" {
		logging.Infof("Hybrid RAG: fallback backend (%s) succeeded", hybridConfig.Fallback)
		return fallbackContext, nil
	}

	return "", fmt.Errorf("both backends failed: primary=%w, fallback=%w", primaryErr, fallbackErr)
}

// retrieveFromBackend is a helper to retrieve from a specific backend
func (r *OpenAIRouter) retrieveFromBackend(traceCtx context.Context, ctx *RequestContext, backendConfig *config.RAGPluginConfig) (string, error) {
	switch backendConfig.Backend {
	case "milvus":
		return r.retrieveFromMilvus(traceCtx, ctx, backendConfig)
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
