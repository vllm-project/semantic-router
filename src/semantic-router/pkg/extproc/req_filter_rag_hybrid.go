package extproc

import (
	"context"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// recallFromHybrid recalls documents from a hybrid source's primary
// backend and, when it fails or recalls none, from its fallback.
func (r *OpenAIRouter) recallFromHybrid(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource, query string, recall config.MatchRecall) ([]match.Scored, error) {
	hybridConfig, err := source.HybridBackendConfig()
	if err != nil {
		return nil, fmt.Errorf("invalid hybrid RAG config: %w", err)
	}
	if hybridConfig.Primary == "" {
		return nil, fmt.Errorf("primary backend is required for hybrid RAG")
	}
	switch strategy := hybridConfig.Strategy; strategy {
	case "", "sequential":
		return r.recallSequential(traceCtx, ctx, source, hybridConfig, query, recall)
	case "parallel":
		return r.recallParallel(traceCtx, ctx, source, hybridConfig, query, recall)
	default:
		return nil, fmt.Errorf("unknown hybrid strategy: %s", strategy)
	}
}

// hybridChild is the source a hybrid backend's primary or fallback recalls
// from.
func hybridChild(source *config.ContextSource, backend string, backendConfig *config.StructuredPayload) *config.ContextSource {
	return &config.ContextSource{Type: config.ContextSourceRAG, Backend: backend, BackendConfig: backendConfig, Match: source.Match}
}

// recallSequential tries the primary backend first, then the fallback.
func (r *OpenAIRouter) recallSequential(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource, hybridConfig *config.HybridRAGConfig, query string, recall config.MatchRecall) ([]match.Scored, error) {
	documents, err := r.recallDocuments(traceCtx, ctx, hybridChild(source, hybridConfig.Primary, hybridConfig.PrimaryConfig), query, recall)
	if err == nil && len(documents) > 0 {
		logging.Infof("Hybrid RAG: primary backend (%s) succeeded", hybridConfig.Primary)
		return documents, nil
	}
	logging.Warnf("Hybrid RAG: primary backend (%s) recalled nothing: %v", hybridConfig.Primary, err)
	if hybridConfig.Fallback == "" {
		if err == nil {
			return documents, nil
		}
		return nil, fmt.Errorf("primary backend failed and no fallback configured: %w", err)
	}
	fallback, fallbackErr := r.recallDocuments(traceCtx, ctx, hybridChild(source, hybridConfig.Fallback, hybridConfig.FallbackConfig), query, recall)
	if fallbackErr != nil {
		if err == nil {
			return nil, fallbackErr
		}
		return nil, fmt.Errorf("both primary and fallback backends failed: primary=%w, fallback=%w", err, fallbackErr)
	}
	logging.Infof("Hybrid RAG: fallback backend (%s) succeeded", hybridConfig.Fallback)
	return fallback, nil
}

// parallelRAGResult is one backend's recall outcome. Latency is copied off a
// private RequestContext so the lookup goroutine never writes the caller's
// request.
type parallelRAGResult struct {
	documents []match.Scored
	err       error
	latency   float64
}

// measureParallelRAGLookup records the full duration of one child lookup. It
// deliberately does not use a backend's RAGRetrievalLatency: some backends do
// not set it, while others record only their HTTP portion. Parallel hybrid
// metrics must use the same end-to-end scope for every backend.
func measureParallelRAGLookup(recall func() ([]match.Scored, error)) parallelRAGResult {
	start := time.Now()
	documents, err := recall()
	return parallelRAGResult{documents: documents, err: err, latency: time.Since(start).Seconds()}
}

// recallParallel starts both backends and returns as soon as the primary
// recalls documents. It does not rank backends: they do not share a score.
// A fallback is used only when the primary fails or recalls none.
func (r *OpenAIRouter) recallParallel(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource, hybridConfig *config.HybridRAGConfig, query string, recall config.MatchRecall) ([]match.Scored, error) {
	// Buffered so a late backend can send after the caller has returned.
	primaryChan := make(chan parallelRAGResult, 1)
	fallbackChan := make(chan parallelRAGResult, 1)
	childCtx, cancel := context.WithCancel(traceCtx)
	defer cancel()
	lookup := func(results chan<- parallelRAGResult, child *config.ContextSource, label string) {
		// Each lookup gets its own RequestContext: backends write latency onto that copy.
		private := *ctx
		defer func() {
			if recovered := recover(); recovered != nil {
				results <- parallelRAGResult{err: fmt.Errorf("panic in %s backend: %v", label, recovered)}
			}
		}()
		if childCtx.Err() != nil {
			results <- parallelRAGResult{err: childCtx.Err()}
			return
		}
		results <- measureParallelRAGLookup(func() ([]match.Scored, error) {
			return r.recallDocuments(childCtx, &private, child, query, recall)
		})
	}
	go lookup(primaryChan, hybridChild(source, hybridConfig.Primary, hybridConfig.PrimaryConfig), "primary")
	if hybridConfig.Fallback != "" {
		go lookup(fallbackChan, hybridChild(source, hybridConfig.Fallback, hybridConfig.FallbackConfig), "fallback")
	}
	chosen, err := selectParallelRAG(primaryChan, hybridConfig.Fallback != "", fallbackChan)
	if err != nil {
		return nil, err
	}
	ctx.RAGRetrievalLatency = chosen.latency
	return chosen.documents, nil
}

// selectParallelRAG prefers a primary that recalled documents and does not
// wait for the fallback once it has. The fallback is used when the primary
// fails or recalls none.
func selectParallelRAG(primary <-chan parallelRAGResult, haveFallback bool, fallback <-chan parallelRAGResult) (parallelRAGResult, error) {
	if !haveFallback {
		res := <-primary
		return res, res.err
	}
	var fallbackRes parallelRAGResult
	fallbackReady := false
	for {
		select {
		case res := <-primary:
			if res.err == nil && len(res.documents) > 0 {
				return res, nil
			}
			if !fallbackReady {
				fallbackRes = <-fallback
			}
			if fallbackRes.err == nil {
				return fallbackRes, nil
			}
			if res.err == nil {
				return res, nil
			}
			return parallelRAGResult{}, fmt.Errorf("both backends failed: primary=%w, fallback=%w", res.err, fallbackRes.err)
		case res := <-fallback:
			if !fallbackReady {
				fallbackRes = res
				fallbackReady = true
			}
		}
	}
}
