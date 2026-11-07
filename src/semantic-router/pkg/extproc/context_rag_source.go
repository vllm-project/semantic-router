package extproc

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/match"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
)

// defaultRAGContextLength is how many characters of documents a rag source
// injects without max_context_length.
const defaultRAGContextLength = 10000

// ragIndex recalls a rag source's documents from its backend, which applies
// recall's top_k and min_similarity where it can and reports each document's
// score as its similarity.
type ragIndex struct {
	router  *OpenAIRouter
	request *RequestContext
	source  *config.ContextSource
}

func (x ragIndex) Recall(ctx context.Context, query string, recall config.MatchRecall) ([]match.Scored, error) {
	return x.router.recallDocuments(ctx, x.request, x.source, query, recall)
}

// recallDocuments asks a rag source's backend for the documents most
// similar to the query.
func (r *OpenAIRouter) recallDocuments(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource, query string, recall config.MatchRecall) ([]match.Scored, error) {
	var documents []match.Scored
	var err error
	switch source.Backend {
	case "vectorstore":
		documents, err = r.recallFromVectorStore(traceCtx, source, query, recall)
	case "milvus":
		documents, err = r.recallFromMilvus(traceCtx, ctx, source, query, recall)
	case "qdrant":
		documents, err = r.recallFromQdrant(traceCtx, ctx, source, query, recall)
	case "external_api":
		documents, err = r.recallFromExternalAPI(traceCtx, ctx, source, query, recall)
	case "openai":
		documents, err = r.recallFromOpenAI(traceCtx, source, query, recall)
	case "hybrid":
		documents, err = r.recallFromHybrid(traceCtx, ctx, source, query, recall)
	default:
		return nil, fmt.Errorf("unknown RAG backend: %s", source.Backend)
	}
	if err != nil {
		return nil, fmt.Errorf("backend %q retrieval failed: %w", source.Backend, err)
	}
	return documents, nil
}

// scoredDocuments are a backend's documents, in its order, under IDs that
// tell them apart.
func scoredDocuments(texts []string, similarities []float64) []match.Scored {
	documents := make([]match.Scored, 0, len(texts))
	for index, text := range texts {
		if strings.TrimSpace(text) == "" {
			continue
		}
		document := match.Scored{Candidate: match.Candidate{ID: strconv.Itoa(index), Text: text}}
		if index < len(similarities) {
			document.Similarity = similarities[index]
		}
		documents = append(documents, document)
	}
	return documents
}

// recallThreshold is the least similarity a backend returns: recall's bound,
// else any cosine.
func recallThreshold(recall config.MatchRecall) float32 {
	if recall.MinSimilarity == nil {
		return -1
	}
	return float32(*recall.MinSimilarity)
}

// retrieveRAGSource runs a rag source: its match recalls the documents from
// the backend and judges them, and the kept ones are injected. A failed
// retrieval is what the source's on_failure says.
func (r *OpenAIRouter) retrieveRAGSource(ctx *RequestContext, plugin *config.ContextPluginConfig, source *config.ContextSource, request *llmprotocol.Request) error {
	decisionName := ctx.VSRSelectedDecisionName
	ragCtx, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanRAGRetrieval)
	defer span.End()
	tracing.SetSpanAttributes(span, attribute.String("rag.backend", source.Backend), attribute.String("rag.decision", decisionName))

	start := time.Now()
	retrieved, err := r.ragSourceContext(ragCtx, ctx, source)
	latency := hybridRetrievalLatency(hybridStrategy(source), time.Since(start).Seconds(), ctx.RAGRetrievalLatency)
	ctx.RAGRetrievalLatency = latency
	tracing.SetSpanAttributes(span, attribute.Float64("rag.latency_seconds", latency), attribute.Bool("rag.success", err == nil))
	if err != nil {
		return handleRAGRetrievalError(ctx, span, source, err, latency)
	}
	retrieved = truncateRAGContext(retrieved, source)
	tracing.SetSpanAttributes(span, attribute.Int("rag.context_length", len(retrieved)))
	if ctx.RAGSimilarityScore > 0 {
		tracing.SetSpanAttributes(span, attribute.Float64("rag.similarity_score", float64(ctx.RAGSimilarityScore)))
	}
	span.SetStatus(codes.Ok, "Retrieval successful")
	metricDecision := requestDecisionStateKey(ctx)
	metrics.RecordRAGRetrieval(source.Backend, metricDecision, "success", latency)
	metrics.RecordRAGContextLength(source.Backend, metricDecision, len(retrieved))
	if ctx.RAGSimilarityScore > 0 {
		metrics.RecordRAGSimilarityScore(source.Backend, metricDecision, ctx.RAGSimilarityScore)
	}
	if retrieved == "" {
		return nil
	}
	if err := r.injectContext(ctx, plugin, config.ContextSourceRAG, retrieved); err != nil {
		logging.Errorf("[RAG] Failed to inject context for decision '%s' (backend=%s): %v", decisionName, source.Backend, err)
		metrics.RecordPluginExecution("rag", metricDecision, "injection_error", 0)
		return nil
	}
	ctx.RAGRetrievedContext = retrieved
	logging.Debugf("RAG source executed: backend=%s, context_len=%d, latency=%.3fs", source.Backend, len(retrieved), latency)
	return nil
}

// ragSourceContext is the documents a rag source keeps for the request,
// joined: cached, or recalled from its backend and kept by its match block.
func (r *OpenAIRouter) ragSourceContext(traceCtx context.Context, ctx *RequestContext, source *config.ContextSource) (string, error) {
	query := ctx.UserContent
	if strings.TrimSpace(query) == "" {
		return "", fmt.Errorf("no user content for RAG retrieval")
	}
	recipe := ctx.Routing.RecipeName()
	if cached, found := r.getCachedRAGContext(recipe, query, source); found {
		return cached, nil
	}
	ctx.RAGBackend = source.Backend
	result, err := r.matcher().Match(traceCtx, config.MatchRAG, source.Match, match.Input{Request: query, Index: ragIndex{router: r, request: ctx, source: source}})
	if err != nil {
		return "", err
	}
	texts := make([]string, len(result.Kept))
	for index, document := range result.Kept {
		texts[index] = document.Text
		ctx.RAGSimilarityScore = max(ctx.RAGSimilarityScore, float32(document.Similarity))
	}
	retrieved := strings.Join(texts, "\n\n---\n\n")
	if retrieved == "" {
		logging.Debugf("[RAG] Backend '%s' kept no documents for query: %s", source.Backend, logging.ContentDescriptor(query))
	}
	r.cacheRetrievedRAGContext(recipe, query, retrieved, source)
	return retrieved, nil
}

// truncateRAGContext keeps the source's max_context_length characters of
// the documents, at a UTF-8 boundary.
func truncateRAGContext(retrieved string, source *config.ContextSource) string {
	maxLength := defaultRAGContextLength
	if source.MaxContextLength != nil {
		maxLength = *source.MaxContextLength
	}
	if runes := []rune(retrieved); len(runes) > maxLength {
		logging.Debugf("RAG context truncated to %d chars", maxLength)
		return string(runes[:maxLength]) + "..."
	}
	return retrieved
}

// hybridRetrievalLatency keeps the winning backend duration for parallel hybrid
// only. Sequential hybrid must keep the wrapper elapsed time, including a
// failed primary that ran before the fallback.
func hybridRetrievalLatency(strategy string, wrapper, recorded float64) float64 {
	if strategy == "parallel" && recorded > 0 {
		return recorded
	}
	return wrapper
}

func hybridStrategy(source *config.ContextSource) string {
	if source == nil || source.Backend != "hybrid" {
		return ""
	}
	hybridConfig, err := source.HybridBackendConfig()
	if err != nil || hybridConfig == nil {
		return ""
	}
	return hybridConfig.Strategy
}

func handleRAGRetrievalError(ctx *RequestContext, span trace.Span, source *config.ContextSource, err error, latency float64) error {
	tracing.RecordError(span, "retrieval_failed")
	metrics.RecordRAGRetrieval(source.Backend, requestDecisionStateKey(ctx), "error", latency)
	switch source.EffectiveOnFailure() {
	case config.RAGOnFailureBlock:
		logging.Errorf("RAG retrieval failed and on_failure=block: %v", err)
		return fmt.Errorf("RAG retrieval failed: %w", err)
	case config.RAGOnFailureWarn:
		logging.Warnf("RAG retrieval failed (on_failure=warn): %v", err)
	default:
		logging.Debugf("RAG retrieval failed (on_failure=skip): %v", err)
	}
	ctx.RAGRetrievedContext = ""
	return nil
}

func (r *OpenAIRouter) getCachedRAGContext(recipe config.RecipeName, query string, source *config.ContextSource) (string, bool) {
	if !source.CacheResults {
		return "", false
	}
	if cached, found := r.getRAGCache(recipe, query, source); found {
		metrics.RecordRAGCacheHit(source.Backend)
		logging.Debugf("RAG cache hit for query: %s", logging.ContentDescriptor(query))
		return cached, true
	}
	metrics.RecordRAGCacheMiss(source.Backend)
	return "", false
}

func (r *OpenAIRouter) cacheRetrievedRAGContext(recipe config.RecipeName, query, retrieved string, source *config.ContextSource) {
	if source.CacheResults && retrieved != "" {
		r.setRAGCache(recipe, query, retrieved, source)
	}
}
