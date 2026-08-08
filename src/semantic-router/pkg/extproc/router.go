package extproc

import (
	"encoding/json"
	"errors"
	"sync"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
	httputil "github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests.
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	// RecipeClassifiers selects the isolated classifier graph for each routing
	// request. Classifier is the default-recipe accessor.
	RecipeClassifiers     *classification.RecipeClassifiers
	ClassificationService *services.ClassificationService
	Cache                 cache.CacheBackend
	ResponseCache         *cache.ResponseCacheService
	responseCacheMu       sync.Mutex
	ContextCompression    *contextcompression.Service
	CompressionRecovery   contextcompression.RecoveryStore
	CompressionEmbedding  embedding.Provider
	CompressionScorer     contextcompression.RelevanceScorer
	contextCompressionMu  sync.Mutex
	ToolsDatabase         *tools.ToolsDatabase
	ToolsRegistry         *tools.Registry // retriever strategy registry
	toolSelectionDBMu     sync.Mutex
	toolSelectionDBByPath map[string]*tools.ToolsDatabase
	ResponseAPIFilter     *ResponseAPIFilter
	ReplayRecorder        *routerreplay.Recorder
	ReplayStoreShared     bool
	// ModelSelector is the registry of advanced model selection algorithms
	// initialized from config.IntelligentRouting.ModelSelection.
	ModelSelector *selection.Registry
	// RecipeModelSelectors keeps mutable algorithm state (Elo, RL, ML adapters,
	// and similar selectors) isolated even when recipes reuse decision names.
	RecipeModelSelectors map[config.RecipeName]*selection.Registry
	LookupTable          lookuptable.LookupTable
	ReplayRecorders      map[string]*routerreplay.Recorder
	MemoryStore          memory.Store
	MemoryExtractor      *memory.MemoryExtractor

	// CredentialResolver resolves per-user LLM API keys from multiple sources
	// (ext_authz injected headers -> static config fallback).
	CredentialResolver *authz.CredentialResolver

	// RateLimiter enforces per-user/model rate limits from multiple sources
	// (Envoy RLS -> local limiter).
	RateLimiter *ratelimit.RateLimitResolver

	// RuntimeRegistry exposes runtime-owned services without forcing request-time
	// paths back through package-global API-server state.
	RuntimeRegistry *routerruntime.Registry

	routerLearningMu      sync.Mutex
	routerLearningRuntime *routerLearningRuntime
	lookupTableCancel     func()

	// generation owns every closeable resource buildRouterComponents produced,
	// registered in construction order so Close tears them down in reverse. Nil
	// for routers assembled by hand, which Close falls back to closing field by
	// field.
	generation *routerruntime.Generation
	closeOnce  sync.Once
	closeErr   error
}

// Close releases every resource this router owns. Only the first call runs the
// teardown and every caller observes the same error, because a router is
// reachable from more than one shutdown path — a reload retiring the lease it
// replaced, and process shutdown retiring the current one — and not all of the
// underlying resources tolerate a second close.
func (r *OpenAIRouter) Close() error {
	if r == nil {
		return nil
	}
	r.closeOnce.Do(func() {
		r.closeErr = r.closeResources()
	})
	return r.closeErr
}

func (r *OpenAIRouter) closeResources() error {
	if r.generation != nil {
		// Covers lookupTableCancel too, so there is nothing else to drive here.
		return r.generation.Close()
	}
	return r.closeOwnedFields()
}

// closeOwnedFields is the fallback for routers assembled by hand rather than by
// buildRouterComponents, which registers the same resources on a Generation.
// Keep the two in step: a resource added to one and not the other leaks on
// exactly one construction path.
//
// Pointer-typed fields rely on the nil-receiver guard inside their own Close.
// Interface-typed fields are nil-checked here, because taking a method value off
// a nil interface panics.
func (r *OpenAIRouter) closeOwnedFields() error {
	if r.lookupTableCancel != nil {
		r.lookupTableCancel()
	}

	var errs []error
	collect := func(err error) {
		if err != nil {
			errs = append(errs, err)
		}
	}

	if r.Cache != nil {
		collect(r.Cache.Close())
	}
	collect(r.ToolsDatabase.Close())
	collect(r.Classifier.Close())
	collect(closeReplayRecorders(r.ReplayRecorder, r.ReplayRecorders, r.ReplayStoreShared))
	collect(r.ModelSelector.Close())
	if r.MemoryStore != nil {
		collect(r.MemoryStore.Close())
	}
	collect(r.RateLimiter.Close())
	if r.CompressionRecovery != nil {
		collect(r.CompressionRecovery.Close())
	}
	collect(r.ResponseAPIFilter.Close())
	collect(r.closeToolSelectionDatabases())

	return errors.Join(errs...)
}

// closeReplayRecorders mirrors the shared-vs-isolated storage distinction the
// read paths already use (see collectRouterReplayRecords): a shared backend has
// every recorder wrapping the same store.Storage, so only replayRecorder is
// closed, while an isolated backend gives each decision its own store and needs
// every recorder closed. An empty map falls back to replayRecorder.
func closeReplayRecorders(
	replayRecorder *routerreplay.Recorder,
	replayRecorders map[string]*routerreplay.Recorder,
	replayStoreShared bool,
) error {
	if replayStoreShared || len(replayRecorders) == 0 {
		if replayRecorder == nil {
			return nil
		}
		return replayRecorder.Close()
	}

	var errs []error
	for _, recorder := range replayRecorders {
		if err := recorder.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

// Ensure OpenAIRouter implements the ext_proc calls.
var _ ext_proc.ExternalProcessorServer = (*OpenAIRouter)(nil)

const routerReplayAPIBasePath = "/v1/router_replay"

// createJSONResponseWithBody creates a direct response with pre-marshaled JSON
// body. When responsePath is non-empty, the v0.4 keystone headers
// (x-vsr-schema-version + x-vsr-response-path) are emitted for that path; pass
// "" for router-management responses (e.g. /v1/models) that are not routed LLM
// responses and therefore carry no response-path. See issue #2203.
func (r *OpenAIRouter) createJSONResponseWithBody(statusCode int, jsonBody []byte, responsePath string) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte("application/json"),
			},
		},
	}
	if responsePath != "" {
		setHeaders = append(setHeaders, httputil.KeystoneHeaderOptions(responsePath)...)
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCodeToImmediateResponseCode(statusCode),
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: setHeaders,
				},
				Body: jsonBody,
			},
		},
	}
}

// createSSEResponseWithBody creates a direct response with pre-marshaled SSE
// (text/event-stream) body. Used when the original client request requested
// streaming (stream: true) but the response is generated by modality routing
// (e.g. image generation) rather than a streaming model backend.
func (r *OpenAIRouter) createSSEResponseWithBody(statusCode int, sseBody []byte, responsePath string) *ext_proc.ProcessingResponse {
	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte("text/event-stream; charset=utf-8"),
			},
		},
	}
	if responsePath != "" {
		setHeaders = append(setHeaders, httputil.KeystoneHeaderOptions(responsePath)...)
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCodeToImmediateResponseCode(statusCode),
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: setHeaders,
				},
				Body: sseBody,
			},
		},
	}
}

// createJSONResponse creates a direct response with JSON content.
func (r *OpenAIRouter) createJSONResponse(statusCode int, data interface{}) *ext_proc.ProcessingResponse {
	jsonData, err := json.Marshal(data)
	if err != nil {
		logging.Errorf("Failed to marshal JSON response: %v", err)
		return r.createErrorResponse(500, "Internal server error")
	}

	// Router-management JSON responses (model lists, classify/route APIs, etc.)
	// are not routed LLM responses, so they carry no response-path.
	return r.createJSONResponseWithBody(statusCode, jsonData, "")
}

// createErrorResponse creates a direct error response.
func (r *OpenAIRouter) createErrorResponse(statusCode int, message string) *ext_proc.ProcessingResponse {
	errorResp := map[string]interface{}{
		"error": map[string]interface{}{
			"message": message,
			"type":    "invalid_request_error",
			"code":    statusCode,
		},
	}

	jsonData, err := json.Marshal(errorResp)
	if err != nil {
		logging.Errorf("Failed to marshal error response: %v", err)
		jsonData = []byte(`{"error":{"message":"Internal server error","type":"internal_error","code":500}}`)
		statusCode = 500
	}

	return r.createJSONResponseWithBody(statusCode, jsonData, headers.ResponsePathError)
}

// shouldClearRouteCache checks if route cache should be cleared.
func (r *OpenAIRouter) shouldClearRouteCache() bool {
	return r.Config.ClearRouteCache
}

// LoadToolsDatabase loads tools from file after embedding models are initialized.
func (r *OpenAIRouter) LoadToolsDatabase() error {
	if !r.ToolsDatabase.IsEnabled() {
		return nil
	}

	if r.Config.Tools.ToolsDBPath == "" {
		logging.Warnf("Tools database enabled but no tools file path configured; skipping load")
		return nil
	}

	if err := r.ToolsDatabase.LoadToolsFromFile(r.Config.Tools.ToolsDBPath); err != nil {
		return err
	}

	// Wire the default embedding retriever into the registry now that
	// the database is loaded and embeddings are available.
	r.ToolsRegistry = tools.NewDefaultRegistry(r.ToolsDatabase)

	return nil
}

// PreloadKnowledgeBases moves lazy KB embedding work out of the first routed
// request and into startup/reload readiness.
func (r *OpenAIRouter) PreloadKnowledgeBases() error {
	if r == nil {
		return nil
	}
	if r.RecipeClassifiers != nil {
		return r.RecipeClassifiers.PreloadKnowledgeBases()
	}
	if r.Classifier == nil {
		return nil
	}
	return r.Classifier.PreloadKnowledgeBases()
}

func (r *OpenAIRouter) RegisterToolStrategy(name string, retriever tools.ToolRetriever) {
	if r.ToolsRegistry == nil {
		r.ToolsRegistry = tools.NewRegistry()
	}
	r.ToolsRegistry.Register(name, retriever)
}
