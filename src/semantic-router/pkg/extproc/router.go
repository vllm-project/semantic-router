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

	// generation owns every closeable resource buildRouterComponents
	// produced for this router, registered in construction order so Close
	// tears them down in reverse. It is nil for routers assembled by hand
	// (e.g. in tests), which Close falls back to closing field by field.
	generation *routerruntime.Generation
	closeOnce  sync.Once
	closeErr   error
}

// Close releases every resource this router owns. It is idempotent and safe
// to call concurrently: only the first call runs the teardown, and every
// caller observes the same error. Idempotence matters because a router can
// be reached by more than one shutdown path — a config reload retiring the
// lease it replaced, and process shutdown retiring whatever lease is
// current — and the underlying resources (gRPC connections, MCP clients)
// are not all safe to close twice.
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
		// Built via buildRouterComponents, which registered a closer for
		// every resource it created — including lookupTableCancel — so the
		// generation is the single source of truth for what this router
		// owns and the only thing Close has to drive.
		return r.generation.Close()
	}
	return r.closeOwnedFields()
}

// closeOwnedFields closes the lookup table goroutines plus every closeable
// field (cache, tools database, classifier, replay recorder(s), model
// selector, memory store, rate limiter) and the lazily loaded per-path tool
// databases. It is the fallback for routers assembled by hand rather than by
// buildRouterComponents, which registers the same set of resources on a
// Generation. Keep the two in step: a resource added to one and not the other
// leaks on exactly one of the two construction paths.
//
// Note the deliberate asymmetry in nil handling: a method value such as
// r.ToolsDatabase.Close is non-nil even when the receiver is a nil pointer,
// so pointer-typed fields rely on the nil-receiver guard inside their own
// Close. Interface-typed fields (Cache, MemoryStore) must be nil-checked
// here, because taking a method value off a nil interface panics.
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
	collect(r.closeToolSelectionDatabases())

	return errors.Join(errs...)
}

// closeReplayRecorders closes the replay recorder(s) a router owns,
// mirroring the shared-vs-isolated storage distinction already used for
// read paths in router_replay_api.go (see collectRouterReplayRecords): a
// shared backend has every recorder in replayRecorders wrapping the same
// store.Storage, so only replayRecorder is closed to avoid closing that
// store more than once. An isolated backend gives each decision its own
// store, so every recorder in the map is closed individually; if the map is
// empty (e.g. a router assembled directly with only ReplayRecorder set,
// as in tests), replayRecorder is closed as a fallback.
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
