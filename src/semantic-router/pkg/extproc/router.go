package extproc

import (
	"encoding/json"
	"errors"
	"sync"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/backendinvoker"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/sessiontelemetry"
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
	// InferenceAccess is the Router-native managed inference access runtime.
	// Nil is valid only when global.services.access.enabled=false.
	InferenceAccess InferenceAccessRuntime
	// OutcomeFeedback owns the authenticated public, replay-bound feedback seam.
	OutcomeFeedback OutcomeFeedbackRuntime
	// OutcomeProjection reads the globally published, rebuildable learning view.
	OutcomeProjection OutcomeLearningProjectionRuntime
	// DispatchCapabilities signs exact, short-lived calls to the private
	// backend invoker in every control-plane mode.
	DispatchCapabilities DispatchCapabilityRuntime
	// ResponseTerminals carries the neutral completion decoded by
	// BackendInvoker; response accounting never reparses client wire.
	ResponseTerminals backendinvoker.ResponseTerminalReader
	ProtocolCodecs    *protocolcodec.Registry

	// RuntimeRegistry exposes runtime-owned services without forcing request-time
	// paths back through package-global API-server state.
	RuntimeRegistry *routerruntime.Registry

	routerLearningMu        sync.Mutex
	routerLearningRuntime   *routerLearningRuntime
	lookupTableCancel       func()
	routerSessionStateStore *sessiontelemetry.RouterSessionStateStoreSlot
}

// Close releases background resources held by the router (e.g. lookup table
// auto-save and periodic re-population goroutines).
func (r *OpenAIRouter) Close() error {
	if r == nil {
		return nil
	}
	if r.lookupTableCancel != nil {
		r.lookupTableCancel()
	}
	r.routerLearningMu.Lock()
	learningRuntime := r.routerLearningRuntime
	r.routerLearningMu.Unlock()
	if learningRuntime != nil {
		learningRuntime.RetireAndWait()
	}
	var closeErrors []error
	if r.CompressionRecovery != nil {
		closeErrors = append(closeErrors, r.CompressionRecovery.Close())
	}
	if r.routerSessionStateStore != nil {
		sessiontelemetry.UnpublishRouterSessionStateStore(r.routerSessionStateStore)
		closeErrors = append(closeErrors, r.routerSessionStateStore.RetireAndClose())
	}
	closeErrors = append(closeErrors, r.closeReplayRecorders())
	return errors.Join(closeErrors...)
}

func (r *OpenAIRouter) closeReplayRecorders() error {
	if r.ReplayStoreShared {
		if r.ReplayRecorder == nil {
			return nil
		}
		return r.ReplayRecorder.Close()
	}
	seen := make(map[*routerreplay.Recorder]struct{}, len(r.ReplayRecorders)+1)
	var closeErrors []error
	for _, recorder := range r.ReplayRecorders {
		if recorder == nil {
			continue
		}
		if _, duplicate := seen[recorder]; duplicate {
			continue
		}
		seen[recorder] = struct{}{}
		closeErrors = append(closeErrors, recorder.Close())
	}
	if r.ReplayRecorder != nil {
		if _, duplicate := seen[r.ReplayRecorder]; !duplicate {
			closeErrors = append(closeErrors, r.ReplayRecorder.Close())
		}
	}
	return errors.Join(closeErrors...)
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
	return nil
}

func (r *OpenAIRouter) RegisterToolStrategy(name string, retriever tools.ToolRetriever) {
	if r.ToolsRegistry == nil {
		r.ToolsRegistry = tools.NewRegistry()
	}
	r.ToolsRegistry.Register(name, retriever)
}
