package extproc

import (
	"encoding/json"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/ratelimit"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/tools"
)

// OpenAIRouter is an Envoy ExtProc server that routes OpenAI API requests.
type OpenAIRouter struct {
	Config               *config.RouterConfig
	CategoryDescriptions []string
	Classifier           *classification.Classifier
	Cache                cache.CacheBackend
	ToolsDatabase        *tools.ToolsDatabase
	ResponseAPIFilter    *ResponseAPIFilter
	ReplayRecorder       *routerreplay.Recorder
	// ModelSelector is the registry of advanced model selection algorithms
	// initialized from config.IntelligentRouting.ModelSelection.
	ModelSelector   *selection.Registry
	ReplayRecorders map[string]*routerreplay.Recorder
	MemoryStore     *memory.MilvusStore
	MemoryExtractor *memory.MemoryExtractor

	// CredentialResolver resolves per-user LLM API keys from multiple sources
	// (ext_authz injected headers -> static config fallback).
	CredentialResolver *authz.CredentialResolver

	// RateLimiter enforces per-user/model rate limits from multiple sources
	// (Envoy RLS -> local limiter).
	RateLimiter *ratelimit.RateLimitResolver
}

// Ensure OpenAIRouter implements the ext_proc calls.
var _ ext_proc.ExternalProcessorServer = (*OpenAIRouter)(nil)

const routerReplayAPIBasePath = "/v1/router_replay"

// handleRouterReplayAPI serves read-only endpoints for router replay records.
func (r *OpenAIRouter) handleRouterReplayAPI(method string, path string) *ext_proc.ProcessingResponse {
	if !r.hasRouterReplayRecorders() {
		return nil
	}

	path = normalizeRouterReplayAPIPath(path)

	switch {
	case isRouterReplayListPath(path):
		return r.handleRouterReplayListAPI(method)
	case strings.HasPrefix(path, routerReplayAPIBasePath+"/"):
		replayID := strings.TrimPrefix(path, routerReplayAPIBasePath+"/")
		return r.handleRouterReplayRecordAPI(method, replayID)
	default:
		return nil
	}
}

func (r *OpenAIRouter) hasRouterReplayRecorders() bool {
	return len(r.ReplayRecorders) > 0 || r.ReplayRecorder != nil
}

func normalizeRouterReplayAPIPath(path string) string {
	if idx := strings.Index(path, "?"); idx != -1 {
		return path[:idx]
	}
	return path
}

func isRouterReplayListPath(path string) bool {
	return path == routerReplayAPIBasePath || path == routerReplayAPIBasePath+"/"
}

func (r *OpenAIRouter) handleRouterReplayListAPI(method string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}

	records := r.collectRouterReplayRecords()
	payload := map[string]interface{}{
		"object": "router_replay.list",
		"count":  len(records),
		"data":   records,
	}
	return r.createJSONResponse(200, payload)
}

func (r *OpenAIRouter) collectRouterReplayRecords() []routerreplay.RoutingRecord {
	var records []routerreplay.RoutingRecord
	for _, recorder := range r.ReplayRecorders {
		records = append(records, recorder.ListAllRecords()...)
	}
	if len(records) == 0 && r.ReplayRecorder != nil {
		return r.ReplayRecorder.ListAllRecords()
	}
	return records
}

func (r *OpenAIRouter) handleRouterReplayRecordAPI(method string, replayID string) *ext_proc.ProcessingResponse {
	if method != "GET" {
		return r.createErrorResponse(405, "method not allowed")
	}
	if replayID == "" {
		return r.createErrorResponse(400, "replay id is required")
	}

	record, ok := r.findRouterReplayRecord(replayID)
	if !ok {
		return r.createErrorResponse(404, "replay record not found")
	}
	return r.createJSONResponse(200, record)
}

func (r *OpenAIRouter) findRouterReplayRecord(replayID string) (routerreplay.RoutingRecord, bool) {
	for _, recorder := range r.ReplayRecorders {
		if record, ok := recorder.GetRecord(replayID); ok {
			return record, true
		}
	}
	if r.ReplayRecorder != nil {
		return r.ReplayRecorder.GetRecord(replayID)
	}
	return routerreplay.RoutingRecord{}, false
}

// createJSONResponseWithBody creates a direct response with pre-marshaled JSON body.
func (r *OpenAIRouter) createJSONResponseWithBody(statusCode int, jsonBody []byte) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCodeToImmediateResponseCode(statusCode),
				},
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						},
					},
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

	return r.createJSONResponseWithBody(statusCode, jsonData)
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

	return r.createJSONResponseWithBody(statusCode, jsonData)
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
		logging.Warnf("Tools database enabled but no tools file path configured")
		return nil
	}

	if err := r.ToolsDatabase.LoadToolsFromFile(r.Config.Tools.ToolsDBPath); err != nil {
		return err
	}

	logging.Infof("Tools database loaded successfully from: %s", r.Config.Tools.ToolsDBPath)
	return nil
}
