//go:build !windows && cgo

package apiserver

import (
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configschema"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/publicmodels"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerruntime"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/startupstatus"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func apiHealthRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/health", Method: "GET", Description: "Health check endpoint"},
			routePolicy{Permission: PermHealthRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleHealth,
			jsonResponse[healthResponse](http.StatusOK, "Successful response"),
		),
		managedRoute(
			EndpointMetadata{Path: "/ready", Method: "GET", Description: "Readiness endpoint that turns green only after startup completes"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleReady,
			jsonResponse[readinessResponse](http.StatusOK, "Successful response"),
			jsonResponse[readinessResponse](http.StatusServiceUnavailable, "Startup is incomplete"),
		),
		managedRoute(
			EndpointMetadata{Path: "/startup-status", Method: "GET", Description: "Detailed router startup and model-download status"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleStartupStatus,
			jsonResponse[startupstatus.State](http.StatusOK, "Successful response"),
			jsonResponse[startupstatus.State](http.StatusServiceUnavailable, "Startup state or readiness is unavailable"),
			emptyResponse(http.StatusInternalServerError, "Startup status could not be encoded"),
		),
		managedRoute(
			EndpointMetadata{Path: apiStatusPath, Method: "GET", Description: "Versioned replica-local startup and configuration status; not a deployment probe"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleStatus,
			jsonResponse[routerruntime.StatusReport](http.StatusOK, "Replica-local observations; unknown readiness is not ready"),
			errorResponses(http.StatusServiceUnavailable),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiRootPath,
				Method:      "GET",
				Description: "Progressive API capability discovery",
				Parameters: []OpenAPIParameter{
					{Name: "view", In: "query", Description: "Omit for a compact capability index or use operations to include endpoint metadata.", Schema: OpenAPISchema{Type: "string", Enum: []string{"index", "operations"}}},
					{Name: "capability", In: "query", Description: "Return one capability group.", Schema: OpenAPISchema{Type: "string"}},
					{Name: "audience", In: "query", Description: "Return operations intended for one caller type.", Schema: OpenAPISchema{Type: "string", Enum: []string{"agent", "operator", "client", "internal"}}},
					{Name: "plane", In: "query", Description: "Return operations from one API plane.", Schema: OpenAPISchema{Type: "string", Enum: []string{"infrastructure", "management", "diagnostic", "data"}}},
					{Name: "visibility", In: "query", Description: "Return primary or advanced operations.", Schema: OpenAPISchema{Type: "string", Enum: []string{"primary", "advanced"}}},
				},
			},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleAPIOverview,
			jsonResponse[APIOverviewResponse](http.StatusOK, "Successful response"),
			errorResponses(400),
		),
		managedRoute(
			EndpointMetadata{
				Path:        "/openapi.json",
				Method:      "GET",
				Description: "OpenAPI 3.0 specification; optionally narrowed to one path or operation",
				Parameters: []OpenAPIParameter{
					{Name: "path", In: "query", Description: "Exact API path to return, for example /api/v1/config.", Schema: OpenAPISchema{Type: "string"}},
					{Name: "method", In: "query", Description: "HTTP method to return for the selected path.", Schema: OpenAPISchema{Type: "string", Enum: []string{"GET", "POST", "PATCH", "PUT", "DELETE"}}},
					{Name: "capability", In: "query", Description: "Return operations in one capability group.", Schema: OpenAPISchema{Type: "string"}},
					{Name: "audience", In: "query", Description: "Return operations intended for agent, operator, client, or internal callers.", Schema: OpenAPISchema{Type: "string", Enum: []string{"agent", "operator", "client", "internal"}}},
					{Name: "plane", In: "query", Description: "Return operations from one API plane.", Schema: OpenAPISchema{Type: "string", Enum: []string{"infrastructure", "management", "diagnostic", "data"}}},
					{Name: "visibility", In: "query", Description: "Return primary or advanced operations.", Schema: OpenAPISchema{Type: "string", Enum: []string{"primary", "advanced"}}},
				},
			},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleOpenAPISpec,
			jsonResponse[OpenAPISpec](http.StatusOK, "Successful response"),
			errorResponses(400, 404),
		),
		managedRoute(
			EndpointMetadata{Path: "/docs", Method: "GET", Description: "Interactive Swagger UI documentation"},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleSwaggerUI,
			mediaResponse(http.StatusOK, "Interactive API documentation", "text/html", OpenAPISchema{Type: "string"}),
		),
	}
}

func apiClassifyRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/intent", Method: "POST", Description: "Classify user queries into routing categories"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleIntentClassification,
			jsonResponse[services.IntentResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[services.IntentRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/pii", Method: "POST", Description: "Detect personally identifiable information in text"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handlePIIDetection,
			jsonResponse[services.PIIResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[services.PIIRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/security", Method: "POST", Description: "Detect jailbreak attempts and security threats"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSecurityDetection,
			jsonResponse[services.SecurityResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[services.SecurityRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/fact-check", Method: "POST", Description: "Classify if text needs fact-checking"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleFactCheckClassification,
			jsonResponse[services.FactCheckResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[services.FactCheckRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/user-feedback", Method: "POST", Description: "Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleUserFeedbackClassification,
			jsonResponse[services.UserFeedbackResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[services.UserFeedbackRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/combined", Method: "POST", Description: "Perform combined classification (intent, PII, and security)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleCombinedClassification,
			jsonResponse[CombinedClassificationResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[CombinedClassificationRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/batch", Method: "POST", Description: "Batch classification with configurable task_type parameter"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchClassification,
			jsonResponse[BatchClassificationResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[BatchClassificationRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/nli", Method: "POST", Description: "Natural language inference classification for premise and hypothesis pairs"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleNLIClassification,
			jsonResponse[services.NLIResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[services.NLIRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/embeddings", Method: "POST", Description: "Generate text and image embeddings"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEmbeddings,
			jsonResponse[EmbeddingResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[EmbeddingRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/similarity", Method: "POST", Description: "Calculate pairwise text similarity"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSimilarity,
			jsonResponse[SimilarityResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[SimilarityRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/similarity/batch", Method: "POST", Description: "Calculate batch text-similarity matches"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchSimilarity,
			jsonResponse[BatchSimilarityResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 413, 429, 500, 503),
			jsonBodyFor[BatchSimilarityRequest](),
		),
	}
}

func apiRoutingRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{
				Path:        apiRoutingPreviewPath,
				Method:      "POST",
				Description: "Preview configured signals and model selection without generating an answer. Supported native-output requests use backend render APIs to resolve per-candidate capacity; paths requiring execution remain unresolved. Learning uses read-only captured state with selection_provenance; preview_context supplies session identity and an optional preview-only sampling seed. A state-dependent or sampled result does not guarantee a later live selection. global.services.api.routing_preview controls the request deadline and concurrent worker bound.",
				Parameters: []OpenAPIParameter{
					queryParameter("trace", "Include per-decision routing trace trees.", "boolean"),
					headerParameter(headers.SRBenchExpectedConfigHash, "Optional lowercase SHA-256 of the active runtime document. Rejects a mismatched generation before evaluating signals.", false),
				},
			},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEvalClassification,
			jsonResponse[services.EvalResponse](http.StatusOK, "Successful response"),
			responseHeaders(http.StatusOK, map[string]OpenAPIHeader{headers.VSRConfigHash: {Description: "SHA-256 of the actual runtime generation used by this Preview.", Schema: OpenAPISchema{Type: "string"}}}),
			errorResponses(400, 412, 413, 429, 500, 504),
			jsonResponseOrError[services.EvalResponse](http.StatusServiceUnavailable, "Decision unresolved, inference canceled, or server shutting down"),
			strictJSONBodyFor[services.IntentRequest](),
		),
	}
}

func apiInventoryRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiInventoryModelsPath, Method: "GET", Description: "Get information about loaded models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleModelsInfo,
			jsonResponse[ModelsInfoResponse](http.StatusOK, "Successful response"),
		),
		managedRoute(
			EndpointMetadata{Path: apiInventoryClassifierPath, Method: "GET", Description: "Get classifier information and status (secrets redacted without secret_view)"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleClassifierInfo,
			jsonResponse[classifierConfigResponse](http.StatusOK, "Successful response"),
		),
		managedRoute(
			EndpointMetadata{Path: apiInventoryEmbeddingModels, Method: "GET", Description: "Get information about loaded embedding models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleEmbeddingModelsInfo,
			jsonResponse[embeddingModelsResponse](http.StatusOK, "Successful response"),
		),
	}
}

func apiOpenAIDataRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/v1/models", Method: "GET", Description: "OpenAI-compatible public model and Entrypoint listing"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleOpenAIModels,
			jsonResponse[publicmodels.OpenAIModelList](http.StatusOK, "Successful response"),
		),
	}
}

func apiObservabilityRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiClassificationMetricsPath, Method: "GET", Description: "Get classification metrics and statistics"},
			routePolicy{Permission: PermMetricsRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleClassificationMetrics,
			jsonResponse[ClassificationMetricsResponse](http.StatusOK, "Successful response"),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiObservabilityOutcomesPath,
				Method:      "POST",
				Description: "Submit Router Learning outcome feedback linked to a replay record",
				Parameters: []OpenAPIParameter{
					headerParameter("Idempotency-Key", "Stable retry key for outcome ingestion.", false),
				},
			},
			routePolicy{Permission: PermLearningIngest, Sensitivity: SensitivityMutation, AuditAction: AuditActionOutcomeIngest},
			(*ClassificationAPIServer).handleRouterOutcome,
			jsonResponse[RouterOutcomeResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 409, 429, 503),
			jsonBodyFor[RouterOutcomeRequest](),
		),
	}
}

func apiResponseCacheRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/capabilities", Method: "GET", Description: "Get response-cache backend capabilities"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheCapabilities,
			pluginOperationFor(config.DecisionPluginResponseCache, "read"),
			jsonResponse[cache.BackendCapabilities](http.StatusOK, "Successful response"),
			errorResponses(503),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/health", Method: "GET", Description: "Check response-cache backend health"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheHealth,
			pluginOperationFor(config.DecisionPluginResponseCache, "read"),
			jsonResponse[cacheHealthResponse](http.StatusOK, "Successful response"),
			errorResponses(503),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/stats", Method: "GET", Description: "Get redacted response-cache statistics"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheStats,
			pluginOperationFor(config.DecisionPluginResponseCache, "read"),
			jsonResponse[cache.CacheStats](http.StatusOK, "Successful response"),
			errorResponses(503),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/test", Method: "POST", Description: "Validate and probe a response-cache candidate configuration"},
			routePolicy{Permission: PermCacheManage, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheTest,
			pluginOperationFor(config.DecisionPluginResponseCache, "probe"),
			jsonResponse[responseCacheTestResponse](http.StatusOK, "Successful response"),
			errorResponses(400),
			jsonResponseOrError[responseCacheTestResponse](http.StatusServiceUnavailable, "Candidate backend is unhealthy or its embedding runtime is unavailable"),
			jsonBodyFor[responseCacheTestRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/invalidate", Method: "POST", Description: "Dry-run or invalidate a scoped response-cache partition"},
			routePolicy{Permission: PermCacheInvalidate, Sensitivity: SensitivityMutation, AuditAction: AuditActionCacheInvalidate},
			(*ClassificationAPIServer).handleResponseCacheInvalidate,
			pluginOperationFor(config.DecisionPluginResponseCache, "mutation"),
			jsonResponse[cache.InvalidationResult](http.StatusOK, "Successful response"),
			errorResponses(400, 501, 503),
			jsonBodyFor[responseCacheInvalidateRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/flush", Method: "POST", Description: "Advance a scoped or global response-cache epoch"},
			routePolicy{Permission: PermCacheManage, Sensitivity: SensitivityMutation, AuditAction: AuditActionCacheFlush},
			(*ClassificationAPIServer).handleResponseCacheFlush,
			pluginOperationFor(config.DecisionPluginResponseCache, "mutation"),
			jsonResponse[cache.InvalidationResult](http.StatusOK, "Successful response"),
			errorResponses(400, 501, 503),
			jsonBodyFor[responseCacheFlushRequest](),
		),
	}
}

func apiContextCompressionRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/capabilities", Method: "GET", Description: "Get context-compression capabilities"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionCapabilities,
			pluginOperationFor(config.DecisionPluginContextCompression, "read"),
			jsonResponse[contextcompression.Capabilities](http.StatusOK, "Successful response"),
		),
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/health", Method: "GET", Description: "Check context-compression runtime health"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionHealth,
			pluginOperationFor(config.DecisionPluginContextCompression, "read"),
			jsonResponse[compressionHealthResponse](http.StatusOK, "Successful response"),
			errorResponses(503),
		),
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/preview", Method: "POST", Description: "Preview context compression without persistence"},
			routePolicy{Permission: PermCompressionPreview, Sensitivity: SensitivityOperational, AuditAction: AuditActionCompressionPreview},
			(*ClassificationAPIServer).handleContextCompressionPreview,
			pluginOperationFor(config.DecisionPluginContextCompression, "preview"),
			jsonResponse[contextCompressionPreviewResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 500, 502, 503),
			jsonBodyFor[contextCompressionPreviewRequest](),
		),
	}
}

func apiConfigRoutes() []apiRoute {
	return append(apiRecipeRoutes(), apiNonRecipeConfigRoutes()...)
}

func apiRecipeRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiRecipesPath, Method: "GET", Description: "List the default and named routing recipes with their entrypoints"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleListRecipes,
			jsonResponse[recipeCollectionResponse](http.StatusOK, "Successful response"),
			errorResponses(500),
			etagResponseHeaders(http.StatusOK),
		),
		managedRoute(
			EndpointMetadata{Path: apiRecipesPath + "/validate", Method: "POST", Description: "Validate a recipe mutation without writing or reloading config"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleValidateRecipe,
			jsonResponse[recipeValidationResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 500),
			strictJSONBodyFor[recipeMutationRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiRecipesPath + "/{name}", Method: "GET", Description: "Read one routing recipe and its entrypoints"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleGetRecipe,
			jsonResponse[managedRecipeRecord](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 500),
			etagResponseHeaders(http.StatusOK),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiRecipesPath + "/{name}",
				Method:      "PUT",
				Description: "Atomically create or replace one routing recipe; requires If-Match",
				Parameters: []OpenAPIParameter{
					headerParameter("If-Match", "ETag returned by the current recipe or recipe collection.", true),
				},
			},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionRecipeSave},
			(*ClassificationAPIServer).handlePutRecipe,
			configMutationResponses(),
			jsonResponse[RouterConfigUpdateResponse](http.StatusCreated, "Recipe created"),
			etagResponseHeaders(http.StatusCreated),
			strictJSONBodyFor[recipeMutationRequest](),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiRecipesPath + "/{name}",
				Method:      "DELETE",
				Description: "Delete an unreferenced named routing recipe; requires If-Match",
				Parameters: []OpenAPIParameter{
					headerParameter("If-Match", "ETag returned by the current recipe or recipe collection.", true),
				},
			},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionRecipeDelete},
			(*ClassificationAPIServer).handleDeleteRecipe,
			configMutationResponses(),
		),
	}
}

func apiNonRecipeConfigRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{
				Path:        apiConfigSchemaPath,
				Method:      "GET",
				Description: "Discover the canonical Router configuration contract progressively or return the complete JSON Schema",
				Parameters: []OpenAPIParameter{
					{Name: "view", In: "query", Description: "Representation to return. Omit for the compact index; use full for the complete schema.", Schema: OpenAPISchema{Type: "string", Enum: []string{"full", "index", "section", "surface"}}},
					{Name: "path", In: "query", Description: "Dot- or slash-delimited config path required by view=section.", Schema: OpenAPISchema{Type: "string"}},
					headerParameter("If-None-Match", "Return 304 when the selected schema representation has this ETag.", false),
					{Name: "expanded", In: "query", Description: "Return a self-contained JSON Schema for view=section instead of its compact field directory.", Schema: OpenAPISchema{Type: "boolean"}},
					{Name: "kind", In: "query", Description: "Surface kind required by view=surface.", Schema: OpenAPISchema{Type: "string", Enum: []string{"signal", "algorithm", "plugin", "projection"}}},
					{Name: "name", In: "query", Description: "Registered surface name required by view=surface.", Schema: OpenAPISchema{Type: "string"}},
				},
			},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleConfigSchema,
			errorResponses(400, 500),
			mediaResponse(http.StatusOK, "Progressive schema directory or section field summary", "application/json", OpenAPISchema{OneOf: []OpenAPISchema{*openAPIRequestSchemaFor[configschema.IndexResponse](), *openAPIRequestSchemaFor[configschema.SectionResponse]()}}),
			mediaResponse(http.StatusOK, "Complete or selected JSON Schema document", "application/schema+json", OpenAPISchema{Type: "object", AdditionalProperties: true}),
			etagResponseHeaders(http.StatusOK),
			emptyResponse(http.StatusNotModified, "Schema matches If-None-Match"),
			etagResponseHeaders(http.StatusNotModified),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigPath, Method: "GET", Description: "Get the current router config as JSON (secrets redacted without secret_view)"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleConfigGet,
			errorResponses(500),
			mediaResponse(http.StatusOK, "Canonical configuration document; field schemas are available at /api/v1/config/schema?view=full", "application/json", OpenAPISchema{Type: "object", AdditionalProperties: true}),
			etagResponseHeaders(http.StatusOK),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigValidatePath, Method: "POST", Description: "Validate and normalize a router config without writing it"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigValidate,
			jsonResponse[RouterConfigValidateResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 500),
			strictJSONBodyFor[RouterConfigUpdateRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigPlanPath, Method: "POST", Description: "Plan an exact merge or replace mutation, including hot-reload compatibility, without writing it"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigPlan,
			jsonResponse[routerConfigPlanResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 409, 500),
			strictJSONBodyFor[routerConfigPlanRequest](),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiConfigPath,
				Method:      "PATCH",
				Description: "Compare-and-swap merge of a router config update (validates, backs up, writes, triggers hot-reload)",
				Parameters: []OpenAPIParameter{
					headerParameter("If-Match", "Required ETag returned by GET /api/v1/config or POST /api/v1/config/plan.", true),
				},
			},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionConfigPatch},
			(*ClassificationAPIServer).handleConfigPatch,
			configMutationResponses(),
			strictJSONBodyFor[RouterConfigUpdateRequest](),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiConfigPath,
				Method:      "PUT",
				Description: "Compare-and-swap replacement of the router config (validates, backs up, writes, triggers hot-reload)",
				Parameters: []OpenAPIParameter{
					headerParameter("If-Match", "Required ETag returned by GET /api/v1/config or POST /api/v1/config/plan.", true),
				},
			},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionConfigPut},
			(*ClassificationAPIServer).handleConfigPut,
			configMutationResponses(),
			strictJSONBodyFor[RouterConfigUpdateRequest](),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiConfigRollbackPath,
				Method:      "POST",
				Description: "Compare-and-swap rollback to a previous router config version",
				Parameters: []OpenAPIParameter{
					headerParameter("If-Match", "Required ETag returned by GET /api/v1/config.", true),
				},
			},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionConfigRollback},
			(*ClassificationAPIServer).handleConfigRollback,
			configMutationResponses(),
			strictJSONBodyFor[routerConfigRollbackRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigVersionsPath, Method: "GET", Description: "List available router config backup versions"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigVersions,
			jsonResponse[[]RouterConfigVersionEntry](http.StatusOK, "Successful response"),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigHashPath, Method: "GET", Description: "Compare persisted source, generated runtime, and active router config hashes"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigHash,
			jsonResponse[configHashResponse](http.StatusOK, "Successful response"),
			errorResponses(500),
		),
	}
}

func apiKnowledgeBaseRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath, Method: "GET", Description: "List configured knowledge bases"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListKnowledgeBases,
			jsonResponse[knowledgeBaseListResponse](http.StatusOK, "Successful response"),
			errorResponses(500),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath, Method: "POST", Description: "Create a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseSave},
			(*ClassificationAPIServer).handleCreateKnowledgeBase,
			jsonResponse[knowledgeBaseDocument](http.StatusCreated, "Successful response"),
			errorResponses(400, 500),
			jsonResponse[knowledgeBaseDocument](http.StatusAccepted, "Saved candidate awaiting publication; poll /api/v1/config/hash for its generated_runtime_hash"),
			jsonResponse[knowledgeBaseDocument](http.StatusServiceUnavailable, "Candidate persisted but runtime activation failed; inspect activation and recover through /api/v1/config"),
			errorResponse(http.StatusConflict, "Conflict, including CONFIG_ACTIVATION_PENDING or CONFIG_ACTIVATION_FAILED while the saved candidate is not active"),
			jsonBodyFor[knowledgeBaseUpsertRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}", Method: "GET", Description: "Read a knowledge base"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBase,
			jsonResponse[knowledgeBaseDocument](http.StatusOK, "Successful response"),
			errorResponses(404, 500),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}/map/metadata", Method: "GET", Description: "Read generated knowledge-base map metadata"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBaseMapMetadata,
			jsonResponse[knowledgeBaseMapMetadataResponse](http.StatusOK, "Successful response"),
			errorResponses(404, 500),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}/map/data.ndjson", Method: "GET", Description: "Stream generated knowledge-base map data as NDJSON"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBaseMapData,
			errorResponses(404, 500),
			mediaResponse(http.StatusOK, "One map point JSON object per line", "application/x-ndjson", OpenAPISchema{Type: "string"}),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}", Method: "PUT", Description: "Update a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseSave},
			(*ClassificationAPIServer).handleUpdateKnowledgeBase,
			jsonResponse[knowledgeBaseDocument](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 500),
			jsonResponse[knowledgeBaseDocument](http.StatusAccepted, "Saved candidate awaiting publication; poll /api/v1/config/hash for its generated_runtime_hash"),
			jsonResponse[knowledgeBaseDocument](http.StatusServiceUnavailable, "Candidate persisted but runtime activation failed; inspect activation and recover through /api/v1/config"),
			errorResponse(http.StatusConflict, "Conflict, including CONFIG_ACTIVATION_PENDING or CONFIG_ACTIVATION_FAILED while the saved candidate is not active"),
			jsonBodyFor[knowledgeBaseUpsertRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}", Method: "DELETE", Description: "Delete a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseDel},
			(*ClassificationAPIServer).handleDeleteKnowledgeBase,
			jsonResponse[knowledgeBaseDeleteResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 500),
			jsonResponse[knowledgeBaseDeleteResponse](http.StatusAccepted, "Saved candidate awaiting publication; poll /api/v1/config/hash for its generated_runtime_hash"),
			jsonResponse[knowledgeBaseDeleteResponse](http.StatusServiceUnavailable, "Candidate persisted but runtime activation failed; inspect activation and recover through /api/v1/config"),
			errorResponse(http.StatusConflict, "Conflict, including CONFIG_ACTIVATION_PENDING or CONFIG_ACTIVATION_FAILED while the saved candidate is not active"),
		),
	}
}

func apiMemoryRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{
				Path:        apiStorageMemoriesPath,
				Method:      "GET",
				Description: "List long-term memories",
				Parameters: []OpenAPIParameter{
					queryParameter("user_id", "Development fallback identity when x-authz-user-id is unavailable.", "string"),
					queryParameter("type", "Comma-separated memory types: semantic, procedural, or episodic.", "string"),
					queryParameter("limit", "Maximum results; defaults to 20 and is capped at 100.", "integer"),
				},
			},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListMemories,
			pluginOperationFor(config.DecisionPluginMemory, "read"),
			jsonResponse[MemoryListResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 500, 503),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiStorageMemoriesPath,
				Method:      "DELETE",
				Description: "Delete memories by scope",
				Parameters: []OpenAPIParameter{
					queryParameter("user_id", "Development fallback identity when x-authz-user-id is unavailable.", "string"),
					queryParameter("type", "Comma-separated memory types to delete: semantic, procedural, or episodic.", "string"),
				},
			},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionMemoryDelete},
			(*ClassificationAPIServer).handleDeleteMemoriesByScope,
			pluginOperationFor(config.DecisionPluginMemory, "mutation"),
			jsonResponse[MemoryDeleteResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 500, 503),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiStorageMemoriesPath + "/{id}",
				Method:      "GET",
				Description: "Read one long-term memory",
				Parameters: []OpenAPIParameter{
					queryParameter("user_id", "Development fallback identity when x-authz-user-id is unavailable.", "string"),
				},
			},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetMemory,
			pluginOperationFor(config.DecisionPluginMemory, "read"),
			jsonResponse[MemoryResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 500, 503),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiStorageMemoriesPath + "/{id}",
				Method:      "DELETE",
				Description: "Delete one long-term memory",
				Parameters: []OpenAPIParameter{
					queryParameter("user_id", "Development fallback identity when x-authz-user-id is unavailable.", "string"),
				},
			},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionMemoryDelete},
			(*ClassificationAPIServer).handleDeleteMemory,
			pluginOperationFor(config.DecisionPluginMemory, "mutation"),
			jsonResponse[MemoryDeleteResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 500, 503),
		),
	}
}

func apiVectorStoreRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath, Method: "POST", Description: "Create a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleCreateVectorStore,
			pluginOperationFor(config.DecisionPluginRAG, "mutation"),
			jsonResponse[vectorstore.VectorStore](http.StatusOK, "Successful response"),
			errorResponses(400, 500, 503),
			jsonBodyWithLimitFor[vectorstore.CreateStoreRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiStorageVectorStoresPath,
				Method:      "GET",
				Description: "List vector stores",
				Parameters: []OpenAPIParameter{
					queryParameter("limit", "Maximum results; defaults to 20 and is capped at 100.", "integer"),
					queryParameter("order", "Sort order by creation time.", "string", "asc", "desc"),
					queryParameter("after", "Return results after this cursor.", "string"),
					queryParameter("before", "Return results before this cursor; mutually exclusive with after.", "string"),
				},
			},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStores,
			pluginOperationFor(config.DecisionPluginRAG, "read"),
			jsonResponse[objectListResponse[*vectorstore.VectorStore]](http.StatusOK, "Successful response"),
			errorResponses(400, 503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}", Method: "GET", Description: "Read a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetVectorStore,
			pluginOperationFor(config.DecisionPluginRAG, "read"),
			jsonResponse[vectorstore.VectorStore](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}", Method: "POST", Description: "Update a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUpdateVectorStore,
			pluginOperationFor(config.DecisionPluginRAG, "mutation"),
			jsonResponse[vectorstore.VectorStore](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
			jsonBodyWithLimitFor[vectorstore.UpdateStoreRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}", Method: "DELETE", Description: "Delete a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteVectorStore,
			pluginOperationFor(config.DecisionPluginRAG, "mutation"),
			jsonResponse[objectDeletedResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/search", Method: "POST", Description: "Search a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSearchVectorStore,
			pluginOperationFor(config.DecisionPluginRAG, "probe"),
			jsonResponse[objectListResponse[vectorstore.SearchResult]](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 500, 503),
			jsonBodyWithLimitFor[SearchRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/files", Method: "POST", Description: "Attach a file to a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleAttachFile,
			jsonResponse[vectorstore.VectorStoreFile](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
			jsonBodyWithLimitFor[AttachFileRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/files", Method: "GET", Description: "List files attached to a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStoreFiles,
			pluginOperationFor(config.DecisionPluginRAG, "read"),
			jsonResponse[objectListResponse[*vectorstore.VectorStoreFile]](http.StatusOK, "Successful response"),
			errorResponses(400, 503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/files/{file_id}", Method: "DELETE", Description: "Detach a file from a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDetachFile,
			jsonResponse[objectDeletedResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
		),
	}
}

func apiFileRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath, Method: "POST", Description: "Upload a file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUploadFile,
			jsonResponse[vectorstore.FileRecord](http.StatusOK, "Successful response"),
			errorResponses(400, 500, 503),
			multipartBody(maxUploadSize, "Multipart upload with a file field and optional purpose field. Documents (.txt, .md, .json, .csv, .html) by default; images (.png, .jpg, .jpeg, .gif, .webp) with purpose=vision."),
		),
		managedRoute(
			EndpointMetadata{
				Path:        apiStorageFilesPath,
				Method:      "GET",
				Description: "List uploaded files",
				Parameters: []OpenAPIParameter{
					queryParameter("purpose", "Filter files by purpose.", "string"),
				},
			},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListFiles,
			jsonResponse[objectListResponse[*vectorstore.FileRecord]](http.StatusOK, "Successful response"),
			errorResponses(503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath + "/{id}", Method: "GET", Description: "Read uploaded-file metadata"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetFile,
			jsonResponse[vectorstore.FileRecord](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath + "/{id}", Method: "DELETE", Description: "Delete an uploaded file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteFile,
			jsonResponse[objectDeletedResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 404, 503),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath + "/{id}/content", Method: "GET", Description: "Download uploaded-file content"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetFileContent,
			errorResponses(400, 404, 500, 503),
			mediaResponse(http.StatusOK, "Uploaded file bytes", "application/octet-stream", OpenAPISchema{Type: "string", Format: "binary"}),
			responseHeaders(http.StatusOK, map[string]OpenAPIHeader{"Content-Disposition": {Description: "Download filename", Schema: OpenAPISchema{Type: "string"}}}),
		),
	}
}

func appendAPIRoutes(routes []apiRoute, groups ...[]apiRoute) []apiRoute {
	for _, group := range groups {
		routes = append(routes, group...)
	}
	return routes
}

func apiPluginObservabilityRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiObservabilityPath + "/plugins/context_compression/stats", Method: "GET", Description: "Get redacted context-compression statistics"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionStats,
			pluginOperationFor(config.DecisionPluginContextCompression, "read"),
			jsonResponse[contextCompressionStatsResponse](http.StatusOK, "Successful response"),
			errorResponses(503),
		),
	}
}

func apiContextRecoveryRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStoragePath + "/context-recovery/invalidate", Method: "POST", Description: "Invalidate a trusted context-recovery request scope"},
			routePolicy{Permission: PermCompressionManage, Sensitivity: SensitivityMutation, AuditAction: AuditActionCompressionInvalidate},
			(*ClassificationAPIServer).handleContextCompressionRecoveryInvalidate,
			pluginOperationFor(config.DecisionPluginContextCompression, "mutation"),
			jsonResponse[recoveryInvalidationResponse](http.StatusOK, "Successful response"),
			errorResponses(400, 502, 503),
			jsonBodyFor[contextCompressionRecoveryInvalidateRequest](),
		),
	}
}
