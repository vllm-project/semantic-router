//go:build !windows && cgo

package apiserver

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/services"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

func apiHealthRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/health", Method: "GET", Description: "Health check endpoint"},
			routePolicy{Permission: PermHealthRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleHealth,
		),
		managedRoute(
			EndpointMetadata{Path: "/ready", Method: "GET", Description: "Readiness endpoint that turns green only after startup completes"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleReady,
		),
		managedRoute(
			EndpointMetadata{Path: "/startup-status", Method: "GET", Description: "Detailed router startup and model-download status"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleStartupStatus,
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
		),
		managedRoute(
			EndpointMetadata{Path: "/docs", Method: "GET", Description: "Interactive Swagger UI documentation"},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleSwaggerUI,
		),
	}
}

func apiClassifyRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/intent", Method: "POST", Description: "Classify user queries into routing categories"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleIntentClassification,
			jsonBodyFor[services.IntentRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/pii", Method: "POST", Description: "Detect personally identifiable information in text"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handlePIIDetection,
			jsonBodyFor[services.PIIRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/security", Method: "POST", Description: "Detect jailbreak attempts and security threats"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSecurityDetection,
			jsonBodyFor[services.SecurityRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/fact-check", Method: "POST", Description: "Classify if text needs fact-checking"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleFactCheckClassification,
			jsonBodyFor[services.FactCheckRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/user-feedback", Method: "POST", Description: "Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleUserFeedbackClassification,
			jsonBodyFor[services.UserFeedbackRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/combined", Method: "POST", Description: "Perform combined classification (intent, PII, and security)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleCombinedClassification,
			jsonBodyFor[CombinedClassificationRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/classify/batch", Method: "POST", Description: "Batch classification with configurable task_type parameter"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchClassification,
			jsonBodyFor[BatchClassificationRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/nli", Method: "POST", Description: "Natural language inference classification for premise and hypothesis pairs"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleNLIClassification,
			jsonBodyFor[services.NLIRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/embeddings", Method: "POST", Description: "Generate text and image embeddings"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEmbeddings,
			jsonBodyFor[EmbeddingRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/similarity", Method: "POST", Description: "Calculate pairwise text similarity"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSimilarity,
			jsonBodyFor[SimilarityRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiDiagnosticsPath + "/similarity/batch", Method: "POST", Description: "Calculate batch text-similarity matches"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchSimilarity,
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
				Description: "Preview all configured signals and the resulting route without invoking a generation backend",
				Parameters: []OpenAPIParameter{
					queryParameter("trace", "Include per-decision routing trace trees.", "boolean"),
				},
			},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEvalClassification,
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
		),
		managedRoute(
			EndpointMetadata{Path: apiInventoryClassifierPath, Method: "GET", Description: "Get classifier information and status (secrets redacted without secret_view)"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleClassifierInfo,
		),
		managedRoute(
			EndpointMetadata{Path: apiInventoryEmbeddingModels, Method: "GET", Description: "Get information about loaded embedding models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleEmbeddingModelsInfo,
		),
	}
}

func apiOpenAIDataRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/v1/models", Method: "GET", Description: "OpenAI-compatible public model and Entrypoint listing"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleOpenAIModels,
		),
	}
}

func apiObservabilityRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiClassificationMetricsPath, Method: "GET", Description: "Get classification metrics and statistics"},
			routePolicy{Permission: PermMetricsRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleClassificationMetrics,
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
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/health", Method: "GET", Description: "Check response-cache backend health"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheHealth,
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/stats", Method: "GET", Description: "Get redacted response-cache statistics"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheStats,
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/audit", Method: "GET", Description: "Get redacted response-cache mutation audit entries"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheAudit,
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/test", Method: "POST", Description: "Validate and probe a response-cache candidate configuration"},
			routePolicy{Permission: PermCacheManage, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheTest,
			jsonBodyFor[responseCacheTestRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/invalidate", Method: "POST", Description: "Dry-run or invalidate a scoped response-cache partition"},
			routePolicy{Permission: PermCacheInvalidate, Sensitivity: SensitivityMutation, AuditAction: AuditActionCacheInvalidate},
			(*ClassificationAPIServer).handleResponseCacheInvalidate,
			jsonBodyFor[responseCacheInvalidateRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiResponseCachePath + "/flush", Method: "POST", Description: "Advance a scoped or global response-cache epoch"},
			routePolicy{Permission: PermCacheManage, Sensitivity: SensitivityMutation, AuditAction: AuditActionCacheFlush},
			(*ClassificationAPIServer).handleResponseCacheFlush,
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
		),
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/health", Method: "GET", Description: "Check context-compression runtime health"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionHealth,
		),
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/stats", Method: "GET", Description: "Get redacted context-compression statistics"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionStats,
		),
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/preview", Method: "POST", Description: "Preview context compression without persistence"},
			routePolicy{Permission: PermCompressionPreview, Sensitivity: SensitivityOperational, AuditAction: AuditActionCompressionPreview},
			(*ClassificationAPIServer).handleContextCompressionPreview,
			jsonBodyFor[contextCompressionPreviewRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiContextCompressionPath + "/recovery/invalidate", Method: "POST", Description: "Invalidate a trusted context-recovery request scope"},
			routePolicy{Permission: PermCompressionManage, Sensitivity: SensitivityMutation, AuditAction: AuditActionCompressionInvalidate},
			(*ClassificationAPIServer).handleContextCompressionRecoveryInvalidate,
			jsonBodyFor[contextCompressionRecoveryInvalidateRequest](),
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
		),
		managedRoute(
			EndpointMetadata{Path: apiRecipesPath + "/validate", Method: "POST", Description: "Validate a recipe mutation without writing or reloading config"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleValidateRecipe,
			strictJSONBodyFor[recipeMutationRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiRecipesPath + "/{name}", Method: "GET", Description: "Read one routing recipe and its entrypoints"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleGetRecipe,
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
					{Name: "expanded", In: "query", Description: "Return a self-contained JSON Schema for view=section instead of its compact field directory.", Schema: OpenAPISchema{Type: "boolean"}},
					{Name: "kind", In: "query", Description: "Surface kind required by view=surface.", Schema: OpenAPISchema{Type: "string", Enum: []string{"signal", "algorithm", "plugin", "projection"}}},
					{Name: "name", In: "query", Description: "Registered surface name required by view=surface.", Schema: OpenAPISchema{Type: "string"}},
				},
			},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleConfigSchema,
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigPath, Method: "GET", Description: "Get the current router config as JSON (secrets redacted without secret_view)"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleConfigGet,
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigValidatePath, Method: "POST", Description: "Validate and normalize a router config without writing it"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigValidate,
			strictJSONBodyFor[RouterConfigUpdateRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigPlanPath, Method: "POST", Description: "Plan an exact merge or replace mutation, including hot-reload compatibility, without writing it"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigPlan,
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
			strictJSONBodyFor[routerConfigRollbackRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigVersionsPath, Method: "GET", Description: "List available router config backup versions"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigVersions,
		),
		managedRoute(
			EndpointMetadata{Path: apiConfigHashPath, Method: "GET", Description: "Compare persisted source, generated runtime, and active router config hashes"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigHash,
		),
	}
}

func apiKnowledgeBaseRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath, Method: "GET", Description: "List configured knowledge bases"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListKnowledgeBases,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath, Method: "POST", Description: "Create a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseSave},
			(*ClassificationAPIServer).handleCreateKnowledgeBase,
			jsonBodyFor[knowledgeBaseUpsertRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}", Method: "GET", Description: "Read a knowledge base"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBase,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}/map/metadata", Method: "GET", Description: "Read generated knowledge-base map metadata"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBaseMapMetadata,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}/map/data.ndjson", Method: "GET", Description: "Stream generated knowledge-base map data as NDJSON"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBaseMapData,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}", Method: "PUT", Description: "Update a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseSave},
			(*ClassificationAPIServer).handleUpdateKnowledgeBase,
			jsonBodyFor[knowledgeBaseUpsertRequest](),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageKnowledgeBasesPath + "/{name}", Method: "DELETE", Description: "Delete a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseDel},
			(*ClassificationAPIServer).handleDeleteKnowledgeBase,
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
		),
	}
}

func apiVectorStoreRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath, Method: "POST", Description: "Create a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleCreateVectorStore,
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
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}", Method: "GET", Description: "Read a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetVectorStore,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}", Method: "POST", Description: "Update a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUpdateVectorStore,
			jsonBodyWithLimitFor[vectorstore.UpdateStoreRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}", Method: "DELETE", Description: "Delete a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteVectorStore,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/search", Method: "POST", Description: "Search a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSearchVectorStore,
			jsonBodyWithLimitFor[SearchRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/files", Method: "POST", Description: "Attach a file to a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleAttachFile,
			jsonBodyWithLimitFor[AttachFileRequest](maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/files", Method: "GET", Description: "List files attached to a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStoreFiles,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageVectorStoresPath + "/{id}/files/{file_id}", Method: "DELETE", Description: "Detach a file from a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDetachFile,
		),
	}
}

func apiFileRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath, Method: "POST", Description: "Upload a file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUploadFile,
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
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath + "/{id}", Method: "GET", Description: "Read uploaded-file metadata"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetFile,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath + "/{id}", Method: "DELETE", Description: "Delete an uploaded file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteFile,
		),
		managedRoute(
			EndpointMetadata{Path: apiStorageFilesPath + "/{id}/content", Method: "GET", Description: "Download uploaded-file content"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetFileContent,
		),
	}
}

func appendAPIRoutes(routes []apiRoute, groups ...[]apiRoute) []apiRoute {
	for _, group := range groups {
		routes = append(routes, group...)
	}
	return routes
}
