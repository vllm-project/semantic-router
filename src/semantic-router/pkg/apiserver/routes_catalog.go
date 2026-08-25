//go:build !windows && cgo

package apiserver

func apiHealthRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/health", Method: "GET", Description: "Health check endpoint"},
			routePolicy{Permission: PermHealthRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleHealth,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/ready", Method: "GET", Description: "Readiness endpoint that turns green only after startup completes"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleReady,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/startup-status", Method: "GET", Description: "Detailed router startup and model-download status"},
			routePolicy{Permission: PermReadyRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleStartupStatus,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1", Method: "GET", Description: "API discovery and documentation"},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleAPIOverview,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/openapi.json", Method: "GET", Description: "OpenAPI 3.0 specification"},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleOpenAPISpec,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/docs", Method: "GET", Description: "Interactive Swagger UI documentation"},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleSwaggerUI,
		),
	}
}

func apiClassifyRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/intent", Method: "POST", Description: "Classify user queries into routing categories"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleIntentClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/pii", Method: "POST", Description: "Detect personally identifiable information in text"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handlePIIDetection,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/security", Method: "POST", Description: "Detect jailbreak attempts and security threats"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSecurityDetection,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/fact-check", Method: "POST", Description: "Classify if text needs fact-checking"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleFactCheckClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/user-feedback", Method: "POST", Description: "Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleUserFeedbackClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/combined", Method: "POST", Description: "Perform combined classification (intent, PII, and security)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleCombinedClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/classify/batch", Method: "POST", Description: "Batch classification with configurable task_type parameter"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/eval", Method: "POST", Description: "Evaluate all configured signals regardless of decision usage"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEvalClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/nli", Method: "POST", Description: "Natural language inference classification for premise and hypothesis pairs"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleNLIClassification,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/embeddings", Method: "POST", Description: "Generate text and image embeddings"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEmbeddings,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/similarity", Method: "POST", Description: "Calculate pairwise text similarity"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSimilarity,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/similarity/batch", Method: "POST", Description: "Calculate batch text-similarity matches"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchSimilarity,
			jsonBody(),
		),
	}
}

func apiInfoRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/info/models", Method: "GET", Description: "Get information about loaded models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleModelsInfo,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/info/classifier", Method: "GET", Description: "Get classifier information and status (secrets redacted without secret_view)"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleClassifierInfo,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/embeddings/models", Method: "GET", Description: "Get information about loaded embedding models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleEmbeddingModelsInfo,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/models", Method: "GET", Description: "OpenAI-compatible model listing"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleOpenAIModels,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/metrics/classification", Method: "GET", Description: "Get classification metrics and statistics"},
			routePolicy{Permission: PermMetricsRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleClassificationMetrics,
		),
	}
}

func apiResponseCacheRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/capabilities", Method: "GET", Description: "Get response-cache backend capabilities"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheCapabilities,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/health", Method: "GET", Description: "Check response-cache backend health"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheHealth,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/stats", Method: "GET", Description: "Get redacted response-cache statistics"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheStats,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/audit", Method: "GET", Description: "Get redacted response-cache mutation audit entries"},
			routePolicy{Permission: PermCacheRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheAudit,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/test", Method: "POST", Description: "Validate and probe a response-cache candidate configuration"},
			routePolicy{Permission: PermCacheManage, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleResponseCacheTest,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/invalidate", Method: "POST", Description: "Dry-run or invalidate a scoped response-cache partition"},
			routePolicy{Permission: PermCacheInvalidate, Sensitivity: SensitivityMutation, AuditAction: AuditActionCacheInvalidate},
			(*ClassificationAPIServer).handleResponseCacheInvalidate,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/response-cache/flush", Method: "POST", Description: "Advance a scoped or global response-cache epoch"},
			routePolicy{Permission: PermCacheManage, Sensitivity: SensitivityMutation, AuditAction: AuditActionCacheFlush},
			(*ClassificationAPIServer).handleResponseCacheFlush,
			jsonBody(),
		),
	}
}

func apiContextCompressionRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/context-compression/capabilities", Method: "GET", Description: "Get context-compression capabilities"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionCapabilities,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/context-compression/health", Method: "GET", Description: "Check context-compression runtime health"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionHealth,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/context-compression/stats", Method: "GET", Description: "Get redacted context-compression statistics"},
			routePolicy{Permission: PermCompressionRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleContextCompressionStats,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/context-compression/preview", Method: "POST", Description: "Preview context compression without persistence"},
			routePolicy{Permission: PermCompressionPreview, Sensitivity: SensitivityOperational, AuditAction: AuditActionCompressionPreview},
			(*ClassificationAPIServer).handleContextCompressionPreview,
			jsonBody(),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/api/v1/context-compression/recovery/invalidate", Method: "POST", Description: "Invalidate a trusted context-recovery request scope"},
			routePolicy{Permission: PermCompressionManage, Sensitivity: SensitivityMutation, AuditAction: AuditActionCompressionInvalidate},
			(*ClassificationAPIServer).handleContextCompressionRecoveryInvalidate,
			jsonBody(),
		),
	}
}

func apiMemoryRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/v1/memory", Method: "GET", Description: "List long-term memories"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListMemories,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/memory", Method: "DELETE", Description: "Delete memories by scope"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionMemoryDelete},
			(*ClassificationAPIServer).handleDeleteMemoriesByScope,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/memory/{id}", Method: "GET", Description: "Read one long-term memory"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetMemory,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/memory/{id}", Method: "DELETE", Description: "Delete one long-term memory"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionMemoryDelete},
			(*ClassificationAPIServer).handleDeleteMemory,
		),
	}
}

func apiVectorStoreRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores", Method: "POST", Description: "Create a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleCreateVectorStore,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores", Method: "GET", Description: "List vector stores"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStores,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "GET", Description: "Read a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetVectorStore,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "POST", Description: "Update a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUpdateVectorStore,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "DELETE", Description: "Delete a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteVectorStore,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/search", Method: "POST", Description: "Search a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSearchVectorStore,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/files", Method: "POST", Description: "Attach a file to a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleAttachFile,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/files", Method: "GET", Description: "List files attached to a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStoreFiles,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/files/{file_id}", Method: "DELETE", Description: "Detach a file from a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDetachFile,
		),
	}
}

func apiFileRoutes() []apiRoute {
	return []apiRoute{
		authorizedRoute(
			EndpointMetadata{Path: "/v1/files", Method: "POST", Description: "Upload a file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUploadFile,
			multipartBody(maxUploadSize, "Multipart upload with a file field and optional purpose field."),
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/files", Method: "GET", Description: "List uploaded files"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListFiles,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/files/{id}", Method: "GET", Description: "Read uploaded-file metadata"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetFile,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/files/{id}", Method: "DELETE", Description: "Delete an uploaded file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteFile,
		),
		authorizedRoute(
			EndpointMetadata{Path: "/v1/files/{id}/content", Method: "GET", Description: "Download uploaded-file content"},
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
