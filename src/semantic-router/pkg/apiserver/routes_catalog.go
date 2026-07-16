//go:build !windows && cgo

package apiserver

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
			EndpointMetadata{Path: "/api/v1", Method: "GET", Description: "API discovery and documentation"},
			routePolicy{Permission: PermDocsRead, Sensitivity: SensitivityPublic},
			(*ClassificationAPIServer).handleAPIOverview,
		),
		managedRoute(
			EndpointMetadata{Path: "/openapi.json", Method: "GET", Description: "OpenAPI 3.0 specification"},
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
			EndpointMetadata{Path: "/api/v1/classify/intent", Method: "POST", Description: "Classify user queries into routing categories"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleIntentClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/classify/pii", Method: "POST", Description: "Detect personally identifiable information in text"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handlePIIDetection,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/classify/security", Method: "POST", Description: "Detect jailbreak attempts and security threats"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSecurityDetection,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/classify/fact-check", Method: "POST", Description: "Classify if text needs fact-checking"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleFactCheckClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/classify/user-feedback", Method: "POST", Description: "Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleUserFeedbackClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/classify/combined", Method: "POST", Description: "Perform combined classification (intent, PII, and security)"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleCombinedClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/classify/batch", Method: "POST", Description: "Batch classification with configurable task_type parameter"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/eval", Method: "POST", Description: "Evaluate all configured signals regardless of decision usage"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEvalClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/nli", Method: "POST", Description: "Natural language inference classification for premise and hypothesis pairs"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleNLIClassification,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/embeddings", Method: "POST", Description: "Generate text and image embeddings"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleEmbeddings,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/similarity", Method: "POST", Description: "Calculate pairwise text similarity"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSimilarity,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/similarity/batch", Method: "POST", Description: "Calculate batch text-similarity matches"},
			routePolicy{Permission: PermClassifyInvoke, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleBatchSimilarity,
			jsonBody(),
		),
	}
}

func apiInfoRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/info/models", Method: "GET", Description: "Get information about loaded models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleModelsInfo,
		),
		managedRoute(
			EndpointMetadata{Path: "/info/classifier", Method: "GET", Description: "Get classifier information and status"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleClassifierInfo,
		),
		managedRoute(
			EndpointMetadata{Path: "/api/v1/embeddings/models", Method: "GET", Description: "Get information about loaded embedding models"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleEmbeddingModelsInfo,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/models", Method: "GET", Description: "OpenAI-compatible model listing"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleOpenAIModels,
		),
		managedRoute(
			EndpointMetadata{Path: "/metrics/classification", Method: "GET", Description: "Get classification metrics and statistics"},
			routePolicy{Permission: PermMetricsRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleClassificationMetrics,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/router/outcomes", Method: "POST", Description: "Submit Router Learning outcome feedback linked to a replay record"},
			routePolicy{Permission: PermLearningIngest, Sensitivity: SensitivityMutation, AuditAction: AuditActionOutcomeIngest},
			(*ClassificationAPIServer).handleRouterOutcome,
			jsonBody(),
		),
	}
}

func apiConfigRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/config/kbs", Method: "GET", Description: "List configured knowledge bases"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListKnowledgeBases,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/kbs", Method: "POST", Description: "Create a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseSave},
			(*ClassificationAPIServer).handleCreateKnowledgeBase,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/config/kbs/{name}", Method: "GET", Description: "Read a knowledge base"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBase,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/kbs/{name}/map/metadata", Method: "GET", Description: "Read generated knowledge-base map metadata"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBaseMapMetadata,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/kbs/{name}/map/data.ndjson", Method: "GET", Description: "Stream generated knowledge-base map data as NDJSON"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetKnowledgeBaseMapData,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/kbs/{name}", Method: "PUT", Description: "Update a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseSave},
			(*ClassificationAPIServer).handleUpdateKnowledgeBase,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/config/kbs/{name}", Method: "DELETE", Description: "Delete a managed knowledge base"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionKnowledgeBaseDel},
			(*ClassificationAPIServer).handleDeleteKnowledgeBase,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/router", Method: "GET", Description: "Get the current router config as JSON"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivitySecretView},
			(*ClassificationAPIServer).handleConfigGet,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/router", Method: "PATCH", Description: "Merge a router config update (validates, backs up, writes, triggers hot-reload)"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionConfigPatch},
			(*ClassificationAPIServer).handleConfigPatch,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/config/router", Method: "PUT", Description: "Replace the router config (validates, backs up, writes, triggers hot-reload)"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionConfigPut},
			(*ClassificationAPIServer).handleConfigPut,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/config/router/rollback", Method: "POST", Description: "Rollback to a previous router config version"},
			routePolicy{Permission: PermConfigWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionConfigRollback},
			(*ClassificationAPIServer).handleConfigRollback,
			jsonBody(),
		),
		managedRoute(
			EndpointMetadata{Path: "/config/router/versions", Method: "GET", Description: "List available router config backup versions"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigVersions,
		),
		managedRoute(
			EndpointMetadata{Path: "/config/hash", Method: "GET", Description: "Get the active router config hash"},
			routePolicy{Permission: PermConfigRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleConfigHash,
		),
	}
}

func apiMemoryRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/v1/memory", Method: "GET", Description: "List long-term memories"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListMemories,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/memory", Method: "DELETE", Description: "Delete memories by scope"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionMemoryDelete},
			(*ClassificationAPIServer).handleDeleteMemoriesByScope,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/memory/{id}", Method: "GET", Description: "Read one long-term memory"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetMemory,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/memory/{id}", Method: "DELETE", Description: "Delete one long-term memory"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionMemoryDelete},
			(*ClassificationAPIServer).handleDeleteMemory,
		),
	}
}

func apiVectorStoreRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores", Method: "POST", Description: "Create a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleCreateVectorStore,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores", Method: "GET", Description: "List vector stores"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStores,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "GET", Description: "Read a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetVectorStore,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "POST", Description: "Update a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUpdateVectorStore,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "DELETE", Description: "Delete a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteVectorStore,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/search", Method: "POST", Description: "Search a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityOperational},
			(*ClassificationAPIServer).handleSearchVectorStore,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/files", Method: "POST", Description: "Attach a file to a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleAttachFile,
			jsonBodyWithLimit(maxVectorStoreJSONBodySize),
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/files", Method: "GET", Description: "List files attached to a vector store"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListVectorStoreFiles,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/vector_stores/{id}/files/{file_id}", Method: "DELETE", Description: "Detach a file from a vector store"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDetachFile,
		),
	}
}

func apiFileRoutes() []apiRoute {
	return []apiRoute{
		managedRoute(
			EndpointMetadata{Path: "/v1/files", Method: "POST", Description: "Upload a file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleUploadFile,
			multipartBody(maxUploadSize, "Multipart upload with a file field and optional purpose field."),
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/files", Method: "GET", Description: "List uploaded files"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleListFiles,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/files/{id}", Method: "GET", Description: "Read uploaded-file metadata"},
			routePolicy{Permission: PermDataRead, Sensitivity: SensitivityConfig},
			(*ClassificationAPIServer).handleGetFile,
		),
		managedRoute(
			EndpointMetadata{Path: "/v1/files/{id}", Method: "DELETE", Description: "Delete an uploaded file"},
			routePolicy{Permission: PermDataWrite, Sensitivity: SensitivityMutation, AuditAction: AuditActionDataWrite},
			(*ClassificationAPIServer).handleDeleteFile,
		),
		managedRoute(
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
