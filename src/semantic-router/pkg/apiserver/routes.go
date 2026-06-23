//go:build !windows && cgo

package apiserver

import (
	"fmt"
	"net/http"
)

type apiRouteHandler func(*ClassificationAPIServer, http.ResponseWriter, *http.Request)

type requestBodyKind string

const (
	requestBodyNone      requestBodyKind = ""
	requestBodyJSON      requestBodyKind = "application/json"
	requestBodyMultipart requestBodyKind = "multipart/form-data"
)

type apiRequestBody struct {
	Kind        requestBodyKind
	Required    bool
	LimitBytes  int64
	Description string
}

type apiRoute struct {
	EndpointMetadata
	Handler     apiRouteHandler
	RequestBody apiRequestBody
}

func (r apiRoute) pattern() string {
	return fmt.Sprintf("%s %s", r.Method, r.Path)
}

func (r apiRoute) bind(s *ClassificationAPIServer) http.HandlerFunc {
	return func(w http.ResponseWriter, req *http.Request) {
		r.Handler(s, w, req)
	}
}

func jsonBody() apiRequestBody {
	return jsonBodyWithLimit(defaultJSONRequestBodyLimit)
}

func jsonBodyWithLimit(limit int64) apiRequestBody {
	return apiRequestBody{
		Kind:       requestBodyJSON,
		Required:   true,
		LimitBytes: limit,
	}
}

func multipartBody(limit int64, description string) apiRequestBody {
	return apiRequestBody{
		Kind:        requestBodyMultipart,
		Required:    true,
		LimitBytes:  limit,
		Description: description,
	}
}

func apiEndpointMetadata() []EndpointMetadata {
	routes := apiRoutes()
	metadata := make([]EndpointMetadata, 0, len(routes))
	for _, route := range routes {
		metadata = append(metadata, route.EndpointMetadata)
	}
	return metadata
}

func apiRoutes() []apiRoute {
	return []apiRoute{
		{EndpointMetadata: EndpointMetadata{Path: "/health", Method: "GET", Description: "Health check endpoint"}, Handler: (*ClassificationAPIServer).handleHealth},
		{EndpointMetadata: EndpointMetadata{Path: "/ready", Method: "GET", Description: "Readiness endpoint that turns green only after startup completes"}, Handler: (*ClassificationAPIServer).handleReady},
		{EndpointMetadata: EndpointMetadata{Path: "/startup-status", Method: "GET", Description: "Detailed router startup and model-download status"}, Handler: (*ClassificationAPIServer).handleStartupStatus},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1", Method: "GET", Description: "API discovery and documentation"}, Handler: (*ClassificationAPIServer).handleAPIOverview},
		{EndpointMetadata: EndpointMetadata{Path: "/openapi.json", Method: "GET", Description: "OpenAPI 3.0 specification"}, Handler: (*ClassificationAPIServer).handleOpenAPISpec},
		{EndpointMetadata: EndpointMetadata{Path: "/docs", Method: "GET", Description: "Interactive Swagger UI documentation"}, Handler: (*ClassificationAPIServer).handleSwaggerUI},

		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/intent", Method: "POST", Description: "Classify user queries into routing categories"}, Handler: (*ClassificationAPIServer).handleIntentClassification, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/pii", Method: "POST", Description: "Detect personally identifiable information in text"}, Handler: (*ClassificationAPIServer).handlePIIDetection, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/security", Method: "POST", Description: "Detect jailbreak attempts and security threats"}, Handler: (*ClassificationAPIServer).handleSecurityDetection, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/fact-check", Method: "POST", Description: "Classify if text needs fact-checking"}, Handler: (*ClassificationAPIServer).handleFactCheckClassification, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/user-feedback", Method: "POST", Description: "Classify user feedback type (satisfied, need_clarification, wrong_answer, want_different)"}, Handler: (*ClassificationAPIServer).handleUserFeedbackClassification, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/combined", Method: "POST", Description: "Perform combined classification (intent, PII, and security)"}, Handler: (*ClassificationAPIServer).handleCombinedClassification, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/classify/batch", Method: "POST", Description: "Batch classification with configurable task_type parameter"}, Handler: (*ClassificationAPIServer).handleBatchClassification, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/eval", Method: "POST", Description: "Evaluate all configured signals regardless of decision usage"}, Handler: (*ClassificationAPIServer).handleEvalClassification, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/nli", Method: "POST", Description: "Natural language inference classification for premise and hypothesis pairs"}, Handler: (*ClassificationAPIServer).handleNLIClassification, RequestBody: jsonBody()},

		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/embeddings", Method: "POST", Description: "Generate text embeddings"}, Handler: (*ClassificationAPIServer).handleEmbeddings, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/similarity", Method: "POST", Description: "Calculate pairwise text similarity"}, Handler: (*ClassificationAPIServer).handleSimilarity, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/similarity/batch", Method: "POST", Description: "Calculate batch text-similarity matches"}, Handler: (*ClassificationAPIServer).handleBatchSimilarity, RequestBody: jsonBody()},

		{EndpointMetadata: EndpointMetadata{Path: "/info/models", Method: "GET", Description: "Get information about loaded models"}, Handler: (*ClassificationAPIServer).handleModelsInfo},
		{EndpointMetadata: EndpointMetadata{Path: "/info/classifier", Method: "GET", Description: "Get classifier information and status"}, Handler: (*ClassificationAPIServer).handleClassifierInfo},
		{EndpointMetadata: EndpointMetadata{Path: "/api/v1/embeddings/models", Method: "GET", Description: "Get information about loaded embedding models"}, Handler: (*ClassificationAPIServer).handleEmbeddingModelsInfo},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/models", Method: "GET", Description: "OpenAI-compatible model listing"}, Handler: (*ClassificationAPIServer).handleOpenAIModels},
		{EndpointMetadata: EndpointMetadata{Path: "/metrics/classification", Method: "GET", Description: "Get classification metrics and statistics"}, Handler: (*ClassificationAPIServer).handleClassificationMetrics},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/router/outcomes", Method: "POST", Description: "Submit Router Learning outcome feedback linked to a replay record"}, Handler: (*ClassificationAPIServer).handleRouterOutcome, RequestBody: jsonBody()},

		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs", Method: "GET", Description: "List configured knowledge bases"}, Handler: (*ClassificationAPIServer).handleListKnowledgeBases},
		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs", Method: "POST", Description: "Create a managed knowledge base"}, Handler: (*ClassificationAPIServer).handleCreateKnowledgeBase, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs/{name}", Method: "GET", Description: "Read a knowledge base"}, Handler: (*ClassificationAPIServer).handleGetKnowledgeBase},
		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs/{name}/map/metadata", Method: "GET", Description: "Read generated knowledge-base map metadata"}, Handler: (*ClassificationAPIServer).handleGetKnowledgeBaseMapMetadata},
		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs/{name}/map/data.ndjson", Method: "GET", Description: "Stream generated knowledge-base map data as NDJSON"}, Handler: (*ClassificationAPIServer).handleGetKnowledgeBaseMapData},
		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs/{name}", Method: "PUT", Description: "Update a managed knowledge base"}, Handler: (*ClassificationAPIServer).handleUpdateKnowledgeBase, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/config/kbs/{name}", Method: "DELETE", Description: "Delete a managed knowledge base"}, Handler: (*ClassificationAPIServer).handleDeleteKnowledgeBase},
		{EndpointMetadata: EndpointMetadata{Path: "/config/router", Method: "GET", Description: "Get the current router config as JSON"}, Handler: (*ClassificationAPIServer).handleConfigGet},
		{EndpointMetadata: EndpointMetadata{Path: "/config/router", Method: "PATCH", Description: "Merge a router config update (validates, backs up, writes, triggers hot-reload)"}, Handler: (*ClassificationAPIServer).handleConfigPatch, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/config/router", Method: "PUT", Description: "Replace the router config (validates, backs up, writes, triggers hot-reload)"}, Handler: (*ClassificationAPIServer).handleConfigPut, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/config/router/rollback", Method: "POST", Description: "Rollback to a previous router config version"}, Handler: (*ClassificationAPIServer).handleConfigRollback, RequestBody: jsonBody()},
		{EndpointMetadata: EndpointMetadata{Path: "/config/router/versions", Method: "GET", Description: "List available router config backup versions"}, Handler: (*ClassificationAPIServer).handleConfigVersions},
		{EndpointMetadata: EndpointMetadata{Path: "/config/hash", Method: "GET", Description: "Get the active router config hash"}, Handler: (*ClassificationAPIServer).handleConfigHash},

		{EndpointMetadata: EndpointMetadata{Path: "/v1/memory", Method: "GET", Description: "List long-term memories"}, Handler: (*ClassificationAPIServer).handleListMemories},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/memory", Method: "DELETE", Description: "Delete memories by scope"}, Handler: (*ClassificationAPIServer).handleDeleteMemoriesByScope},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/memory/{id}", Method: "GET", Description: "Read one long-term memory"}, Handler: (*ClassificationAPIServer).handleGetMemory},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/memory/{id}", Method: "DELETE", Description: "Delete one long-term memory"}, Handler: (*ClassificationAPIServer).handleDeleteMemory},

		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores", Method: "POST", Description: "Create a vector store"}, Handler: (*ClassificationAPIServer).handleCreateVectorStore, RequestBody: jsonBodyWithLimit(maxVectorStoreJSONBodySize)},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores", Method: "GET", Description: "List vector stores"}, Handler: (*ClassificationAPIServer).handleListVectorStores},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "GET", Description: "Read a vector store"}, Handler: (*ClassificationAPIServer).handleGetVectorStore},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "POST", Description: "Update a vector store"}, Handler: (*ClassificationAPIServer).handleUpdateVectorStore, RequestBody: jsonBodyWithLimit(maxVectorStoreJSONBodySize)},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}", Method: "DELETE", Description: "Delete a vector store"}, Handler: (*ClassificationAPIServer).handleDeleteVectorStore},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}/search", Method: "POST", Description: "Search a vector store"}, Handler: (*ClassificationAPIServer).handleSearchVectorStore, RequestBody: jsonBodyWithLimit(maxVectorStoreJSONBodySize)},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}/files", Method: "POST", Description: "Attach a file to a vector store"}, Handler: (*ClassificationAPIServer).handleAttachFile, RequestBody: jsonBodyWithLimit(maxVectorStoreJSONBodySize)},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}/files", Method: "GET", Description: "List files attached to a vector store"}, Handler: (*ClassificationAPIServer).handleListVectorStoreFiles},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/vector_stores/{id}/files/{file_id}", Method: "DELETE", Description: "Detach a file from a vector store"}, Handler: (*ClassificationAPIServer).handleDetachFile},

		{EndpointMetadata: EndpointMetadata{Path: "/v1/files", Method: "POST", Description: "Upload a file"}, Handler: (*ClassificationAPIServer).handleUploadFile, RequestBody: multipartBody(maxUploadSize, "Multipart upload with a file field and optional purpose field.")},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/files", Method: "GET", Description: "List uploaded files"}, Handler: (*ClassificationAPIServer).handleListFiles},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/files/{id}", Method: "GET", Description: "Read uploaded-file metadata"}, Handler: (*ClassificationAPIServer).handleGetFile},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/files/{id}", Method: "DELETE", Description: "Delete an uploaded file"}, Handler: (*ClassificationAPIServer).handleDeleteFile},
		{EndpointMetadata: EndpointMetadata{Path: "/v1/files/{id}/content", Method: "GET", Description: "Download uploaded-file content"}, Handler: (*ClassificationAPIServer).handleGetFileContent},
	}
}
