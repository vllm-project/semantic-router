//go:build !windows && cgo

package apiserver

const (
	apiRootPath = "/api/v1"

	apiConfigPath         = apiRootPath + "/config"
	apiConfigSchemaPath   = apiConfigPath + "/schema"
	apiConfigValidatePath = apiConfigPath + "/validate"
	apiConfigPlanPath     = apiConfigPath + "/plan"
	apiConfigVersionsPath = apiConfigPath + "/versions"
	apiConfigRollbackPath = apiConfigPath + "/rollback"
	apiConfigHashPath     = apiConfigPath + "/hash"
	apiRecipesPath        = apiConfigPath + "/recipes"

	apiRoutingPreviewPath = apiRootPath + "/routing/preview"

	apiInventoryPath             = apiRootPath + "/inventory"
	apiInventoryModelsPath       = apiInventoryPath + "/models"
	apiInventoryClassifierPath   = apiInventoryPath + "/classifier"
	apiInventoryEmbeddingModels  = apiInventoryPath + "/embedding-models"
	apiObservabilityPath         = apiRootPath + "/observability"
	apiObservabilityReplaysPath  = apiObservabilityPath + "/replays"
	apiObservabilityOutcomesPath = apiObservabilityPath + "/outcomes"
	apiClassificationMetricsPath = apiObservabilityPath + "/classification-metrics"
	apiDiagnosticsPath           = apiRootPath + "/diagnostics"
	apiStoragePath               = apiRootPath + "/storage"
	apiStorageKnowledgeBasesPath = apiStoragePath + "/knowledge-bases"
	apiStorageMemoriesPath       = apiStoragePath + "/memories"
	apiStorageVectorStoresPath   = apiStoragePath + "/vector-stores"
	apiStorageFilesPath          = apiStoragePath + "/files"
	apiResponseCachePath         = apiRootPath + "/response-cache"
	apiContextCompressionPath    = apiRootPath + "/context-compression"
)
