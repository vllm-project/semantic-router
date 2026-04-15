package routerruntime

import (
	"context"
	"fmt"
	"io"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/postgres"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// VectorStoreRuntime owns the vector-store services shared by the API server
// and request-time RAG retrieval.
type VectorStoreRuntime struct {
	FileStore      *vectorstore.FileStore
	Backend        vectorstore.VectorStoreBackend
	Manager        *vectorstore.Manager
	Pipeline       *vectorstore.IngestionPipeline
	Embedder       vectorstore.Embedder
	registryCloser io.Closer
}

func NewVectorStoreRuntime(cfg *config.RouterConfig) (*VectorStoreRuntime, error) {
	if cfg == nil {
		return nil, fmt.Errorf("vector store runtime requires config")
	}
	if err := cfg.VectorStore.Validate(); err != nil {
		return nil, err
	}
	cfg.VectorStore.ApplyDefaults()

	storeReg, fileReg, regCloser, err := buildMetadataRegistries(cfg)
	if err != nil {
		return nil, fmt.Errorf("failed to create metadata registry: %w", err)
	}
	emitMetadataStoreWarning(cfg)

	fileStore, err := vectorstore.NewFileStore(cfg.VectorStore.FileStorageDir, fileReg)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store file store: %w", err)
	}

	backend, err := vectorstore.NewBackend(cfg.VectorStore.BackendType, buildVectorStoreBackendConfigs(cfg))
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store backend: %w", err)
	}

	manager := vectorstore.NewManager(backend, storeReg, cfg.VectorStore.EmbeddingDimension, cfg.VectorStore.BackendType)

	ctx := context.Background()
	if err := manager.LoadFromRegistry(ctx); err != nil {
		logging.Warnf("Failed to load vector store registry on startup: %v", err)
	}
	if err := fileStore.LoadFromRegistry(ctx); err != nil {
		logging.Warnf("Failed to load file registry on startup: %v", err)
	}

	embedder := vectorstore.NewCandleEmbedder(cfg.VectorStore.EmbeddingModel, cfg.VectorStore.EmbeddingDimension)
	pipeline := vectorstore.NewIngestionPipeline(backend, fileStore, manager, embedder, vectorstore.PipelineConfig{
		Workers:   cfg.VectorStore.IngestionWorkers,
		QueueSize: 100,
	})
	pipeline.Start()

	return &VectorStoreRuntime{
		FileStore:      fileStore,
		Backend:        backend,
		Manager:        manager,
		Pipeline:       pipeline,
		Embedder:       embedder,
		registryCloser: regCloser,
	}, nil
}

func buildMetadataRegistries(cfg *config.RouterConfig) (vectorstore.StoreRegistry, vectorstore.FileRegistry, io.Closer, error) {
	switch cfg.VectorStore.MetadataStore {
	case "postgres":
		pgCfg := buildVectorStorePostgresConfig(cfg.VectorStore.MetadataPostgres)
		reg, err := vectorstore.NewPostgresMetadataRegistry(pgCfg)
		if err != nil {
			return nil, nil, nil, err
		}
		return reg, reg, reg, nil
	default:
		reg := vectorstore.NewMemoryMetadataRegistry()
		return reg, reg, nil, nil
	}
}

func buildVectorStorePostgresConfig(src *config.VectorStoreMetadataPostgresConfig) *postgres.Config {
	return &postgres.Config{
		Host:            src.Host,
		Port:            src.Port,
		Database:        src.Database,
		User:            src.User,
		Password:        src.Password,
		SSLMode:         src.SSLMode,
		MaxOpenConns:    src.MaxOpenConns,
		MaxIdleConns:    src.MaxIdleConns,
		ConnMaxLifetime: src.ConnMaxLifetime,
		TableName:       src.TableName,
	}
}

func emitMetadataStoreWarning(cfg *config.RouterConfig) {
	if cfg.VectorStore.MetadataStore != "memory" {
		return
	}
	bt := cfg.VectorStore.BackendType
	if bt == "milvus" || bt == "valkey" {
		logging.Warnf(
			"vector_store.metadata_store is 'memory' but backend_type is '%s'; "+
				"store metadata will be lost on restart — set metadata_store to 'postgres' for durability",
			bt,
		)
	}
}

func (r *VectorStoreRuntime) Shutdown() error {
	if r == nil {
		return nil
	}
	if r.Pipeline != nil {
		r.Pipeline.Stop()
	}
	if r.registryCloser != nil {
		if err := r.registryCloser.Close(); err != nil {
			logging.Warnf("Failed to close metadata registry: %v", err)
		}
	}
	if r.Backend != nil {
		return r.Backend.Close()
	}
	return nil
}

func (r *VectorStoreRuntime) LogInitialized(component string, cfg *config.RouterConfig) {
	if r == nil || cfg == nil {
		return
	}
	logging.ComponentEvent(component, "vector_store_initialized", map[string]interface{}{
		"backend": cfg.VectorStore.BackendType,
		"model":   cfg.VectorStore.EmbeddingModel,
		"dim":     cfg.VectorStore.EmbeddingDimension,
		"workers": cfg.VectorStore.IngestionWorkers,
	})
}

func buildVectorStoreBackendConfigs(cfg *config.RouterConfig) vectorstore.BackendConfigs {
	switch cfg.VectorStore.BackendType {
	case "memory":
		maxEntries := 100000
		if cfg.VectorStore.Memory != nil && cfg.VectorStore.Memory.MaxEntriesPerStore > 0 {
			maxEntries = cfg.VectorStore.Memory.MaxEntriesPerStore
		}
		return vectorstore.BackendConfigs{
			Memory: vectorstore.MemoryBackendConfig{MaxEntriesPerStore: maxEntries},
		}
	case "milvus":
		return vectorstore.BackendConfigs{
			Milvus: vectorstore.MilvusBackendConfig{
				Address: fmt.Sprintf("%s:%d", cfg.VectorStore.Milvus.Connection.Host, cfg.VectorStore.Milvus.Connection.Port),
			},
		}
	case "llama_stack":
		lsCfg := cfg.VectorStore.LlamaStack
		return vectorstore.BackendConfigs{
			LlamaStack: vectorstore.LlamaStackBackendConfig{
				Endpoint:              lsCfg.Endpoint,
				AuthToken:             lsCfg.AuthToken,
				EmbeddingModel:        lsCfg.EmbeddingModel,
				EmbeddingDimension:    cfg.VectorStore.EmbeddingDimension,
				RequestTimeoutSeconds: lsCfg.RequestTimeoutSeconds,
				SearchType:            lsCfg.SearchType,
			},
		}
	case "valkey":
		vCfg := cfg.VectorStore.Valkey
		return vectorstore.BackendConfigs{
			Valkey: vectorstore.ValkeyBackendConfig{
				Host:             vCfg.Host,
				Port:             vCfg.Port,
				Password:         vCfg.Password,
				Database:         vCfg.Database,
				CollectionPrefix: vCfg.CollectionPrefix,
				MetricType:       vCfg.MetricType,
				IndexM:           vCfg.IndexM,
				IndexEf:          vCfg.IndexEfConstruction,
				ConnectTimeout:   vCfg.ConnectTimeout,
			},
		}
	default:
		return vectorstore.BackendConfigs{}
	}
}
