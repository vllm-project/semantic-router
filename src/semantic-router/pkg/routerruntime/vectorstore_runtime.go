package routerruntime

import (
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// VectorStoreRuntime owns the vector-store services shared by the API server
// and request-time RAG retrieval.
type VectorStoreRuntime struct {
	FileStore *vectorstore.FileStore
	Backend   vectorstore.VectorStoreBackend
	Manager   *vectorstore.Manager
	Pipeline  *vectorstore.IngestionPipeline
	Embedder  vectorstore.Embedder
}

func NewVectorStoreRuntime(cfg *config.RouterConfig) (*VectorStoreRuntime, error) {
	if cfg == nil {
		return nil, fmt.Errorf("vector store runtime requires config")
	}
	if err := cfg.VectorStore.Validate(); err != nil {
		return nil, err
	}
	cfg.VectorStore.ApplyDefaults()

	fileStore, err := vectorstore.NewFileStore(cfg.VectorStore.FileStorageDir)
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store file store: %w", err)
	}

	backend, err := vectorstore.NewBackend(cfg.VectorStore.BackendType, buildVectorStoreBackendConfigs(cfg))
	if err != nil {
		return nil, fmt.Errorf("failed to create vector store backend: %w", err)
	}

	manager := vectorstore.NewManager(backend, cfg.VectorStore.EmbeddingDimension, cfg.VectorStore.BackendType)
	embedder := vectorstore.NewCandleEmbedder(cfg.VectorStore.EmbeddingModel, cfg.VectorStore.EmbeddingDimension)
	pipeline := vectorstore.NewIngestionPipeline(backend, fileStore, manager, embedder, vectorstore.PipelineConfig{
		Workers:   cfg.VectorStore.IngestionWorkers,
		QueueSize: 100,
	})
	pipeline.Start()

	return &VectorStoreRuntime{
		FileStore: fileStore,
		Backend:   backend,
		Manager:   manager,
		Pipeline:  pipeline,
		Embedder:  embedder,
	}, nil
}

func (r *VectorStoreRuntime) Shutdown() error {
	if r == nil {
		return nil
	}
	if r.Pipeline != nil {
		r.Pipeline.Stop()
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
