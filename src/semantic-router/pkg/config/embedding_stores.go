package config

import (
	"fmt"
	"maps"
	"slices"
)

// The backends an embedding provider reports.
const (
	EmbeddingBackendModelRuntime     = ModelRuntimeProvider
	EmbeddingBackendOpenAICompatible = "openai_compatible"
)

// EmbeddingBinding names the deployment that embeds the vectors of a store or
// of an external collection, and their dimension. A store or collection
// without one embeds with global.model_catalog.embeddings.default.
type EmbeddingBinding struct {
	Deployment string `json:"deployment,omitempty" yaml:"deployment,omitempty"`
	// Dimension is the width of the vectors: the deployment's own, or a
	// narrower one its model serves. Zero keeps the store's width.
	Dimension int `json:"dimension,omitempty" yaml:"dimension,omitempty"`
}

func (b *EmbeddingBinding) width() int {
	if b == nil {
		return 0
	}
	return b.Dimension
}

// EmbeddingDeploymentOf is the deployment a binding names, else
// embeddings.default.
func (c *RouterConfig) EmbeddingDeploymentOf(binding *EmbeddingBinding) string {
	if binding != nil && binding.Deployment != "" {
		return binding.Deployment
	}
	return c.EmbeddingDeployment
}

// Stores that named no model were indexed with these views of Vela
// Embedding: memory at 256 dimensions, and the in-memory response cache at
// 256 dimensions from layer 6. A store that sets no dimension keeps them on a
// Vela Embedding deployment, so its vectors stay valid; any other model is
// read at its full output.
const (
	VelaEmbeddingMemoryDimension = 256
	velaEmbeddingCacheDimension  = 256
	velaEmbeddingCacheLayer      = 6
)

// servesVelaEmbedding reports whether a deployment runs Vela Embedding on the
// model runtime.
func (c *RouterConfig) servesVelaEmbedding(name string) bool {
	deployment, ok := c.ModelDeployments[name]
	if !ok || !deployment.IsModelRuntime() {
		return false
	}
	spec := registeredModel(deployment.Artifact)
	return spec != nil && spec.LocalPath == VelaEmbeddingModel
}

// ResponseCacheEmbedding is the deployment the response cache embeds with and
// the view it reads; zero dimension and layer read the full output, or the
// width a persistent backend's vector field sets.
func (c *RouterConfig) ResponseCacheEmbedding() (name string, dimension, layer int) {
	name = c.EmbeddingDeploymentOf(c.SemanticCache.Embedding)
	if width := c.SemanticCache.Embedding.width(); width > 0 {
		return name, width, 0
	}
	if c.SemanticCache.inMemory() && c.servesVelaEmbedding(name) {
		return name, velaEmbeddingCacheDimension, velaEmbeddingCacheLayer
	}
	return name, 0, 0
}

func (c ResponseCacheStoreConfig) inMemory() bool {
	return c.BackendType == "" || c.BackendType == "memory"
}

// vectorWidth is the width the vector field of a persistent backend sets.
func (c ResponseCacheStoreConfig) vectorWidth() (string, int) {
	switch c.BackendType {
	case "redis":
		if c.Redis != nil {
			return "redis.index.vector_field.dimension", c.Redis.Index.VectorField.Dimension
		}
	case "valkey":
		if c.Valkey != nil {
			return "valkey.index.vector_field.dimension", c.Valkey.Index.VectorField.Dimension
		}
	case "milvus", "hybrid":
		if c.Milvus != nil {
			return "milvus.collection.vector_field.dimension", c.Milvus.Collection.VectorField.Dimension
		}
	}
	return "", 0
}

// MemoryEmbedding is the deployment memory embeds with and the width of its
// vectors: its backend's dimension, else the binding's, else Vela
// Embedding's memory view; zero is the model's full output.
func (c *RouterConfig) MemoryEmbedding() (name string, dimension int) {
	name = c.EmbeddingDeploymentOf(c.Memory.Embedding)
	if _, width := c.Memory.backendWidth(); width > 0 {
		return name, width
	}
	if width := c.Memory.Embedding.width(); width > 0 {
		return name, width
	}
	if c.servesVelaEmbedding(name) {
		return name, VelaEmbeddingMemoryDimension
	}
	return name, 0
}

// backendWidth is the vector dimension the memory backend in use sets.
func (m MemoryConfig) backendWidth() (string, int) {
	switch m.Backend {
	case "", "milvus":
		return "milvus.dimension", m.Milvus.Dimension
	case "valkey":
		if m.Valkey != nil {
			return "valkey.dimension", m.Valkey.Dimension
		}
	case "qdrant":
		if m.Qdrant != nil {
			return "qdrant.dimension", m.Qdrant.Dimension
		}
	}
	return "", 0
}

// VectorStoreEmbedding is the deployment the vector store embeds documents
// and queries with, and the width of its vectors; zero is the model's full
// output.
func (c *RouterConfig) VectorStoreEmbedding() (string, int) {
	if c.VectorStore == nil {
		return c.EmbeddingDeployment, 0
	}
	return c.EmbeddingDeploymentOf(c.VectorStore.Embedding), c.VectorStore.Embedding.width()
}

// RAGQueryEmbeddings are the bindings a RAG source embeds its queries with,
// by their path under the source: its backend's, for a backend that searches
// a collection with a query vector the router computes, and a hybrid
// backend's two. A nil binding embeds with embeddings.default.
func RAGQueryEmbeddings(source *ContextSource) map[string]*EmbeddingBinding {
	bindings := map[string]*EmbeddingBinding{}
	collectRAGQueryEmbeddings(source, "backend_config", bindings)
	return bindings
}

func collectRAGQueryEmbeddings(source *ContextSource, path string, into map[string]*EmbeddingBinding) {
	if source == nil {
		return
	}
	switch source.Backend {
	case "milvus":
		if backend, err := source.MilvusBackendConfig(); err == nil {
			into[path+".embedding"] = backend.Embedding
		}
	case "qdrant":
		if backend, err := source.QdrantBackendConfig(); err == nil {
			into[path+".embedding"] = backend.Embedding
		}
	case "external_api":
		if backend, err := source.ExternalAPIBackendConfig(); err == nil && backend.EmbedsQuery() {
			into[path+".embedding"] = backend.Embedding
		}
	case "hybrid":
		if hybrid, err := source.HybridBackendConfig(); err == nil {
			collectRAGQueryEmbeddings(&ContextSource{Backend: hybrid.Primary, BackendConfig: hybrid.PrimaryConfig}, path+".primary_config", into)
			collectRAGQueryEmbeddings(&ContextSource{Backend: hybrid.Fallback, BackendConfig: hybrid.FallbackConfig}, path+".fallback_config", into)
		}
	}
}

// The embedding consumers: a RAG source's query embedding, the embeddings
// API (text, and the media of the multimodal deployment), the response
// cache, memory and vector store ingestion.
const (
	EmbeddingConsumerRAG           = "rag"
	EmbeddingConsumerAPI           = "api"
	EmbeddingConsumerMultimodalAPI = "multimodal_api"
	EmbeddingConsumerResponseCache = "response_cache"
	EmbeddingConsumerMemory        = "memory"
	EmbeddingConsumerVectorStore   = "vector_store"
)

// EmbeddingRequirement is one consumer of an embedding deployment and the
// view it reads; zero dimension and layer read the full output.
type EmbeddingRequirement struct {
	Deployment, Consumer string
	Dimension, Layer     int
}

// EmbeddingRequirements lists the embedding consumers one prepared scope
// runs: its RAG sources, and with sharedServices the embeddings API, the
// response cache, memory and vector store ingestion.
func EmbeddingRequirements(cfg *RouterConfig, sharedServices bool) []EmbeddingRequirement {
	cacheNeeded := sharedServices && cfg.NeedsSemanticResponseCache()
	cfg = cfg.ModelConsumerScope()
	var result []EmbeddingRequirement
	for i := range cfg.Decisions {
		bindings := RAGQueryEmbeddings(cfg.ContextPlugin(&cfg.Decisions[i]).Source(ContextSourceRAG))
		for _, path := range slices.Sorted(maps.Keys(bindings)) {
			binding := bindings[path]
			result = append(result, EmbeddingRequirement{Deployment: cfg.EmbeddingDeploymentOf(binding), Consumer: EmbeddingConsumerRAG, Dimension: binding.width()})
		}
	}
	if !sharedServices {
		return result
	}
	if cfg.API.Embeddings.Enabled {
		result = append(result, EmbeddingRequirement{Deployment: cfg.EmbeddingDeployment, Consumer: EmbeddingConsumerAPI})
		if cfg.MultimodalEmbeddingDeployment != "" {
			result = append(result, EmbeddingRequirement{Deployment: cfg.MultimodalEmbeddingDeployment, Consumer: EmbeddingConsumerMultimodalAPI})
		}
	}
	if cacheNeeded {
		name, dimension, layer := cfg.ResponseCacheEmbedding()
		result = append(result, EmbeddingRequirement{Deployment: name, Consumer: EmbeddingConsumerResponseCache, Dimension: dimension, Layer: layer})
	}
	if MemoryConfigured(cfg) {
		name, dimension := cfg.MemoryEmbedding()
		result = append(result, EmbeddingRequirement{Deployment: name, Consumer: EmbeddingConsumerMemory, Dimension: dimension})
	}
	if cfg.VectorStore != nil && cfg.VectorStore.Enabled {
		name, dimension := cfg.VectorStoreEmbedding()
		result = append(result, EmbeddingRequirement{Deployment: name, Consumer: EmbeddingConsumerVectorStore, Dimension: dimension})
	}
	return result
}

// MemoryConfigured reports whether memory runs: the store is enabled, or a
// decision has a memory plugin.
func MemoryConfigured(cfg *RouterConfig) bool {
	if cfg.Memory.Enabled {
		return true
	}
	for _, decision := range cfg.Decisions {
		if decision.HasPlugin("memory") {
			return true
		}
	}
	return false
}

// embeddingDeploymentsInUse lists the deployments the embedding consumers of
// the global services and of every routing scope run.
func (c *RouterConfig) embeddingDeploymentsInUse() map[string]bool {
	used := map[string]bool{}
	for _, requirement := range EmbeddingRequirements(c.ConfigForGlobalModelServices(), true) {
		used[requirement.Deployment] = true
	}
	for _, scoped := range c.consumerScopes() {
		for _, requirement := range EmbeddingRequirements(scoped, false) {
			used[requirement.Deployment] = true
		}
	}
	delete(used, "")
	return used
}

// validateEmbeddingStores checks, at load, the embedding deployments
// global.model_catalog.embeddings names and the bindings of the stores.
func validateEmbeddingStores(cfg *RouterConfig) error {
	for _, named := range []struct{ path, name string }{
		{"global.model_catalog.embeddings.default", cfg.EmbeddingDeployment},
		{"global.model_catalog.embeddings.multimodal", cfg.MultimodalEmbeddingDeployment},
	} {
		if named.name == "" {
			continue
		}
		if _, err := cfg.embeddingDeployment(named.path, named.name); err != nil {
			return err
		}
	}
	if err := cfg.validateEmbeddingBinding("global.stores.response_cache.embedding", cfg.SemanticCache.Embedding); err != nil {
		return err
	}
	if field, width := cfg.SemanticCache.vectorWidth(); width > 0 && cfg.SemanticCache.Embedding.width() > 0 && width != cfg.SemanticCache.Embedding.width() {
		return fmt.Errorf("global.stores.response_cache.embedding.dimension: %d differs from %s %d; keep one", cfg.SemanticCache.Embedding.width(), field, width)
	}
	if err := cfg.validateEmbeddingBinding("global.stores.memory.embedding", cfg.Memory.Embedding); err != nil {
		return err
	}
	if field, width := cfg.Memory.backendWidth(); width > 0 && cfg.Memory.Embedding.width() > 0 && width != cfg.Memory.Embedding.width() {
		return fmt.Errorf("global.stores.memory.embedding.dimension: %d differs from %s %d; keep one", cfg.Memory.Embedding.width(), field, width)
	}
	if cfg.VectorStore != nil {
		return cfg.validateEmbeddingBinding("global.stores.vector_store.embedding", cfg.VectorStore.Embedding)
	}
	return nil
}

// validateRAGQueryEmbeddings checks the bindings a RAG source at path embeds
// its queries with.
func validateRAGQueryEmbeddings(cfg *RouterConfig, path string, source *ContextSource) error {
	bindings := RAGQueryEmbeddings(source)
	for _, at := range slices.Sorted(maps.Keys(bindings)) {
		if err := cfg.validateEmbeddingBinding(path+"."+at, bindings[at]); err != nil {
			return err
		}
	}
	return nil
}

// validateEmbeddingBinding checks a binding at path: its deployment, or
// embeddings.default, serves embeddings, at a dimension it can produce.
func (c *RouterConfig) validateEmbeddingBinding(path string, binding *EmbeddingBinding) error {
	if binding == nil {
		return nil
	}
	if binding.Dimension < 0 {
		return fmt.Errorf("%s.dimension must be positive", path)
	}
	at := path + ".deployment"
	if binding.Deployment == "" {
		at = "global.model_catalog.embeddings.default"
	}
	name := c.EmbeddingDeploymentOf(binding)
	deployment, err := c.embeddingDeployment(at, name)
	if err != nil || binding.Dimension == 0 {
		return err
	}
	if declared := deployment.DeclaredDimensions(); declared > 0 && declared != binding.Dimension {
		return fmt.Errorf("%s.dimension: deployment %q answers with %d dimensions (capabilities.dimensions), not %d", path, name, declared, binding.Dimension)
	}
	if spec := registeredModel(deployment.Artifact); deployment.IsModelRuntime() && spec != nil && spec.EmbeddingDim > 0 && binding.Dimension > spec.EmbeddingDim {
		return fmt.Errorf("%s.dimension: deployment %q runs %s, which embeds at most %d dimensions, not %d", path, name, spec.RepoID, spec.EmbeddingDim, binding.Dimension)
	}
	return nil
}

// embeddingDeployment is the declared deployment name, at path, which must
// serve embeddings: an openai_embeddings deployment, or a model_runtime one
// whose built-in model is an embedding model.
func (c *RouterConfig) embeddingDeployment(path, name string) (ModelDeployment, error) {
	if name == "" {
		return ModelDeployment{}, fmt.Errorf("%s: no embedding deployment is named; name one (vela-embedding is pre-declared)", path)
	}
	deployment, ok := c.ModelDeployments[name]
	if !ok {
		return ModelDeployment{}, fmt.Errorf("%s: unknown deployment %q; %s", path, name, declaredDeployments(c.ModelDeployments))
	}
	deployment = deployment.WithDefaults()
	switch deployment.Provider {
	case OpenAIEmbeddingsProvider:
		return deployment, nil
	case ModelRuntimeProvider:
		if spec := registeredModel(deployment.Artifact); spec != nil && deployment.Managed() &&
			spec.Purpose != PurposeEmbedding && spec.Purpose != PurposeSemanticSimilarity {
			return ModelDeployment{}, fmt.Errorf("%s: deployment %q runs %s, a %s model that serves no embeddings", path, name, spec.RepoID, spec.Purpose)
		}
		return deployment, nil
	}
	return ModelDeployment{}, fmt.Errorf("%s: deployment %q is a %s deployment; embeddings need a model_runtime or openai_embeddings deployment", path, name, deployment.Provider)
}
