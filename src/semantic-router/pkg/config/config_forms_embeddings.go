package config

import (
	"fmt"
	"path/filepath"
	"slices"
	"strings"

	"gopkg.in/yaml.v3"
)

// The embedding model keywords stores named before they named a deployment:
// mmbert named Vela Embedding, which embeddings.default names; qwen3 and
// multimodal named the models embeddings.semantic's qwen3_model_path and
// multimodal_model_path held, multimodal Vela 1.0 Omni-Nano by default, as
// the omni-nano deployment; gemma and bert named models Vela Embedding
// replaced.
const (
	embeddingKeywordVela        = "mmbert"
	embeddingKeywordQwen3       = "qwen3"
	embeddingKeywordMultimodal  = "multimodal"
	multimodalKeywordDeployment = "omni-nano"
	multimodalKeywordModel      = "models/vela-1.0-omni-nano"
)

var retiredEmbeddingKeywords = []string{"gemma", "bert"}

// embeddingForms are the embedding settings from before every store and
// collection named its deployment: embedding bindings, the stores' model
// keywords and the vector store's width, admission by model keyword, and
// embeddings.semantic, rewritten after them because they read the models it
// held; then the forms from before embeddings served recall alone: a bare
// embeddings.default name, embeddings.multimodal and the embeddings API.
var embeddingForms = []ConfigForm{
	{
		Code:    CodeRemovedField,
		Paths:   slices.Concat(RoutingScopes("model_bindings.embedding"), []string{"global.model_catalog.bindings.embedding"}),
		What:    "the embedding binding is removed",
		Becomes: "global.model_catalog.embeddings.default; a RAG source's backend_config.embedding.deployment for a recipe's own",
		Migrate: migrateEmbeddingBinding,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.stores.response_cache.embedding_model", "global.stores.memory.embedding_model", "global.stores.vector_store.embedding_model"},
		What:    "embedding_model is removed",
		Becomes: "embedding.deployment names the store's deployment; without it the store embeds with embeddings.default",
		Migrate: migrateEmbeddingKeyword,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.stores.vector_store.embedding_dimension"},
		What:    "embedding_dimension is removed",
		Becomes: "embedding.dimension",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, "embedding_dimension")
			return setEmbeddingField(m, at.Parent, "dimension", at.Node)
		},
	},
	{
		Code:  CodeRemovedField,
		Paths: []string{"global.model_catalog.admission.*"},
		Match: func(_ *yaml.Node, path string) bool {
			return strings.HasPrefix(strings.TrimPrefix(path, "global.model_catalog.admission."), "embedding:")
		},
		What:    "admission by embedding model keyword is removed",
		Becomes: "the admission of the embedding deployment, keyed by its name",
		Migrate: migrateEmbeddingAdmission,
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.embeddings.semantic"},
		What:    "embeddings.semantic is removed",
		Becomes: "embeddings.default names the embedding deployment; a store's embedding block names its own",
		Migrate: migrateEmbeddingSemantic,
	},
	{
		Code:  CodeInvalidValue,
		Paths: []string{"global.model_catalog.embeddings.default"},
		Match: func(node *yaml.Node, _ string) bool { return node.Kind == yaml.ScalarNode },
		Describe: func(at FormMatch) (string, string) {
			return fmt.Sprintf("embeddings.default %q is a bare deployment name", at.Node.Value), fmt.Sprintf("embeddings.default: {deployment: %s}, as system.decision_model names its deployment", at.Node.Value)
		},
		Migrate: func(m *Migration, at FormMatch) error {
			setEmbeddingDefault(m, at.Parent, at.Node.Value)
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.model_catalog.embeddings.multimodal"},
		What:    "embeddings.multimodal is removed: embeddings serve text recall alone",
		Becomes: "nothing; a decision about an image asks a systemone question with input.image",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, "multimodal")
			m.Note(at.Path, fmt.Sprintf("deployment %s embedded the images and audio of the embeddings API, which is removed; a deployment nothing names is never started", at.Node.Value))
			deleteEmpty(m.Root(), "global", "model_catalog", "embeddings")
			return nil
		},
	},
	{
		Code:    CodeRemovedField,
		Paths:   []string{"global.services.api.embeddings"},
		What:    "the embeddings API is removed: embeddings serve recall alone",
		Becomes: "nothing; the embedding deployment starts when the response cache, memory or a vector store recalls with it",
		Migrate: func(m *Migration, at FormMatch) error {
			m.Delete(at.Parent, "embeddings")
			deleteEmpty(m.Root(), "global", "services", "api")
			return nil
		},
	},
}

// embeddingDefaultName is the deployment embeddings.default names, in its
// {deployment} form or as the bare name a form rewrites.
func embeddingDefaultName(embeddings *yaml.Node) string {
	if name := scalarAt(embeddings, "default"); name != "" {
		return name
	}
	return scalarAt(mappingValue(embeddings, "default"), "deployment")
}

// setEmbeddingDefault writes embeddings.default: {deployment: name}.
func setEmbeddingDefault(m *Migration, embeddings *yaml.Node, name string) {
	binding := &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
	m.Set(binding, "deployment", Scalar(name))
	m.Set(embeddings, "default", binding)
}

// migrateEmbeddingBinding writes a global embedding binding as
// embeddings.default, and a recipe's as the deployment of the scope's
// RAG sources that embed their queries.
func migrateEmbeddingBinding(m *Migration, at FormMatch) error {
	name := at.Node.Value
	if at.Node.Kind == yaml.MappingNode {
		name = scalarAt(at.Node, "deployment")
	}
	m.Delete(at.Parent, at.Key)
	if strings.HasPrefix(at.Path, "global.") {
		deleteEmpty(m.Root(), "global", "model_catalog", "bindings")
		embeddings := ensureMapping(m.Root(), "global", "model_catalog", "embeddings")
		if current := embeddingDefaultName(embeddings); current != "" && current != name {
			return fmt.Errorf("%s: embeddings.default %q and bindings.embedding %q name different deployments; keep embeddings.default", at.Path, current, name)
		}
		setEmbeddingDefault(m, embeddings, name)
		return nil
	}
	deleteEmpty(scopeRouting(m.Root(), at.Path), "model_bindings")
	sources := 0
	for _, decision := range sequenceAt(scopeRouting(m.Root(), at.Path), "decisions") {
		for _, plugin := range sequenceAt(decision, "plugins") {
			if scalarAt(plugin, "type") != DecisionPluginContext {
				continue
			}
			for _, source := range sequenceAt(plugin, "configuration", "sources") {
				if scalarAt(source, "type") != ContextSourceRAG {
					continue
				}
				count, err := bindRAGQueryEmbedding(m, scalarAt(source, "backend"), mappingValue(source, "backend_config"), name)
				if err != nil {
					return fmt.Errorf("%s: %w", at.Path, err)
				}
				sources += count
			}
		}
	}
	if sources == 0 {
		m.Note(at.Path, fmt.Sprintf("no RAG source of the recipe embeds its queries, so deployment %s embeds nothing here", name))
	}
	return nil
}

// bindRAGQueryEmbedding names deployment in the backend configuration of a
// RAG backend that embeds its queries, and in a hybrid backend's two; it
// returns how many it named.
func bindRAGQueryEmbedding(m *Migration, backend string, config *yaml.Node, deployment string) (int, error) {
	if config == nil || config.Kind != yaml.MappingNode {
		return 0, nil
	}
	switch backend {
	case "milvus", "qdrant":
	case "external_api":
		if format := scalarAt(config, "request_format"); format != "pinecone" && format != "weaviate" {
			return 0, nil
		}
	case "hybrid":
		primary, err := bindRAGQueryEmbedding(m, scalarAt(config, "primary"), mappingValue(config, "primary_config"), deployment)
		if err != nil {
			return 0, err
		}
		fallback, err := bindRAGQueryEmbedding(m, scalarAt(config, "fallback"), mappingValue(config, "fallback_config"), deployment)
		return primary + fallback, err
	default:
		return 0, nil
	}
	return 1, setEmbeddingField(m, config, "deployment", Scalar(deployment))
}

// migrateEmbeddingKeyword points a store at the deployment its keyword
// implied, declared in global.model_catalog.deployments when the document
// declares none of that name; mmbert, and a retired model, leave the store on
// embeddings.default.
func migrateEmbeddingKeyword(m *Migration, at FormMatch) error {
	keyword := strings.ToLower(strings.TrimSpace(at.Node.Value))
	m.Delete(at.Parent, "embedding_model")
	name, deployment, err := keywordEmbeddingDeployment(m.Root(), keyword)
	if err != nil {
		return fmt.Errorf("%s: %w", at.Path, err)
	}
	switch {
	case slices.Contains(retiredEmbeddingKeywords, keyword):
		m.Note(at.Path, fmt.Sprintf("embedding_model %s named a retired model; the store embeds with embeddings.default, so its vectors must be indexed again", keyword))
	case deployment != nil:
		m.Note(at.Path, fmt.Sprintf("the store embeds with deployment %s, which serves %s, the model embedding_model %s named", name, deployment.Artifact, keyword))
	}
	if name == "" {
		return nil
	}
	if deployment != nil {
		if err := declareEmbeddingDeployment(m, name, *deployment); err != nil {
			return err
		}
	}
	return setEmbeddingField(m, at.Parent, "deployment", Scalar(name))
}

// keywordEmbeddingDeployment is the deployment an embedding model keyword
// implied, "" for embeddings.default, with the deployment to declare when
// the document declares none of its name. A remote default served every text
// keyword.
func keywordEmbeddingDeployment(root *yaml.Node, keyword string) (string, *ModelDeployment, error) {
	embeddings := mappingAt(root, "global", "model_catalog", "embeddings")
	semantic := mappingValue(embeddings, "semantic")
	switch {
	case keyword == embeddingKeywordVela || slices.Contains(retiredEmbeddingKeywords, keyword):
		return "", nil, nil
	case keyword == embeddingKeywordMultimodal:
		if name := scalarAt(embeddings, "multimodal"); name != "" {
			return name, nil, nil
		}
		model := scalarAt(semantic, "multimodal_model_path")
		if model == "" {
			model = multimodalKeywordModel
		}
		return impliedEmbeddingDeployment(root, model, semanticUsesCPU(semantic), multimodalKeywordDeployment)
	case keyword == embeddingKeywordQwen3:
		if remoteEmbeddingDefault(root) {
			return "", nil, nil
		}
		model := scalarAt(semantic, "qwen3_model_path")
		if model == "" {
			return "", nil, fmt.Errorf("embedding_model qwen3 names no model: embeddings.semantic.qwen3_model_path is unset; name the store's deployment in embedding.deployment")
		}
		return impliedEmbeddingDeployment(root, model, semanticUsesCPU(semantic), "")
	}
	return "", nil, fmt.Errorf("embedding_model %q is no embedding model keyword; name the store's deployment in embedding.deployment", keyword)
}

// impliedEmbeddingDeployment is the deployment that serves a model on the
// model runtime: one the document declares for it on the same device, else
// a new one named name, or after the model.
func impliedEmbeddingDeployment(root *yaml.Node, model string, useCPU bool, name string) (string, *ModelDeployment, error) {
	model = replacedEmbeddingModel(model)
	deployment, err := modelRuntimeDeploymentOf(model, useCPU)
	if err != nil {
		return "", nil, err
	}
	deployment.Input.Overflow = "truncate"
	var declared map[string]ModelDeployment
	if node := mappingAt(root, "global", "model_catalog", "deployments"); node != nil {
		if err := node.Decode(&declared); err != nil {
			return "", nil, err
		}
	}
	for _, key := range sortedModelKeys(declared) {
		existing := declared[key].WithDefaults()
		if existing.IsModelRuntime() && existing.Artifact == deployment.Artifact && existing.Revision == deployment.Revision && existing.Device == deployment.Device {
			return key, nil, nil
		}
	}
	if name == "" {
		name = filepath.Base(strings.TrimRight(model, "/"))
		if spec := GetModelByPath(model); spec != nil {
			name = filepath.Base(spec.LocalPath)
		}
		if deployment.Device != "cpu" {
			name += "-" + deployment.Device
		}
	}
	return name, &deployment, nil
}

// remoteEmbeddingDefault reports whether one remote model served every text
// embedding: embeddings.default names an openai_embeddings deployment, or
// embeddings.semantic declares a remote endpoint.
func remoteEmbeddingDefault(root *yaml.Node) bool {
	embeddings := mappingAt(root, "global", "model_catalog", "embeddings")
	if name := embeddingDefaultName(embeddings); name != "" {
		return scalarAt(mappingAt(root, "global", "model_catalog", "deployments", name), "provider") == OpenAIEmbeddingsProvider
	}
	semantic := mappingValue(embeddings, "semantic")
	return usesRemoteEmbeddingEndpoint(semantic)
}

// usesRemoteEmbeddingEndpoint reports whether embeddings.semantic embedded
// through its endpoint: an openai_compatible backend or the remote model type,
// with a base URL.
func usesRemoteEmbeddingEndpoint(semantic *yaml.Node) bool {
	settings := mappingValue(semantic, "embedding_config")
	remote := strings.EqualFold(scalarAt(settings, "backend"), EmbeddingBackendOpenAICompatible) || strings.EqualFold(scalarAt(settings, "model_type"), "remote")
	return remote && strings.TrimSpace(scalarAt(mappingValue(semantic, "endpoint"), "base_url")) != ""
}

// semanticUsesCPU is embeddings.semantic.use_cpu, which defaulted to true.
func semanticUsesCPU(semantic *yaml.Node) bool {
	return scalarAt(semantic, "use_cpu") != "false"
}

// declareEmbeddingDeployment writes deployment under name in
// global.model_catalog.deployments unless the document declares that name.
func declareEmbeddingDeployment(m *Migration, name string, deployment ModelDeployment) error {
	deployments := ensureMapping(m.Root(), "global", "model_catalog", "deployments")
	if mappingValue(deployments, name) != nil {
		return nil
	}
	return m.SetValue(deployments, name, deployment)
}

// setEmbeddingField writes key of the embedding block of holder, which must
// not set it to something else already.
func setEmbeddingField(m *Migration, holder *yaml.Node, key string, value *yaml.Node) error {
	embedding := mappingValue(holder, "embedding")
	if embedding == nil {
		embedding = &yaml.Node{Kind: yaml.MappingNode, Tag: "!!map"}
		m.Set(holder, "embedding", embedding)
	}
	if current := mappingValue(embedding, key); current != nil && current.Value != value.Value {
		return fmt.Errorf("embedding.%s is %s already, not %s", key, current.Value, value.Value)
	}
	m.Set(embedding, key, value)
	return nil
}

// migrateEmbeddingAdmission keys an admission by the deployment its keyword
// implied; the deployment's own admission, when written, wins as it did.
func migrateEmbeddingAdmission(m *Migration, at FormMatch) error {
	name := ""
	if keyword := strings.TrimPrefix(at.Key, "embedding:"); keyword != "remote" {
		implied, _, err := keywordEmbeddingDeployment(m.Root(), keyword)
		if err != nil {
			return fmt.Errorf("%s: %w", at.Path, err)
		}
		name = implied
	}
	if name == "" {
		var err error
		if name, err = defaultEmbeddingDeploymentOf(m.Root()); err != nil {
			return fmt.Errorf("%s: %w", at.Path, err)
		}
	}
	if mappingValue(at.Parent, name) != nil {
		m.Delete(at.Parent, at.Key)
		m.Note(at.Path, fmt.Sprintf("deployment %s has its own admission, which gated it already", name))
		return nil
	}
	return m.Rename(at.Parent, at.Key, name)
}

// defaultEmbeddingDeploymentOf is the deployment embeddings.default names
// once embeddings.semantic is rewritten.
func defaultEmbeddingDeploymentOf(root *yaml.Node) (string, error) {
	embeddings := mappingAt(root, "global", "model_catalog", "embeddings")
	if name := embeddingDefaultName(embeddings); name != "" {
		return name, nil
	}
	name, _, err := semanticEmbeddingDeployment(root, mappingValue(embeddings, "semantic"))
	if name == "" && err == nil {
		name = EmbeddingDefaultDeployment
	}
	return name, err
}

// migrateEmbeddingSemantic writes the models embeddings.semantic chose as
// deployments: a remote endpoint as an openai_embeddings deployment, and an
// mmbert_model_path other than Vela Embedding's, or Vela Embedding off the
// CPU, as a model_runtime one; embeddings.default names the one the document
// does not name otherwise. The rest held the embeddings API's default view
// and the settings of the embedding classifiers, which are removed.
func migrateEmbeddingSemantic(m *Migration, at FormMatch) error {
	semantic, embeddings := at.Node, at.Parent
	name, deployment, err := semanticEmbeddingDeployment(m.Root(), semantic)
	if err != nil {
		return fmt.Errorf("%s: %w", at.Path, err)
	}
	m.Delete(embeddings, "semantic")
	if model := scalarAt(semantic, "mmbert_model_path"); model != "" {
		if replacement, ok := legacyModelReplacement(model); ok {
			m.Note(at.Path, fmt.Sprintf("mmbert_model_path %s is replaced by %s: %s", model, replacement.target, replacement.note))
		}
	}
	settings := mappingValue(semantic, "embedding_config")
	if strings.EqualFold(scalarAt(settings, "model_type"), embeddingKeywordQwen3) {
		m.Note(at.Path, "the embeddings API served qwen3 by default; it serves embeddings.default now, so name a Qwen3 deployment there to keep it, which every store without an embedding block then embeds with")
	}
	// The response cache fell back to the embedding classifiers' threshold.
	if threshold := mappingValue(settings, "min_score_threshold"); threshold != nil && threshold.Value != "0.5" {
		if store := mappingAt(m.Root(), "global", "stores", "response_cache"); store != nil && mappingValue(store, "similarity_threshold") == nil {
			m.Set(store, "similarity_threshold", threshold)
		}
	}
	if name != "" {
		if deployment != nil {
			if err := declareEmbeddingDeployment(m, name, *deployment); err != nil {
				return err
			}
		}
		if current := embeddingDefaultName(embeddings); current != "" && current != name {
			m.Note(at.Path, fmt.Sprintf("embeddings.default names %s, so deployment %s, the model embeddings.semantic chose, embeds for nothing until a store names it", current, name))
		} else {
			setEmbeddingDefault(m, embeddings, name)
		}
	}
	deleteEmpty(m.Root(), "global", "model_catalog", "embeddings")
	return nil
}

// replacedEmbeddingModel is the model the runtime serves for an embedding
// model reference: the Vela 1.0 release that replaces an earlier registry
// model, or the reference itself.
func replacedEmbeddingModel(model string) string {
	if replacement, ok := legacyModelReplacement(model); ok {
		return replacement.target
	}
	return model
}

// semanticEmbeddingDeployment is the deployment embeddings.semantic chose for
// the default text embedding model, "" for vela-embedding as pre-declared.
func semanticEmbeddingDeployment(root *yaml.Node, semantic *yaml.Node) (string, *ModelDeployment, error) {
	if usesRemoteEmbeddingEndpoint(semantic) {
		var endpoint struct {
			BaseURL          string `yaml:"base_url"`
			Model            string `yaml:"model"`
			APIKeyEnv        string `yaml:"api_key_env"`
			TimeoutSeconds   int    `yaml:"timeout_seconds"`
			MaxRetries       int    `yaml:"max_retries"`
			MaxResponseBytes int64  `yaml:"max_response_bytes"`
			Dimensions       int    `yaml:"dimensions"`
		}
		if err := mappingValue(semantic, "endpoint").Decode(&endpoint); err != nil {
			return "", nil, err
		}
		deployment := ModelDeployment{Provider: OpenAIEmbeddingsProvider, Endpoint: strings.TrimRight(strings.TrimSpace(endpoint.BaseURL), "/"), Model: endpoint.Model}
		if endpoint.APIKeyEnv != "" {
			deployment.Auth = &DeploymentAuth{APIKeyEnv: endpoint.APIKeyEnv}
		}
		deployment.TimeoutMs = endpoint.TimeoutSeconds * 1000
		if endpoint.MaxRetries > 0 {
			deployment.MaxRetries = &endpoint.MaxRetries
		}
		if endpoint.MaxResponseBytes > 0 {
			deployment.Limits = &DeploymentLimits{MaxResponseBytes: endpoint.MaxResponseBytes}
		}
		if endpoint.Dimensions > 0 {
			deployment.Capabilities = &DeploymentCapabilities{Dimensions: endpoint.Dimensions}
		}
		return "remote-embedding", &deployment, nil
	}
	model := scalarAt(semantic, "mmbert_model_path")
	if model == "" {
		model = VelaEmbeddingModel
	}
	model = replacedEmbeddingModel(model)
	useCPU := semanticUsesCPU(semantic)
	if model == VelaEmbeddingModel && useCPU {
		return "", nil, nil
	}
	return impliedEmbeddingDeployment(root, model, useCPU, "")
}
