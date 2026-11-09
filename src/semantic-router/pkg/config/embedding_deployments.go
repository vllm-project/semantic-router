package config

import (
	"fmt"
	"path/filepath"
	"reflect"
	"strings"
)

// An embedding model keyword (mmbert, qwen3, multimodal), which response
// cache, memory, vector store and preference settings still name, is an
// alias of a deployment: embeddings.multimodal for multimodal and
// embeddings.default for the default text model, or the pre-declared
// omni-nano and vela-embedding when a configuration declares them again;
// else the declared deployment that serves the keyword's configured model
// (the pre-declared ones by default), else a deployment of that model named
// after it.

// embeddingModelPaths are the deprecated per-keyword model paths.
func (c *RouterConfig) embeddingModelPath(model string) (string, bool) {
	switch model {
	case "mmbert":
		return c.MmBertModelPath, true
	case "qwen3":
		return c.Qwen3ModelPath, true
	case "multimodal":
		return c.MultiModalModelPath, true
	}
	return "", false
}

// EmbeddingModelDeployment resolves an embedding model keyword to the
// deployment that serves it. Inputs over its budget are truncated, as
// embeddings always were.
func (c *RouterConfig) EmbeddingModelDeployment(model string) (string, ModelDeployment, error) {
	model = strings.ToLower(strings.TrimSpace(model))
	if field, name := c.namedEmbeddingDeployment(model); name != "" {
		deployment, ok := c.ModelDeployments[name]
		if !ok {
			return "", ModelDeployment{}, fmt.Errorf("%s: unknown deployment %q; %s", field, name, declaredDeployments(c.ModelDeployments))
		}
		return name, deployment.WithDefaults(), nil
	}
	path, served := c.embeddingModelPath(model)
	if !served {
		return "", ModelDeployment{}, fmt.Errorf("embedding model %q is not served by the model runtime; use mmbert, qwen3 or multimodal, or an openai_embeddings deployment (vllm-sr config migrate rewrites legacy settings)", model)
	}
	deployment, err := ImplicitModelRuntimeDeployment(path, c.EmbeddingModels.UseCPU)
	if err != nil {
		return "", ModelDeployment{}, fmt.Errorf("embedding model %s: %w", model, err)
	}
	deployment.Input.Overflow = "truncate"
	deployment = deployment.WithDefaults()
	for _, name := range sortedModelKeys(c.ModelDeployments) {
		declared := c.ModelDeployments[name].WithDefaults()
		if declared.IsModelRuntime() && declared.Managed() && len(declared.Replicas) == 0 &&
			declared.Artifact == deployment.Artifact && declared.Revision == deployment.Revision && declared.Device == deployment.Device {
			return name, declared, nil
		}
	}
	name := filepath.Base(strings.TrimRight(path, "/"))
	if spec := GetModelByPath(path); spec != nil {
		name = filepath.Base(spec.LocalPath)
	}
	if deployment.Device != "cpu" {
		name += "-" + deployment.Device
	}
	return name, deployment, nil
}

// namedEmbeddingDeployment is the deployment a configuration names for an
// embedding keyword, with the field that names it: embeddings.multimodal for
// multimodal; for the default text model, a model_runtime deployment
// embeddings.default names (a remote one serves every text embedding
// elsewhere); else a pre-declared deployment the configuration declares again
// under its own name. A pre-declared deployment left as it is resolves by the
// model it serves, so a deprecated model path or device keeps its own.
func (c *RouterConfig) namedEmbeddingDeployment(model string) (string, string) {
	switch model {
	case "multimodal":
		if c.MultimodalEmbeddingDeployment != "" {
			return "embeddings.multimodal", c.MultimodalEmbeddingDeployment
		}
		return "deployments", c.redeclaredDeployment(MultimodalEmbeddingDeployment)
	case DefaultEmbeddingModel:
		if name := c.EmbeddingDeployment; name != "" && c.ModelDeployments[name].IsModelRuntime() {
			return "embeddings.default", name
		}
		return "deployments", c.redeclaredDeployment(EmbeddingDefaultDeployment)
	}
	return "", ""
}

// redeclaredDeployment is name when the configuration declares its own
// deployment under that pre-declared name, and "" while the pre-declared one
// stands.
func (c *RouterConfig) redeclaredDeployment(name string) string {
	declared, ok := c.ModelDeployments[name]
	predeclared, known := DefaultModelDeployments()[name]
	if !ok || !known || reflect.DeepEqual(declared.WithDefaults(), predeclared.WithDefaults()) {
		return ""
	}
	return name
}

// EmbeddingAdmission is the admission of an embedding model's deployment:
// its own, else the deprecated per-keyword key embedding:<model>.
func (c *RouterConfig) EmbeddingAdmission(name, model string) AdmissionConfig {
	if admission, ok := c.ModelAdmission[name]; ok {
		return admission
	}
	return c.ModelAdmission["embedding:"+model]
}

// UsesRemoteEmbeddings reports whether one remote embedding model serves
// every text embedding: embeddings.default names an openai_embeddings
// deployment, or the deprecated embeddings.semantic.endpoint declares one, as
// it always did.
func (c *RouterConfig) UsesRemoteEmbeddings() bool {
	if c == nil {
		return false
	}
	if c.EmbeddingModels.UsesRemoteEmbeddingBackend() {
		return true
	}
	deployment, ok := c.ModelDeployments[c.EmbeddingDeployment]
	return ok && deployment.Provider == OpenAIEmbeddingsProvider
}
