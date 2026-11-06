package config

import (
	"fmt"
	"net/url"
	"strings"
)

// validateEmbeddingContracts validates the embedding model configuration, so
// a misconfigured model fails at config-load time rather than at first use.
func validateEmbeddingContracts(cfg *RouterConfig) error {
	return validateEmbeddingModelContracts(cfg)
}

func validateEmbeddingModelContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateMmBertModelPath(cfg.EmbeddingModels.MmBertModelPath); err != nil {
		return err
	}
	if err := validateRemoteEmbeddingProviderConfig(cfg); err != nil {
		return err
	}
	return nil
}

func validateRemoteEmbeddingProviderConfig(cfg *RouterConfig) error {
	models := cfg.EmbeddingModels
	if !models.UsesRemoteEmbeddingBackend() {
		return nil
	}
	if strings.TrimSpace(models.Endpoint.BaseURL) == "" && cfg.EmbeddingDeployment != "" {
		return fmt.Errorf("global.model_catalog.embeddings.semantic.embedding_config: backend %s and model_type %s describe the deprecated embeddings.semantic.endpoint; embeddings.default names the remote embedding deployment, so remove them", EmbeddingBackendOpenAICompatible, EmbeddingModelTypeRemote)
	}

	var problems []string
	problems = append(problems, validateRemoteEmbeddingEndpoint(models.Endpoint)...)
	problems = append(problems, validateRemoteEmbeddingDimensions(models)...)
	problems = append(problems, validateRemoteEmbeddingModelType(models.EmbeddingConfig.ModelType)...)

	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("invalid remote embedding provider configuration:\n  - %s", strings.Join(problems, "\n  - "))
}

func validateRemoteEmbeddingEndpoint(endpoint EmbeddingEndpointConfig) []string {
	var problems []string
	baseURL := strings.TrimSpace(endpoint.BaseURL)
	if baseURL == "" {
		problems = append(problems, "endpoint.base_url is required")
	} else if parsed, err := url.Parse(baseURL); err != nil || parsed.Scheme == "" || parsed.Host == "" {
		problems = append(problems, fmt.Sprintf("endpoint.base_url must include a valid scheme and host, got %q", endpoint.BaseURL))
	}
	if strings.TrimSpace(endpoint.Model) == "" {
		problems = append(problems, "endpoint.model is required")
	}
	if endpoint.TimeoutSeconds < 0 {
		problems = append(problems, "endpoint.timeout_seconds must be non-negative")
	}
	if endpoint.MaxRetries < 0 {
		problems = append(problems, "endpoint.max_retries must be non-negative")
	}
	if endpoint.MaxResponseBytes < 0 {
		problems = append(problems, "endpoint.max_response_bytes must be non-negative")
	}
	return problems
}

func validateRemoteEmbeddingDimensions(models EmbeddingModels) []string {
	var problems []string
	endpoint := models.Endpoint
	if endpoint.Dimensions < 0 {
		problems = append(problems, "endpoint.dimensions must be non-negative")
	}
	targetDimension := models.EmbeddingConfig.TargetDimension
	if targetDimension < 0 {
		problems = append(problems, "embedding_config.target_dimension must be non-negative")
	}
	if endpoint.Dimensions > 0 && targetDimension > 0 && endpoint.Dimensions != targetDimension {
		problems = append(problems, fmt.Sprintf("endpoint.dimensions (%d) must match embedding_config.target_dimension (%d)", endpoint.Dimensions, targetDimension))
	}
	return problems
}

func validateRemoteEmbeddingModelType(rawModelType string) []string {
	modelType := strings.ToLower(strings.TrimSpace(rawModelType))
	if modelType == "" || modelType == EmbeddingModelTypeRemote {
		return nil
	}
	return []string{fmt.Sprintf("embedding_config.model_type must be %q for remote backend, got %q", EmbeddingModelTypeRemote, rawModelType)}
}

// validateMmBertModelPath rejects classic BERT models in the mmbert_model_path
// slot: the model runtime has no family for classic BERT (e.g.
// all-MiniLM-L12-v2), so catching it here gives the user a clear message at
// config-load time instead of a failed deployment at startup.
func validateMmBertModelPath(modelPath string) error {
	if modelPath == "" {
		return nil
	}
	model := GetModelByPath(modelPath)
	if model == nil {
		return nil
	}
	if model.Purpose == PurposeSemanticSimilarity {
		return fmt.Errorf(
			"mmbert_model_path is set to %q, which is a classic BERT model (%s, %s) that the model runtime does not serve; "+
				"set mmbert_model_path to Vela Embedding ('models/Vela-1.0-Encoder-307M-Embedding') or use an OpenAI-compatible embedding endpoint",
			modelPath, model.RepoID, model.ParameterSize,
		)
	}
	return nil
}

// ValidateEmbeddingContracts is the exported counterpart of the private
// validateEmbeddingContracts function. It remains available for narrow callers
// that need only the embedding-modality slice; Kubernetes reconciliation should
// prefer ValidateKubernetesConfigContracts so every shared family validator runs
// through one dispatch surface.
func ValidateEmbeddingContracts(cfg *RouterConfig) error {
	return validateEmbeddingContracts(cfg)
}
