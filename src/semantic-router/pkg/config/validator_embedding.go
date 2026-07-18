package config

import (
	"fmt"
	"net/url"
	"strings"
)

// validateEmbeddingContracts validates embedding signal config against the
// runtime embedding model. Mirrors the per-family validator pattern in
// validator_modality.go and friends, and is wired into validateConfigStructure
// so misconfigured rules fail at config-load time rather than at first use.
func validateEmbeddingContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateMmBertModelPath(cfg.EmbeddingModels.MmBertModelPath); err != nil {
		return err
	}
	if err := validateRemoteEmbeddingProviderConfig(cfg.EmbeddingModels); err != nil {
		return err
	}
	return validateEmbeddingRuleModalities(cfg.EmbeddingRules, cfg.EmbeddingModels.EmbeddingConfig.ModelType)
}

func validateRemoteEmbeddingProviderConfig(models EmbeddingModels) error {
	if !models.UsesRemoteEmbeddingBackend() {
		return nil
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
// slot. Classic BERT (e.g. all-MiniLM-L12-v2) uses a different tensor layout
// than ModernBERT/mmBERT and will crash the Rust loader with a cryptic tensor
// name mismatch. Catching it here gives the user a clear message at config-load
// time instead of a crash at model-init time.
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
			"mmbert_model_path is set to %q, which is a classic BERT model (%s, %s). "+
				"Classic BERT models are not compatible with the mmBERT loader. "+
				"Use 'bert_model_path' for this model instead, or set mmbert_model_path "+
				"to a ModernBERT-based model such as 'models/mmbert-embed-32k-2d-matryoshka'",
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

// validateEmbeddingRuleModalities checks every embedding rule's declared
// query_modality is recognized and compatible with the configured embedding
// model. Returns a non-nil error listing every misconfigured rule, or nil
// when all rules pass.
//
// Rules:
//   - unset or "text": always allowed.
//   - "image": requires global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal.
//     Rejected otherwise so the candidates and queries cannot end up embedded
//     in mismatched spaces.
//   - "audio": rejected at config-load with a clear "planned" message. The
//     schema accepts the value so future configs do not need to migrate, but
//     the audio FFI is not yet exposed by candle-binding, so any rule
//     declaring audio cannot be served and is loud-failed early.
//   - any other value: rejected as an unknown modality so a typo like "imag"
//     cannot silently drop a rule out of every classification path.
func validateEmbeddingRuleModalities(rules []EmbeddingRule, modelType string) error {
	normalizedModelType := strings.ToLower(strings.TrimSpace(modelType))
	var problems []string
	for _, rule := range rules {
		raw := QueryModality(strings.ToLower(strings.TrimSpace(string(rule.QueryModality))))
		switch raw {
		case "", QueryModalityText:
			// Text is always allowed; preserves existing behavior for rules
			// that omit query_modality entirely.
		case QueryModalityImage:
			if normalizedModelType != "multimodal" {
				problems = append(problems, fmt.Sprintf(
					"embedding rule %q declares query_modality=image, which requires global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal. Remove the rule, set query_modality to text, or change model_type to multimodal (current model_type=%q)",
					rule.Name, modelType))
			}
		case QueryModalityAudio:
			problems = append(problems, fmt.Sprintf(
				"embedding rule %q declares query_modality=audio, but the audio FFI is not yet exposed by candle-binding. Audio query support is planned; remove the rule or set query_modality to text/image until the FFI lands",
				rule.Name))
		default:
			problems = append(problems, fmt.Sprintf(
				"embedding rule %q declares unknown query_modality=%q (allowed values: text, image, audio)",
				rule.Name, string(rule.QueryModality)))
		}
	}
	if len(problems) == 0 {
		return nil
	}
	return fmt.Errorf("invalid embedding rule configuration:\n  - %s", strings.Join(problems, "\n  - "))
}
