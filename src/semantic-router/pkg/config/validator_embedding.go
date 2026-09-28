package config

import (
	"fmt"
	"math"
	"net/url"
	"strings"
)

// validateEmbeddingContracts validates embedding signal config against the
// runtime embedding model. Mirrors the per-family validator pattern in
// validator_modality.go and friends, and is wired into validateConfigStructure
// so misconfigured rules fail at config-load time rather than at first use.
func validateEmbeddingContracts(cfg *RouterConfig) error {
	if err := validateEmbeddingModelContracts(cfg); err != nil {
		return err
	}
	return validateEmbeddingSignalContracts(cfg)
}

func validateEmbeddingModelContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	if err := validateMmBertModelPath(cfg.EmbeddingModels.MmBertModelPath); err != nil {
		return err
	}
	if err := validateRemoteEmbeddingProviderConfig(cfg.EmbeddingModels); err != nil {
		return err
	}
	return nil
}

func validateEmbeddingSignalContracts(cfg *RouterConfig) error {
	if cfg == nil {
		return nil
	}
	bound := cfg.ModelBindings["embedding"].Deployment != "" || cfg.GlobalModelBindings["embedding"].Deployment != ""
	return validateEmbeddingRuleModalities(cfg.EmbeddingRules, cfg.EmbeddingModels.EmbeddingConfig.ModelType, bound)
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
// Media queries require a multimodal default or an explicit binding. Prepared
// capability validation verifies that binding's actual encoders before startup.
func validateEmbeddingRuleModalities(rules []EmbeddingRule, modelType string, explicitBinding bool) error {
	normalizedModelType := strings.ToLower(strings.TrimSpace(modelType))
	var problems []string
	for _, rule := range rules {
		if rule.HasImageCandidates() && normalizedModelType != "multimodal" && !explicitBinding {
			problems = append(problems, fmt.Sprintf("embedding rule %q image candidates require model_type=multimodal or an explicit embedding binding with image capability", rule.Name))
		}
		// Every score needs a positive bank; negative-only or empty rules cannot match.
		if len(rule.Candidates)+len(rule.ImageCandidates) == 0 {
			problems = append(problems, fmt.Sprintf("embedding rule %q requires positive candidates or image_candidates", rule.Name))
		}
		switch rule.AggregationMethodConfiged {
		case "", AggregationMethodMax, AggregationMethodMean, AggregationMethodAny:
		default:
			problems = append(problems, fmt.Sprintf("embedding rule %q aggregation_method must be max, mean, or any", rule.Name))
		}
		bound := float32(1)
		if rule.HasNegativeCandidates() {
			bound = 2
		}
		if math.IsNaN(float64(rule.SimilarityThreshold)) || math.IsInf(float64(rule.SimilarityThreshold), 0) || rule.SimilarityThreshold < -bound || rule.SimilarityThreshold > bound {
			problems = append(problems, fmt.Sprintf("embedding rule %q threshold must be finite and within [%g, %g]", rule.Name, -bound, bound))
		}
		for _, values := range [][]string{rule.Candidates, rule.ImageCandidates, rule.NegativeCandidates, rule.NegativeImageCandidates} {
			for _, value := range values {
				if strings.TrimSpace(value) == "" {
					problems = append(problems, fmt.Sprintf("embedding rule %q has an empty candidate", rule.Name))
				}
			}
		}
		raw := QueryModality(strings.ToLower(strings.TrimSpace(string(rule.QueryModality))))
		switch raw {
		case "", QueryModalityText:
			// Text is always allowed; preserves existing behavior for rules
			// that omit query_modality entirely.
		case QueryModalityImage, QueryModalityAudio:
			if normalizedModelType != "multimodal" && !explicitBinding {
				problems = append(problems, fmt.Sprintf(
					"embedding rule %q declares query_modality=%s, which requires global.model_catalog.embeddings.semantic.embedding_config.model_type=multimodal or an explicit embedding binding with that capability. Remove the rule, set query_modality to text, or change model_type to multimodal (current model_type=%q)",
					rule.Name, raw, modelType))
			}
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
