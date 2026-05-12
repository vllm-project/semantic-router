package config

import (
	"fmt"
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
	return validateEmbeddingRuleModalities(cfg.EmbeddingRules, cfg.EmbeddingModels.EmbeddingConfig.ModelType)
}

// ValidateEmbeddingContracts is the exported counterpart of the private
// validateEmbeddingContracts function. It exists so packages outside pkg/config
// (notably the reconciler in pkg/k8s) can apply the embedding-modality
// contract directly, since validateConfigStructure short-circuits on
// ConfigSource == ConfigSourceKubernetes and would otherwise let CRD-loaded
// configs skip a contract that file-loaded configs are held to.
//
// The signature mirrors the private function 1:1 and the call site reads
// ValidateEmbeddingContracts(cfg) so the contract has a single canonical
// shape regardless of source.
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
