package mcpserver

import (
	"bytes"
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// cloneConfig performs a deep copy via YAML round-trip.
func cloneConfig(in *config.RouterConfig) (*config.RouterConfig, error) {
	if in == nil {
		return nil, fmt.Errorf("config is nil")
	}
	b, err := yaml.Marshal(in)
	if err != nil {
		return nil, err
	}
	var out config.RouterConfig
	dec := yaml.NewDecoder(bytes.NewReader(b))
	dec.KnownFields(true)
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func parseYAMLStrict[T any](yamlStr string) (*T, error) {
	var out T
	dec := yaml.NewDecoder(bytes.NewBufferString(yamlStr))
	dec.KnownFields(true)
	if err := dec.Decode(&out); err != nil {
		return nil, err
	}
	return &out, nil
}

func persistConfig(path string, cfg *config.RouterConfig) error {
	if path == "" {
		return nil
	}
	b, err := yaml.Marshal(cfg)
	if err != nil {
		return err
	}
	return os.WriteFile(path, b, 0o644)
}

// Supported entity kinds.
const (
	KindDecision         = "decision"
	KindKeywordRule      = "keyword_rule"
	KindEmbeddingRule    = "embedding_rule"
	KindFactCheckRule    = "fact_check_rule"
	KindUserFeedbackRule = "user_feedback_rule"
	KindPreferenceRule   = "preference_rule"
	KindLanguageRule     = "language_rule"
	KindContextRule      = "context_rule"
	KindLatencyRule      = "latency_rule"
	KindComplexityRule   = "complexity_rule"
	KindModel            = "model"
)

func supportedKinds() []string {
	return []string{
		KindDecision,
		KindKeywordRule,
		KindEmbeddingRule,
		KindFactCheckRule,
		KindUserFeedbackRule,
		KindPreferenceRule,
		KindLanguageRule,
		KindContextRule,
		KindLatencyRule,
		KindComplexityRule,
		KindModel,
	}
}
