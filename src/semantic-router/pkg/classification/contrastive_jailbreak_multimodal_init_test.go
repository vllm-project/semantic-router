package classification

import (
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/embedding"
)

func multimodalJailbreakBuilder(rules []config.JailbreakRule) *classifierOptionBuilder {
	cfg := &config.RouterConfig{}
	cfg.RoutingScope = config.DefaultRecipeName
	cfg.EmbeddingConfig.ModelType = "multimodal"
	cfg.JailbreakRules = rules
	builder := newClassifierOptionBuilder(cfg, nil)
	builder.embeddingSet = embedding.NewSet(nil, "multimodal")
	return builder
}

func TestContrastiveJailbreakSkipsMultimodalInitWithoutContrastiveRules(t *testing.T) {
	for name, rules := range map[string][]config.JailbreakRule{
		"no rules":              nil,
		"classifier rules only": {{Name: "guard", Method: "classifier", Threshold: .5}},
	} {
		t.Run(name, func(t *testing.T) {
			builder := multimodalJailbreakBuilder(rules)
			apply, err := builder.buildContrastiveJailbreakClassifiersOption()
			if err != nil {
				t.Fatalf("recipe without contrastive rules must not need the multimodal embedding: %v", err)
			}
			if apply != nil {
				t.Fatal("no contrastive classifiers should be registered")
			}
		})
	}
}

func TestContrastiveJailbreakStillRequiresMultimodalEmbeddingForContrastiveRules(t *testing.T) {
	builder := multimodalJailbreakBuilder([]config.JailbreakRule{{Name: "contrast", Method: "contrastive", Threshold: .5, JailbreakPatterns: []string{"ignore"}, BenignPatterns: []string{"hello"}}})
	_, err := builder.buildContrastiveJailbreakClassifiersOption()
	if err == nil || !strings.Contains(err.Error(), `embedding model "multimodal" was not prepared`) {
		t.Fatalf("expected unprepared multimodal embedding error, got %v", err)
	}
}
