package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestExplicitCategoryVariantsUseMatchingInference(t *testing.T) {
	tests := []struct {
		name    string
		variant string
	}{
		{name: "modernbert", variant: config.CategoryVariantModernBERT},
		{name: "mmbert32k", variant: config.CategoryVariantMmBERT32K},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			_, got := categoryDependenciesForVariant(test.variant)
			switch test.variant {
			case config.CategoryVariantModernBERT:
				if _, ok := got.(ModernBERTCategoryInferenceImpl); !ok {
					t.Fatalf("inference = %T, want explicit ModernBERT inference", got)
				}
			case config.CategoryVariantMmBERT32K:
				if _, ok := got.(*MmBERT32KCategoryInferenceImpl); !ok {
					t.Fatalf("inference = %T, want mmBERT-32K inference", got)
				}
			}
		})
	}
}
