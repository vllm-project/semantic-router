package k8s

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestConvertSignals_EmbeddingQueryModality covers the converter pass-through
// of the QueryModality field added to v1alpha1.EmbeddingSignal. The runtime
// config.EmbeddingRule has supported QueryModality since the multimodal
// query-signals work in #1867; this test fixes the CRD->runtime translation
// so an IntelligentRoute can declare image/audio modality rules.
func TestConvertSignals_EmbeddingQueryModality(t *testing.T) {
	cases := []struct {
		name     string
		input    string
		expected config.QueryModality
	}{
		{
			name:     "OmittedDefaultsToEmpty",
			input:    "",
			expected: "", // EffectiveQueryModality() resolves "" to QueryModalityText downstream
		},
		{
			name:     "ExplicitText",
			input:    "text",
			expected: config.QueryModalityText,
		},
		{
			name:     "Image",
			input:    "image",
			expected: config.QueryModalityImage,
		},
		{
			name:     "Audio",
			input:    "audio",
			expected: config.QueryModalityAudio,
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			signals := v1alpha1.Signals{
				Embeddings: []v1alpha1.EmbeddingSignal{
					{
						Name:              "rule-under-test",
						Threshold:         0.75,
						Candidates:        []string{"anchor phrase"},
						AggregationMethod: "max",
						QueryModality:     tc.input,
					},
				},
			}

			got := convertSignals(signals)

			assert.Len(t, got.Embeddings, 1, "exactly one embedding rule should be converted")
			assert.Equal(t, tc.expected, got.Embeddings[0].QueryModality,
				"QueryModality should round-trip from CRD signal to canonical EmbeddingRule")
		})
	}
}

// TestConvertSignals_EmbeddingQueryModality_EffectiveDefault verifies that
// the omitted-modality case still resolves to text via the runtime helper,
// preserving backward compatibility for rules authored before the field
// existed.
func TestConvertSignals_EmbeddingQueryModality_EffectiveDefault(t *testing.T) {
	signals := v1alpha1.Signals{
		Embeddings: []v1alpha1.EmbeddingSignal{{
			Name:       "legacy-rule",
			Threshold:  0.7,
			Candidates: []string{"anchor"},
			// QueryModality intentionally unset to mirror legacy CRDs
		}},
	}

	got := convertSignals(signals)

	assert.Equal(t, config.QueryModalityText, got.Embeddings[0].EffectiveQueryModality(),
		"EffectiveQueryModality of an unset CRD field should resolve to text")
}
