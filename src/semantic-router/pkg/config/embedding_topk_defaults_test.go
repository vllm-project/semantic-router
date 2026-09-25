package config

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestEmbeddingTopKDefaultsPreserveAllAcceptedSignals(t *testing.T) {
	for _, tc := range []struct {
		name  string
		input *int
		want  int
	}{
		{name: "omitted"},
		{name: "explicit_zero", input: canonicalIntPtr(0)},
		{name: "explicit_one", input: canonicalIntPtr(1), want: 1},
		{name: "explicit_three", input: canonicalIntPtr(3), want: 3},
		{name: "negative_fallback", input: canonicalIntPtr(-1)},
	} {
		t.Run(tc.name, func(t *testing.T) {
			input := HNSWConfig{TopK: tc.input}
			before := 0
			if tc.input != nil {
				before = *tc.input
			}
			got := input.WithDefaults()
			require.NotNil(t, got.TopK)
			require.Equal(t, tc.want, *got.TopK)
			if tc.input == nil {
				require.Nil(t, input.TopK)
			} else {
				require.Equal(t, before, *input.TopK, "defaulting must not modify the caller's setting")
			}
		})
	}
	canonical := DefaultGlobalConfig()
	require.NotNil(t, canonical.EmbeddingConfig.TopK)
	require.Zero(t, *canonical.EmbeddingConfig.TopK)
}
