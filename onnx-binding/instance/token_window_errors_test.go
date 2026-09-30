//go:build !windows && cgo && (amd64 || arm64)

package instance

import "testing"

func TestTokenWindowErrorsRetainTypedNativeBoundary(t *testing.T) {
	for _, tc := range []struct {
		name    string
		changes map[string]any
		kind    string
	}{
		{
			name: "non_bio_head",
			changes: map[string]any{
				"id2label": map[string]string{"0": "outside", "1": "secret"},
			},
			kind: "configuration",
		},
		{
			name: "graph_class_width_mismatch",
			changes: map[string]any{
				"num_labels": 3,
				"id2label":   map[string]string{"0": "O", "1": "B-SECRET", "2": "I-SECRET"},
			},
			kind: "invalid_output",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			options := specialTokenFixture(t, "token")
			options.MaxInputTokens = 16
			configureSequenceFixture(t, options, tc.changes)
			model, err := LoadTokenClassifier(options)
			if err != nil {
				t.Fatal(err)
			}
			defer model.Close()
			output, err := model.DetectWindows("hello world test", SequenceWindowOptions{Size: 4, Overlap: 1})
			errorKind(t, err, tc.kind)
			if len(output.Spans) != 0 || len(output.Windows) != 0 {
				t.Fatal("failed native call returned partial token spans")
			}
		})
	}
}
