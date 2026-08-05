package classification

import (
	"os"
	"path/filepath"
	"testing"
)

func writeModelConfig(t *testing.T, body string) string {
	t.Helper()
	dir := t.TempDir()
	if body != "" {
		if err := os.WriteFile(filepath.Join(dir, "config.json"), []byte(body), 0o644); err != nil {
			t.Fatalf("write config.json: %v", err)
		}
	}
	return dir
}

func TestIsModernBertModel(t *testing.T) {
	tests := []struct {
		name string
		body string
		want bool
	}{
		{
			name: "model_type modernbert",
			body: `{"model_type":"modernbert","position_embedding_type":"sans_pos"}`,
			want: true,
		},
		{
			name: "model_type uppercase",
			body: `{"model_type":"ModernBERT"}`,
			want: true,
		},
		{
			name: "architectures sequence classification",
			body: `{"architectures":["ModernBertForSequenceClassification"]}`,
			want: true,
		},
		{
			name: "architectures token classification",
			body: `{"architectures":["ModernBertForTokenClassification"]}`,
			want: true,
		},
		{
			name: "traditional bert",
			body: `{"model_type":"bert","hidden_act":"gelu","architectures":["BertForSequenceClassification"]}`,
			want: false,
		},
		{
			name: "roberta",
			body: `{"model_type":"roberta"}`,
			want: false,
		},
		{
			name: "empty config",
			body: `{}`,
			want: false,
		},
		{
			name: "malformed json",
			body: `{not json`,
			want: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := writeModelConfig(t, tc.body)
			if got := isModernBertModel(dir); got != tc.want {
				t.Errorf("isModernBertModel(%q) = %v, want %v", tc.body, got, tc.want)
			}
		})
	}
}

func TestIsModernBertModelMissingConfig(t *testing.T) {
	// No config.json written: detection must return false, not panic, so callers
	// fall back to the existing auto-detect-first ordering.
	dir := writeModelConfig(t, "")
	if isModernBertModel(dir) {
		t.Errorf("isModernBertModel with no config.json = true, want false")
	}
	if isModernBertModel(filepath.Join(dir, "does-not-exist")) {
		t.Errorf("isModernBertModel on nonexistent path = true, want false")
	}
}
