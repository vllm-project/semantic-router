package classification

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func writeScratchFile(t *testing.T, dir, name, body string) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
	return path
}

const modelConfigFourClass = `{"id2label":{"0":"SAT","1":"NEED_CLARIFICATION","2":"WRONG_ANSWER","3":"WANT_DIFFERENT"}}`

func TestValidateLabelMappingAgainstModelConfig(t *testing.T) {
	tests := []struct {
		name        string
		modelConfig string
		writeConfig bool
		idxToLabel  map[string]string
		mappingName string
		wantErr     string
	}{
		{
			name:        "agreeing mapping passes",
			modelConfig: modelConfigFourClass,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "SAT", "1": "NEED_CLARIFICATION", "2": "WRONG_ANSWER", "3": "WANT_DIFFERENT"},
			mappingName: "label_mapping.json",
		},
		{
			name:        "case difference is not a disagreement",
			modelConfig: modelConfigFourClass,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "sat", "1": "need_clarification", "2": "wrong_answer", "3": "want_different"},
			mappingName: "label_mapping.json",
		},
		{
			// The runtime normalizes SAT and SATISFIED to the same label, so a
			// sidecar that spells the alias out contradicts nothing.
			name:        "a supported alias is not a disagreement",
			modelConfig: modelConfigFourClass,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "SATISFIED", "1": "NEED_CLARIFICATION", "2": "WRONG_ANSWER", "3": "WANT_DIFFERENT"},
			mappingName: "label_mapping.json",
		},
		{
			name:        "permuted mapping is rejected",
			modelConfig: modelConfigFourClass,
			writeConfig: true,
			// The permutation mmbert32k-feedback-detector-merged ships in its
			// own label_mapping.json.
			idxToLabel:  map[string]string{"0": "NEED_CLARIFICATION", "1": "SAT", "2": "WANT_DIFFERENT", "3": "WRONG_ANSWER"},
			mappingName: "label_mapping.json",
			wantErr:     "disagrees with",
		},
		{
			name:        "missing index is rejected",
			modelConfig: modelConfigFourClass,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "SAT", "1": "NEED_CLARIFICATION", "2": "WRONG_ANSWER"},
			mappingName: "label_mapping.json",
			wantErr:     "has no entry for index",
		},
		{
			name:        "the model config itself is not cross-checked",
			modelConfig: modelConfigFourClass,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "anything"},
			mappingName: "config.json",
		},
		{
			name:        "a model without config.json is not a disagreement",
			writeConfig: false,
			idxToLabel:  map[string]string{"0": "SAT"},
			mappingName: "label_mapping.json",
		},
		{
			name:        "a config without id2label is not a disagreement",
			modelConfig: `{"architectures":["ModernBertForSequenceClassification"]}`,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "SAT"},
			mappingName: "label_mapping.json",
		},
		{
			name:        "an unparsable config is an error, not a skip",
			modelConfig: `{"id2label":`,
			writeConfig: true,
			idxToLabel:  map[string]string{"0": "SAT"},
			mappingName: "label_mapping.json",
			wantErr:     "failed to parse",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			if tc.writeConfig {
				writeScratchFile(t, dir, "config.json", tc.modelConfig)
			}
			mappingPath := filepath.Join(dir, tc.mappingName)
			err := ValidateLabelMappingAgainstModelConfig(mappingPath, dir, tc.idxToLabel)
			switch {
			case tc.wantErr == "" && err != nil:
				t.Fatalf("unexpected error: %v", err)
			case tc.wantErr != "" && err == nil:
				t.Fatalf("expected an error containing %q", tc.wantErr)
			case tc.wantErr != "" && !strings.Contains(err.Error(), tc.wantErr):
				t.Fatalf("error %q does not contain %q", err, tc.wantErr)
			}
		})
	}
}

func TestValidateLabelMappingAgainstModelConfigWithoutModelDir(t *testing.T) {
	if err := ValidateLabelMappingAgainstModelConfig("mapping.json", "", map[string]string{"0": "SAT"}); err != nil {
		t.Fatalf("an unknown model directory has nothing to cross-check: %v", err)
	}
}

// The detector refuses a sidecar mapping that contradicts the model it is
// configured against, instead of silently relabelling every prediction.
func TestFeedbackDetectorRejectsContradictingSidecarMapping(t *testing.T) {
	dir := t.TempDir()
	writeScratchFile(t, dir, "config.json", modelConfigFourClass)
	sidecar := writeScratchFile(t, dir, "label_mapping.json",
		`{"idx_to_label":{"0":"NEED_CLARIFICATION","1":"SAT","2":"WANT_DIFFERENT","3":"WRONG_ANSWER"},`+
			`"label_to_idx":{"NEED_CLARIFICATION":0,"SAT":1,"WANT_DIFFERENT":2,"WRONG_ANSWER":3}}`)

	d, err := NewFeedbackDetector(&config.FeedbackDetectorConfig{
		Enabled: true, ModelID: dir, FeedbackMappingPath: sidecar,
	})
	if err != nil {
		t.Fatalf("NewFeedbackDetector: %v", err)
	}
	err = d.loadMapping(sidecar)
	if err == nil {
		t.Fatal("expected the contradicting sidecar to be rejected")
	}
	if !strings.Contains(err.Error(), "disagrees with") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// The shipped configuration leaves feedback_mapping_path empty and reads the
// model's own config.json, which must keep working.
func TestFeedbackDetectorAcceptsModelConfigMapping(t *testing.T) {
	dir := t.TempDir()
	modelConfig := writeScratchFile(t, dir, "config.json", modelConfigFourClass)

	d, err := NewFeedbackDetector(&config.FeedbackDetectorConfig{Enabled: true, ModelID: dir})
	if err != nil {
		t.Fatalf("NewFeedbackDetector: %v", err)
	}
	if err := d.loadMapping(modelConfig); err != nil {
		t.Fatalf("loading the model's own config.json must work: %v", err)
	}
	if got := d.mapping.IdxToLabel["0"]; got != FeedbackLabelSatisfied {
		t.Fatalf("index 0 resolved to %q, want %q", got, FeedbackLabelSatisfied)
	}
}
