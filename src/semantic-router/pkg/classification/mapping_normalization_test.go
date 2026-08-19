package classification

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func writeJailbreakMapping(t *testing.T, body string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "jailbreak_type_mapping.json")
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatalf("failed to write mapping: %v", err)
	}
	return path
}

// LabelCount() reads only the label->index map, but GetIndexForJailbreakType
// also resolves through the index->label maps. The loader used to normalize
// just one direction, so an index->label-only file left LabelCount() at 0
// while lookups still worked - the disagreement that made
// alignScoresToMapping allocate a wrongly-sized distribution. Both directions
// are now filled in, so LabelCount() is authoritative.
func TestLoadJailbreakMapping_BackfillsLabelToIdxFromIdxToLabel(t *testing.T) {
	path := writeJailbreakMapping(t, `{"idx_to_label":{"0":"benign","1":"jailbreak"}}`)

	mapping, err := LoadJailbreakMapping(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := mapping.LabelCount(); got != 2 {
		t.Errorf("LabelCount() = %d, want 2", got)
	}
	if idx, ok := mapping.LabelToIdx["jailbreak"]; !ok || idx != 1 {
		t.Errorf("LabelToIdx[jailbreak] = (%d, %v), want (1, true)", idx, ok)
	}
}

func TestLoadJailbreakMapping_BackfillsLabelToIdxFromIDToLabel(t *testing.T) {
	path := writeJailbreakMapping(t, `{"id_to_label":{"0":"benign","1":"jailbreak"}}`)

	mapping, err := LoadJailbreakMapping(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got := mapping.LabelCount(); got != 2 {
		t.Errorf("LabelCount() = %d, want 2", got)
	}
	if idx, ok := mapping.IndexForLabel("benign"); !ok || idx != 0 {
		t.Errorf("IndexForLabel(benign) = (%d, %v), want (0, true)", idx, ok)
	}
}

// A file whose two directions disagree is what still let a truncated response
// through: LabelCount() sized the distribution from the smaller map while an
// extra class remained resolvable, so a top-1 answer passed the completeness
// guard and dropped the missing class.
func TestLoadJailbreakMapping_RejectsDisagreeingDirections(t *testing.T) {
	tests := []struct {
		name string
		body string
	}{
		{
			name: "index->label declares a class the label->index map omits",
			body: `{"label_to_idx":{"benign":0},"idx_to_label":{"0":"benign","1":"jailbreak"}}`,
		},
		{
			name: "the two directions name different labels for one index",
			body: `{"label_to_idx":{"benign":0},"idx_to_label":{"0":"jailbreak"}}`,
		},
		{
			name: "label->index declares a class the index->label map omits",
			body: `{"label_to_idx":{"benign":0,"jailbreak":1},"idx_to_label":{"0":"benign"}}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadJailbreakMapping(writeJailbreakMapping(t, tt.body))
			if err == nil {
				t.Fatal("expected an error for a self-inconsistent mapping")
			}
			if !strings.Contains(err.Error(), "inconsistent") {
				t.Errorf("error %q should describe the inconsistency", err)
			}
		})
	}
}

// The shapes the training pipeline and the shipped models actually use must
// keep loading: it writes label_to_id and id_to_label from the same dicts.
func TestLoadJailbreakMapping_AcceptsConsistentShippedShapes(t *testing.T) {
	bodies := []string{
		`{"label_to_id":{"benign":0,"jailbreak":1},"id_to_label":{"0":"benign","1":"jailbreak"}}`,
		`{"label_to_idx":{"benign":0,"jailbreak":1},"idx_to_label":{"0":"benign","1":"jailbreak"}}`,
		`{"label_to_idx":{"benign":0,"jailbreak":1}}`,
	}
	for _, body := range bodies {
		mapping, err := LoadJailbreakMapping(writeJailbreakMapping(t, body))
		if err != nil {
			t.Fatalf("unexpected error for %s: %v", body, err)
		}
		if got := mapping.LabelCount(); got != 2 {
			t.Errorf("LabelCount() = %d for %s, want 2", got, body)
		}
	}
}

// Regression for the silent under-reporting: with a consistent mapping the
// completeness guard sees the true index space, so a top-1 response is
// rejected instead of yielding a truncated distribution whose missing
// positive class scored 0.0.
func TestAlignScoresToMapping_RejectsTruncatedResponseAfterNormalization(t *testing.T) {
	path := writeJailbreakMapping(t, `{"idx_to_label":{"0":"benign","1":"jailbreak"}}`)
	mapping, err := LoadJailbreakMapping(path)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := alignScoresToMapping(mapping, []httpClassifyLabelScore{
		{Label: "benign", Score: 1.0},
	}); err == nil {
		t.Error("expected an error when the response omits a configured label")
	}
}

// The sentinel guard must still see a sentinel that only appears in an
// index->label map.
func TestLoadJailbreakMapping_RejectsSentinelInIndexOnlyMapping(t *testing.T) {
	path := writeJailbreakMapping(t, `{"idx_to_label":{"0":"benign","1":"classification_error"}}`)

	if _, err := LoadJailbreakMapping(path); err == nil {
		t.Error("expected an error when an index-only mapping configures the sentinel label")
	}
}
