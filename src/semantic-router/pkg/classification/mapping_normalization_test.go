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

// A file whose maps disagree is what still let a truncated response through:
// LabelCount() sized the distribution from one map while an extra class
// remained resolvable through another, so a top-1 answer passed the
// completeness guard and dropped the missing class. Every one of the four
// accepted maps has to be in scope, not just the two standard ones.
func TestLoadJailbreakMapping_RejectsDisagreeingMaps(t *testing.T) {
	tests := []struct {
		name string
		body string
		want string
	}{
		{
			name: "the two directions name different labels for one index",
			body: `{"label_to_idx":{"benign":0},"idx_to_label":{"0":"jailbreak"}}`,
			want: "index 0 is claimed by both",
		},
		{
			name: "the alternative label->id map disagrees about an index",
			body: `{"label_to_idx":{"benign":0,"jailbreak":1},"label_to_id":{"jailbreak":2}}`,
			want: "inconsistent",
		},
		{
			name: "two labels claim the same index",
			body: `{"label_to_idx":{"benign":0,"jailbreak":0}}`,
			want: "index 0 is claimed by both",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			_, err := LoadJailbreakMapping(writeJailbreakMapping(t, tt.body))
			if err == nil {
				t.Fatal("expected an error for a self-inconsistent mapping")
			}
			if !strings.Contains(err.Error(), tt.want) {
				t.Errorf("error %q should mention %q", err, tt.want)
			}
		})
	}
}

// Maps that merely say different amounts of the same thing are unioned, not
// rejected - and the union is what fixes the truncation, because LabelCount()
// then covers every class any map can resolve. All four maps feed the union,
// including the alternative HuggingFace pair that GetJailbreakTypeFromIndex
// also reads.
func TestLoadJailbreakMapping_UnionsRedundantMaps(t *testing.T) {
	tests := []struct {
		name  string
		body  string
		want  int
		label string
	}{
		{
			name:  "index->label declares a class the label->index map omits",
			body:  `{"label_to_idx":{"benign":0},"idx_to_label":{"0":"benign","1":"jailbreak"}}`,
			want:  2,
			label: "jailbreak",
		},
		{
			name:  "label->index declares a class the index->label map omits",
			body:  `{"label_to_idx":{"benign":0,"jailbreak":1},"idx_to_label":{"0":"benign"}}`,
			want:  2,
			label: "jailbreak",
		},
		{
			name: "the alternative id->label map declares an extra class",
			body: `{"label_to_idx":{"benign":0,"jailbreak":1},"idx_to_label":{"0":"benign","1":"jailbreak"},` +
				`"id_to_label":{"0":"benign","1":"jailbreak","2":"injection"}}`,
			want:  3,
			label: "injection",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			mapping, err := LoadJailbreakMapping(writeJailbreakMapping(t, tt.body))
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if got := mapping.LabelCount(); got != tt.want {
				t.Errorf("LabelCount() = %d, want %d", got, tt.want)
			}
			idx, ok := mapping.IndexForLabel(tt.label)
			if !ok {
				t.Fatalf("IndexForLabel(%q) not found", tt.label)
			}
			// The class now has a slot, so a response omitting it is rejected
			// rather than silently scored 0.0.
			if label, ok := mapping.LabelFromIndex(idx); !ok || label != tt.label {
				t.Errorf("LabelFromIndex(%d) = (%q, %v), want (%q, true)", idx, label, ok, tt.label)
			}
			if _, err := alignScoresToMapping(mapping, []httpClassifyLabelScore{{Label: "benign", Score: 1.0}}); err == nil {
				t.Error("expected a top-1 response to be rejected as incomplete")
			}
		})
	}
}

// alignScoresToMapping allocates exactly LabelCount() slots and indexes them by
// resolved label index, so a sparse or 1-based mapping cannot work. Rejecting it
// at load beats rejecting every response at request time - which under
// on_error: block would block all traffic.
func TestLoadJailbreakMapping_RejectsNonContiguousIndices(t *testing.T) {
	for _, body := range []string{
		`{"idx_to_label":{"0":"benign","5":"jailbreak"}}`,
		`{"idx_to_label":{"1":"benign","2":"jailbreak"}}`,
		`{"label_to_idx":{"benign":1,"jailbreak":2}}`,
	} {
		_, err := LoadJailbreakMapping(writeJailbreakMapping(t, body))
		if err == nil {
			t.Fatalf("expected an error for non-contiguous indices in %s", body)
		}
		if !strings.Contains(err.Error(), "contiguous") {
			t.Errorf("error %q should mention contiguity for %s", err, body)
		}
	}
}

// A non-numeric index key must be named, not silently dropped into a confusing
// count mismatch.
func TestLoadJailbreakMapping_RejectsNonNumericIndexKey(t *testing.T) {
	_, err := LoadJailbreakMapping(writeJailbreakMapping(t, `{"idx_to_label":{"a":"benign","1":"jailbreak"}}`))
	if err == nil {
		t.Fatal("expected an error for a non-numeric index->label key")
	}
	if !strings.Contains(err.Error(), "not a number") {
		t.Errorf("error %q should name the bad key", err)
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
