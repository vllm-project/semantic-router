package classification

import (
	"os"
	"path/filepath"
	"testing"
)

func TestGetIndexForJailbreakType(t *testing.T) {
	tests := []struct {
		name      string
		mapping   JailbreakMapping
		label     string
		wantIndex int
		wantOK    bool
	}{
		{
			name:      "label_to_idx form",
			mapping:   JailbreakMapping{LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1}},
			label:     "jailbreak",
			wantIndex: 1,
			wantOK:    true,
		},
		{
			name:      "alternative label_to_id form",
			mapping:   JailbreakMapping{LabelToID: map[string]int{"benign": 0, "jailbreak": 1}},
			label:     "jailbreak",
			wantIndex: 1,
			wantOK:    true,
		},
		{
			name:      "reverse lookup from idx_to_label when label maps are absent",
			mapping:   JailbreakMapping{IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"}},
			label:     "jailbreak",
			wantIndex: 1,
			wantOK:    true,
		},
		{
			name:      "reverse lookup from id_to_label alternative form",
			mapping:   JailbreakMapping{IDToLabel: map[string]string{"0": "benign", "2": "jailbreak"}},
			label:     "jailbreak",
			wantIndex: 2,
			wantOK:    true,
		},
		{
			name:      "label not present",
			mapping:   JailbreakMapping{LabelToIdx: map[string]int{"benign": 0}},
			label:     "jailbreak",
			wantIndex: 0,
			wantOK:    false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotIndex, gotOK := tt.mapping.GetIndexForJailbreakType(tt.label)
			if gotOK != tt.wantOK {
				t.Fatalf("GetIndexForJailbreakType(%q) ok = %v, want %v", tt.label, gotOK, tt.wantOK)
			}
			if gotOK && gotIndex != tt.wantIndex {
				t.Errorf("GetIndexForJailbreakType(%q) index = %d, want %d", tt.label, gotIndex, tt.wantIndex)
			}
		})
	}
}

// TestJailbreakMapping_SequenceLabelMappingMethods guards the thin
// sequenceLabelMapping wrapper methods (http_classifier.go) against silently
// delegating to the wrong underlying jailbreak-named method.
func TestJailbreakMapping_SequenceLabelMappingMethods(t *testing.T) {
	mapping := &JailbreakMapping{
		LabelToIdx: map[string]int{"benign": 0, "jailbreak": 1},
		IdxToLabel: map[string]string{"0": "benign", "1": "jailbreak"},
	}

	if idx, ok := mapping.IndexForLabel("jailbreak"); !ok || idx != 1 {
		t.Errorf("IndexForLabel(%q) = (%d, %v), want (1, true)", "jailbreak", idx, ok)
	}
	if label, ok := mapping.LabelFromIndex(0); !ok || label != "benign" {
		t.Errorf("LabelFromIndex(0) = (%q, %v), want (%q, true)", label, ok, "benign")
	}
	if count := mapping.LabelCount(); count != 2 {
		t.Errorf("LabelCount() = %d, want 2", count)
	}
}

// TestCategoryMapping_SequenceLabelMappingMethods guards CategoryMapping's
// sequenceLabelMapping wrapper methods, added so a category http_classify
// backend (#2760) can reuse alignScoresToMapping/assignScoreToMapping.
func TestCategoryMapping_SequenceLabelMappingMethods(t *testing.T) {
	mapping := &CategoryMapping{
		CategoryToIdx: map[string]int{"business": 0, "law": 1},
		IdxToCategory: map[string]string{"0": "business", "1": "law"},
	}

	if idx, ok := mapping.IndexForLabel("law"); !ok || idx != 1 {
		t.Errorf("IndexForLabel(%q) = (%d, %v), want (1, true)", "law", idx, ok)
	}
	if _, ok := mapping.IndexForLabel("unknown"); ok {
		t.Error("IndexForLabel(\"unknown\") ok = true, want false")
	}
	if label, ok := mapping.LabelFromIndex(0); !ok || label != "business" {
		t.Errorf("LabelFromIndex(0) = (%q, %v), want (%q, true)", label, ok, "business")
	}
	if count := mapping.LabelCount(); count != 2 {
		t.Errorf("LabelCount() = %d, want 2", count)
	}
}

func TestLoadCategoryMappingRejectsNonBijections(t *testing.T) {
	valid := `{"category_to_idx":{"math":0,"law":1},"idx_to_category":{"0":"math","1":"law"}}`
	tests := map[string]string{
		"mismatched reverse label": `{"category_to_idx":{"math":0,"law":1},"idx_to_category":{"0":"law","1":"math"}}`,
		"duplicate index":          `{"category_to_idx":{"math":0,"law":0},"idx_to_category":{"0":"math","1":"law"}}`,
		"missing reverse entry":    `{"category_to_idx":{"math":0,"law":1},"idx_to_category":{"0":"math"}}`,
		"non-contiguous index":     `{"category_to_idx":{"math":0,"law":2},"idx_to_category":{"0":"math","2":"law"}}`,
		"non-numeric reverse key":  `{"category_to_idx":{"math":0},"idx_to_category":{"zero":"math"}}`,
	}
	for name, data := range tests {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "category_mapping.json")
			if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
				t.Fatal(err)
			}
			if _, err := LoadCategoryMapping(path); err == nil {
				t.Fatal("expected invalid category mapping to be rejected")
			}
		})
	}
	path := filepath.Join(t.TempDir(), "category_mapping.json")
	if err := os.WriteFile(path, []byte(valid), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCategoryMapping(path); err != nil {
		t.Fatalf("valid category mapping rejected: %v", err)
	}
}

func TestLoadCategoryMappingAcceptsClassificationErrorLabel(t *testing.T) {
	path := filepath.Join(t.TempDir(), "category_mapping.json")
	data := `{"category_to_idx":{"math":0,"classification_error":1},"idx_to_category":{"0":"math","1":"classification_error"}}`
	if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadCategoryMapping(path); err != nil {
		t.Fatalf("classification_error category label rejected: %v", err)
	}
}

// TestLoadJailbreakMapping_RejectsSentinelCollision guards on_error: block's
// fail-closed sentinel (JailbreakClassificationErrorType): a deployment
// whose mapping file configures a real label with that exact name would
// make a genuine detection indistinguishable from a classify failure - see
// @adaamko's review on #2918/#2930. LoadJailbreakMapping must reject it
// instead of loading it silently.
//
// Every supported mapping shape is covered on purpose. An earlier version of
// this guard indexed LabelToIdx directly, which caught the two shapes that
// populate a label->index map (label_to_idx, and label_to_id via the loader's
// back-fill) but missed the two that declare only an index->label map: those
// left LabelToIdx empty and still resolved the sentinel through
// GetJailbreakTypeFromIndex at runtime.
//
// Note the shipped mmbert32k mapping declares both directions, so it was
// already covered; only a hand-written index->label-only mapping could reach
// the gap.
func TestLoadJailbreakMapping_RejectsSentinelCollision(t *testing.T) {
	for name, data := range map[string]string{
		"label_to_idx_and_idx_to_label": `{"label_to_idx":{"benign":0,"classification_error":1},"idx_to_label":{"0":"benign","1":"classification_error"}}`,
		"idx_to_label_only":             `{"idx_to_label":{"0":"benign","1":"classification_error"}}`,
		"id_to_label_only_huggingface":  `{"id_to_label":{"0":"benign","1":"classification_error"}}`,
		"label_to_id_only_huggingface":  `{"label_to_id":{"benign":0,"classification_error":1}}`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "jailbreak_mapping.json")
			if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
				t.Fatalf("failed to write test mapping file: %v", err)
			}

			mapping, err := LoadJailbreakMapping(path)
			if err != nil {
				return
			}
			resolved, _ := mapping.GetJailbreakTypeFromIndex(1)
			t.Errorf("expected an error when a configured label collides with the on_error: block sentinel, "+
				"but the mapping loaded and index 1 resolves to %q", resolved)
		})
	}
}

// TestLoadJailbreakMapping_AllowsOrdinaryLabels guards against the collision
// check rejecting every mapping by mistake - in every supported shape, so
// widening the guard above cannot silently start rejecting real mappings
// (notably the shipped label_to_id/id_to_label one).
func TestLoadJailbreakMapping_AllowsOrdinaryLabels(t *testing.T) {
	for name, data := range map[string]string{
		"label_to_idx_and_idx_to_label": `{"label_to_idx":{"benign":0,"jailbreak":1},"idx_to_label":{"0":"benign","1":"jailbreak"}}`,
		"idx_to_label_only":             `{"idx_to_label":{"0":"benign","1":"jailbreak"}}`,
		"shipped_huggingface_shape":     `{"label_to_id":{"benign":0,"jailbreak":1},"id_to_label":{"0":"benign","1":"jailbreak"}}`,
	} {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join(t.TempDir(), "jailbreak_mapping.json")
			if err := os.WriteFile(path, []byte(data), 0o600); err != nil {
				t.Fatalf("failed to write test mapping file: %v", err)
			}

			if _, err := LoadJailbreakMapping(path); err != nil {
				t.Errorf("unexpected error loading an ordinary mapping: %v", err)
			}
		})
	}
}
