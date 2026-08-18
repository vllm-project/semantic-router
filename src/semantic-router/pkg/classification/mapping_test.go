package classification

import "testing"

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
