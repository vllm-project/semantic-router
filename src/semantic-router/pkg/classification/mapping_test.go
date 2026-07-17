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
