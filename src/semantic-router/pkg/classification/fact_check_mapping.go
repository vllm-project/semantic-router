package classification

import (
	"encoding/json"
	"fmt"
	"os"
)

// FactCheckLabel constants for fact-check classification labels
const (
	FactCheckLabelNeeded    = "FACT_CHECK_NEEDED"
	FactCheckLabelNotNeeded = "NO_FACT_CHECK_NEEDED"
)

// FactCheckMapping holds the mapping between indices and fact-check labels
type FactCheckMapping struct {
	LabelToIdx  map[string]int    `json:"label_to_idx"`
	IdxToLabel  map[string]string `json:"idx_to_label"`
	Description map[string]string `json:"description,omitempty"`
}

// LoadFactCheckMapping loads the fact-check mapping from a JSON file
func LoadFactCheckMapping(path string) (*FactCheckMapping, error) {
	if path == "" {
		return nil, fmt.Errorf("fact-check mapping path is empty")
	}

	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read fact-check mapping file: %w", err)
	}

	// Parse the JSON data
	var mapping FactCheckMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse fact-check mapping JSON: %w", err)
	}

	// Validate the mapping has required labels
	if len(mapping.LabelToIdx) < 2 {
		return nil, fmt.Errorf("fact-check mapping must have at least 2 labels, got %d", len(mapping.LabelToIdx))
	}

	return &mapping, nil
}

// GetLabelFromIndex converts a class index to fact-check label using the mapping
func (m *FactCheckMapping) GetLabelFromIndex(classIndex int) (string, bool) {
	label, ok := m.IdxToLabel[fmt.Sprintf("%d", classIndex)]
	return label, ok
}

// GetIndexFromLabel converts a fact-check label to class index
func (m *FactCheckMapping) GetIndexFromLabel(label string) (int, bool) {
	idx, ok := m.LabelToIdx[label]
	return idx, ok
}

// GetLabelCount returns the number of labels in the mapping
func (m *FactCheckMapping) GetLabelCount() int {
	return len(m.LabelToIdx)
}

// IsFactCheckNeeded returns true if the label indicates fact-check is needed
func (m *FactCheckMapping) IsFactCheckNeeded(label string) bool {
	return label == FactCheckLabelNeeded
}

// GetDescription returns the description for a label if available
func (m *FactCheckMapping) GetDescription(label string) (string, bool) {
	if m.Description == nil {
		return "", false
	}
	desc, ok := m.Description[label]
	return desc, ok
}
