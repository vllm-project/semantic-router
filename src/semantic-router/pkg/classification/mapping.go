package classification

import (
	"encoding/json"
	"fmt"
	"os"
)

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx         map[string]int    `json:"category_to_idx"`
	IdxToCategory         map[string]string `json:"idx_to_category"`
	CategorySystemPrompts map[string]string `json:"category_system_prompts,omitempty"` // Optional per-category system prompts from MCP server
	CategoryDescriptions  map[string]string `json:"category_descriptions,omitempty"`   // Optional category descriptions
}

// PIIMapping holds the mapping between indices and PII types
type PIIMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
}

// JailbreakMapping holds the mapping between indices and jailbreak types
type JailbreakMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
}

// LoadCategoryMapping loads the category mapping from a JSON file
func LoadCategoryMapping(path string) (*CategoryMapping, error) {
	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read mapping file: %w", err)
	}

	// Parse the JSON data
	var mapping CategoryMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse mapping JSON: %w", err)
	}

	return &mapping, nil
}

// LoadPIIMapping loads the PII mapping from a JSON file
func LoadPIIMapping(path string) (*PIIMapping, error) {
	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read PII mapping file: %w", err)
	}

	// Parse the JSON data
	var mapping PIIMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse PII mapping JSON: %w", err)
	}

	return &mapping, nil
}

// LoadJailbreakMapping loads the jailbreak mapping from a JSON file
func LoadJailbreakMapping(path string) (*JailbreakMapping, error) {
	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read jailbreak mapping file: %w", err)
	}

	// Parse the JSON data
	var mapping JailbreakMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse jailbreak mapping JSON: %w", err)
	}

	return &mapping, nil
}

// GetCategoryFromIndex converts a class index to category name using the mapping
func (cm *CategoryMapping) GetCategoryFromIndex(classIndex int) (string, bool) {
	categoryName, ok := cm.IdxToCategory[fmt.Sprintf("%d", classIndex)]
	return categoryName, ok
}

// GetPIITypeFromIndex converts a class index to PII type name using the mapping
func (pm *PIIMapping) GetPIITypeFromIndex(classIndex int) (string, bool) {
	piiType, ok := pm.IdxToLabel[fmt.Sprintf("%d", classIndex)]
	return piiType, ok
}

// TranslatePIIType translates a PII type from Rust binding format to named type.
// Handles formats like "class_6" → "DATE_TIME" and passes through already-named types.
// Also strips BIO prefixes (B-PERSON → PERSON).
func (pm *PIIMapping) TranslatePIIType(rawType string) string {
	if pm == nil {
		return rawType
	}

	// Check if it's already a known label (exact match or in IdxToLabel values)
	for _, label := range pm.IdxToLabel {
		if rawType == label {
			return rawType // Already a proper label name
		}
	}

	// Check if it's in class_X format
	if len(rawType) > 6 && rawType[:6] == "class_" {
		indexStr := rawType[6:]
		if label, ok := pm.IdxToLabel[indexStr]; ok {
			return label
		}
	}

	// Check if it's in LABEL_X format (from Rust binding)
	if len(rawType) > 6 && rawType[:6] == "LABEL_" {
		indexStr := rawType[6:]
		if label, ok := pm.IdxToLabel[indexStr]; ok {
			return label
		}
	}

	// Strip BIO prefix if present (B-PERSON → PERSON, I-DATE_TIME → DATE_TIME)
	if len(rawType) > 2 && rawType[1] == '-' {
		prefix := rawType[0]
		if prefix == 'B' || prefix == 'I' || prefix == 'O' || prefix == 'E' {
			return rawType[2:]
		}
	}

	return rawType
}

// GetJailbreakTypeFromIndex converts a class index to jailbreak type name using the mapping
func (jm *JailbreakMapping) GetJailbreakTypeFromIndex(classIndex int) (string, bool) {
	jailbreakType, ok := jm.IdxToLabel[fmt.Sprintf("%d", classIndex)]
	return jailbreakType, ok
}

// GetCategoryCount returns the number of categories in the mapping
func (cm *CategoryMapping) GetCategoryCount() int {
	return len(cm.CategoryToIdx)
}

// GetCategorySystemPrompt returns the system prompt for a specific category if available
func (cm *CategoryMapping) GetCategorySystemPrompt(category string) (string, bool) {
	if cm.CategorySystemPrompts == nil {
		return "", false
	}
	prompt, ok := cm.CategorySystemPrompts[category]
	return prompt, ok
}

// GetCategoryDescription returns the description for a given category
func (cm *CategoryMapping) GetCategoryDescription(category string) (string, bool) {
	if cm.CategoryDescriptions == nil {
		return "", false
	}
	desc, ok := cm.CategoryDescriptions[category]
	return desc, ok
}

// GetPIITypeCount returns the number of PII types in the mapping
func (pm *PIIMapping) GetPIITypeCount() int {
	return len(pm.LabelToIdx)
}

// GetJailbreakTypeCount returns the number of jailbreak types in the mapping
func (jm *JailbreakMapping) GetJailbreakTypeCount() int {
	return len(jm.LabelToIdx)
}
