package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
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
// Supports both naming conventions: label_to_idx/idx_to_label and label_to_id/id_to_label
type JailbreakMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
	// Alternative naming (for HuggingFace compatibility)
	LabelToID map[string]int    `json:"label_to_id"`
	IDToLabel map[string]string `json:"id_to_label"`
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
// Supports both label_to_idx/idx_to_label and label_to_id/id_to_label formats
func LoadJailbreakMapping(path string) (*JailbreakMapping, error) {
	// Read the mapping file
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read jailbreak mapping file: %w", err)
	}

	// Parse the JSON data - will populate whichever fields match
	var mapping JailbreakMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse jailbreak mapping JSON: %w", err)
	}

	// If standard fields are empty but alternative fields are populated,
	// copy from alternative fields to standard fields for internal use
	if len(mapping.LabelToIdx) == 0 && len(mapping.LabelToID) > 0 {
		mapping.LabelToIdx = mapping.LabelToID
	}
	if len(mapping.IdxToLabel) == 0 && len(mapping.IDToLabel) > 0 {
		mapping.IdxToLabel = mapping.IDToLabel
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

// stripBIOPrefix removes the BIO sequence labeling prefix from a PII type string.
// For example: "B-PERSON" → "PERSON", "I-DATE_TIME" → "DATE_TIME", "PERSON" → "PERSON".
func stripBIOPrefix(s string) string {
	if len(s) > 2 && s[1] == '-' {
		switch s[0] {
		case 'B', 'I', 'E':
			return s[2:]
		}
	}
	return s
}

// TranslatePIIType translates a PII type from Rust binding format to named type.
// Handles formats like "class_6" → "DATE_TIME" and passes through already-named types.
// Also strips BIO prefixes (B-PERSON → PERSON, I-DATE_TIME → DATE_TIME).
// This includes BIO prefixes that may be embedded in the mapping file's label values.
func (pm *PIIMapping) TranslatePIIType(rawType string) string {
	// Strip BIO prefix unconditionally — must happen BEFORE the nil guard so
	// that "B-PERSON" → "PERSON" even when no mapping file is loaded.
	normalized := stripBIOPrefix(rawType)

	if pm == nil {
		return normalized
	}

	// Check if it's already a known label (exact match in IdxToLabel values,
	// comparing after stripping BIO from both sides).
	for _, label := range pm.IdxToLabel {
		if normalized == stripBIOPrefix(label) {
			return normalized
		}
	}

	// Check if it's in class_X format
	if len(normalized) > 6 && normalized[:6] == "class_" {
		indexStr := normalized[6:]
		if label, ok := pm.IdxToLabel[indexStr]; ok {
			// Strip BIO prefix from the mapped label: mapping files may store
			// BIO-tagged values like "I-PERSON" rather than bare "PERSON".
			return stripBIOPrefix(label)
		}
	}

	// Check if it's in LABEL_X format (from Rust binding)
	if len(normalized) > 6 && normalized[:6] == "LABEL_" {
		indexStr := normalized[6:]
		if label, ok := pm.IdxToLabel[indexStr]; ok {
			// Strip BIO prefix from the mapped label: mapping files may store
			// BIO-tagged values like "I-PERSON" rather than bare "PERSON".
			return stripBIOPrefix(label)
		}
	}

	return normalized
}

// GetJailbreakTypeFromIndex converts a class index to jailbreak type name using the mapping
// Supports both idx_to_label and id_to_label field names
func (jm *JailbreakMapping) GetJailbreakTypeFromIndex(classIndex int) (string, bool) {
	indexStr := fmt.Sprintf("%d", classIndex)

	// Try standard field first
	if jailbreakType, ok := jm.IdxToLabel[indexStr]; ok {
		return jailbreakType, true
	}

	// Fall back to alternative field
	jailbreakType, ok := jm.IDToLabel[indexStr]
	return jailbreakType, ok
}

// GetIndexForJailbreakType converts a jailbreak type name to its class index using
// the mapping. It is the inverse of GetJailbreakTypeFromIndex and supports both the
// label_to_idx/idx_to_label and label_to_id/id_to_label naming conventions, falling
// back to a reverse scan of the index->label maps when the label->index maps are absent.
func (jm *JailbreakMapping) GetIndexForJailbreakType(label string) (int, bool) {
	if idx, ok := jm.LabelToIdx[label]; ok {
		return idx, true
	}
	if idx, ok := jm.LabelToID[label]; ok {
		return idx, true
	}
	if idx, ok := reverseLookupIndex(jm.IdxToLabel, label); ok {
		return idx, true
	}
	return reverseLookupIndex(jm.IDToLabel, label)
}

// reverseLookupIndex scans an index->label map for the given label and returns its
// numeric index.
func reverseLookupIndex(idxToLabel map[string]string, label string) (int, bool) {
	for indexStr, mapped := range idxToLabel {
		if mapped == label {
			if idx, err := strconv.Atoi(indexStr); err == nil {
				return idx, true
			}
		}
	}
	return 0, false
}

// GetCategoryCount returns the number of categories in the mapping
func (cm *CategoryMapping) GetCategoryCount() int {
	return len(cm.CategoryToIdx)
}

// IndexForLabel satisfies sequenceLabelMapping (http_classifier.go) for
// CategoryMapping, letting a category http_classify backend reuse the same
// label-alignment validator jailbreak already uses.
func (cm *CategoryMapping) IndexForLabel(label string) (int, bool) {
	idx, ok := cm.CategoryToIdx[label]
	return idx, ok
}

// LabelFromIndex satisfies sequenceLabelMapping for CategoryMapping.
func (cm *CategoryMapping) LabelFromIndex(classIndex int) (string, bool) {
	return cm.GetCategoryFromIndex(classIndex)
}

// LabelCount satisfies sequenceLabelMapping for CategoryMapping.
func (cm *CategoryMapping) LabelCount() int {
	return cm.GetCategoryCount()
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
// Supports both label_to_idx and label_to_id field names
func (jm *JailbreakMapping) GetJailbreakTypeCount() int {
	// Try standard field first
	if len(jm.LabelToIdx) > 0 {
		return len(jm.LabelToIdx)
	}
	// Fall back to alternative field
	return len(jm.LabelToID)
}

// IndexForLabel satisfies sequenceLabelMapping (http_classifier.go) for
// JailbreakMapping by delegating to the existing jailbreak-named lookup.
func (jm *JailbreakMapping) IndexForLabel(label string) (int, bool) {
	return jm.GetIndexForJailbreakType(label)
}

// LabelFromIndex satisfies sequenceLabelMapping for JailbreakMapping.
func (jm *JailbreakMapping) LabelFromIndex(classIndex int) (string, bool) {
	return jm.GetJailbreakTypeFromIndex(classIndex)
}

// LabelCount satisfies sequenceLabelMapping for JailbreakMapping.
func (jm *JailbreakMapping) LabelCount() int {
	return jm.GetJailbreakTypeCount()
}

// resolveSinglePositiveIndex resolves the configured positive_labels against
// mapping down to a single class index, for backends (like http_chat) that
// only ever produce one binary verdict and cannot represent more than one
// positive class. Unlike the multi-label backends (candle, http_classify),
// which sum every positive label's independent probability, a label here
// that isn't found in mapping is skipped rather than treated as an error -
// matching the general "at least one configured label must exist" leniency
// enforced elsewhere (validateJailbreakPositiveLabels) - but if two or more
// configured labels resolve to genuinely different class indices, that's a
// real misconfiguration for a binary-verdict backend, not a benign no-op,
// and is rejected rather than silently scored against only the first one.
func resolveSinglePositiveIndex(mapping *JailbreakMapping, positiveLabels []string) (int, error) {
	resolvedIdx := -1
	for _, label := range resolvePositiveLabels(positiveLabels) {
		idx, ok := mapping.GetIndexForJailbreakType(label)
		if !ok {
			continue
		}
		if resolvedIdx == -1 {
			resolvedIdx = idx
			continue
		}
		if idx != resolvedIdx {
			return 0, fmt.Errorf(
				"configured positive_labels %v resolve to more than one class index in jailbreak_mapping (%d and %d); "+
					"this backend produces a single binary verdict and cannot treat more than one class as positive",
				positiveLabels, resolvedIdx, idx)
		}
	}
	if resolvedIdx == -1 {
		return 0, fmt.Errorf("none of the configured positive_labels %v were found in jailbreak_mapping", positiveLabels)
	}
	return resolvedIdx, nil
}
