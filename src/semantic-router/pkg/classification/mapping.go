package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
)

// complexityDifficultyLabels is the set of difficulty labels the complexity
// signal can emit. The runtime consumers — classifyComplexityDifficulty in
// complexity_rule_scoring.go and the neutral-band/margin logic in
// classifier_signal_complexity.go — hardcode exactly these lowercase values, so
// a trained-model mapping that emits anything else (a different case, a synonym)
// produces a signal the decision schema can never reference. Keep in sync with
// those consumers and with complexityDifficultyLevels in
// pkg/config/validator_complexity.go.
var complexityDifficultyLabels = map[string]struct{}{
	"easy":   {},
	"medium": {},
	"hard":   {},
}

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

// ComplexityMapping holds the mapping between class indices and difficulty labels
// (e.g. easy/medium/hard) for the trained complexity classifier. It accepts several
// common naming conventions so a model's existing mapping file can be used directly:
//   - label_to_idx / idx_to_label (router convention)
//   - label_to_id / id_to_label (HuggingFace config convention)
//   - category_to_idx / idx_to_category (category-classifier mapping convention,
//     e.g. category_mapping.json shipped alongside merged classifier checkpoints)
type ComplexityMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"`
	IdxToLabel map[string]string `json:"idx_to_label"`
	// Alternative naming (for HuggingFace compatibility)
	LabelToID map[string]int    `json:"label_to_id"`
	IDToLabel map[string]string `json:"id_to_label"`
	// Alternative naming (category-classifier mapping convention)
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
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

// LoadComplexityMapping loads the complexity difficulty mapping from a JSON file.
// Supports both label_to_idx/idx_to_label and label_to_id/id_to_label formats.
func LoadComplexityMapping(path string) (*ComplexityMapping, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read complexity mapping file: %w", err)
	}

	var mapping ComplexityMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse complexity mapping JSON: %w", err)
	}

	// If the canonical fields are empty but an alternative naming convention is
	// populated, copy it into the canonical fields for internal use.
	if len(mapping.LabelToIdx) == 0 {
		if len(mapping.LabelToID) > 0 {
			mapping.LabelToIdx = mapping.LabelToID
		} else if len(mapping.CategoryToIdx) > 0 {
			mapping.LabelToIdx = mapping.CategoryToIdx
		}
	}
	if len(mapping.IdxToLabel) == 0 {
		if len(mapping.IDToLabel) > 0 {
			mapping.IdxToLabel = mapping.IDToLabel
		} else if len(mapping.IdxToCategory) > 0 {
			mapping.IdxToLabel = mapping.IdxToCategory
		}
	}

	if err := validateComplexityMapping(&mapping); err != nil {
		return nil, fmt.Errorf("invalid complexity mapping %q: %w", path, err)
	}

	return &mapping, nil
}

// validateComplexityMapping checks that a normalized complexity mapping is
// usable at runtime: it must map a contiguous block of class indices [0, N)
// onto the supported lowercase difficulty labels easy/medium/hard.
//
// Without this, a JSON file that parses but carries none of the recognized keys
// yields empty maps and every classification silently misses
// (GetDifficultyFromIndex returns ok=false); and a mapping with a stray label
// like "HARD" or an arbitrary class name loads successfully but emits a
// difficulty the decision schema can never match.
func validateComplexityMapping(cm *ComplexityMapping) error {
	if len(cm.IdxToLabel) == 0 {
		return fmt.Errorf("mapping is empty; expected an idx_to_label (or id_to_label / idx_to_category) block")
	}

	indices := make([]int, 0, len(cm.IdxToLabel))
	for idxStr, label := range cm.IdxToLabel {
		idx, err := strconv.Atoi(idxStr)
		if err != nil {
			return fmt.Errorf("class index %q is not an integer", idxStr)
		}
		if _, ok := complexityDifficultyLabels[label]; !ok {
			return fmt.Errorf("class index %d maps to unsupported label %q; supported labels are easy, medium, hard", idx, label)
		}
		indices = append(indices, idx)
	}

	sort.Ints(indices)
	for i, idx := range indices {
		if idx != i {
			return fmt.Errorf("class indices must be contiguous starting at 0; got %v", indices)
		}
	}

	return nil
}

// GetDifficultyFromIndex converts a class index to a difficulty label using the mapping.
func (cm *ComplexityMapping) GetDifficultyFromIndex(classIndex int) (string, bool) {
	label, ok := cm.IdxToLabel[fmt.Sprintf("%d", classIndex)]
	return label, ok
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
