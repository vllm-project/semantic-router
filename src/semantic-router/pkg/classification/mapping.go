package classification

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx         map[string]int    `json:"category_to_idx"`
	IdxToCategory         map[string]string `json:"idx_to_category"`
	CategorySystemPrompts map[string]string `json:"category_system_prompts,omitempty"` // Optional per-category system prompts from MCP server
	CategoryDescriptions  map[string]string `json:"category_descriptions,omitempty"`   // Optional category descriptions
}

// PIIMapping holds the mapping between indices and PII types
// Supports both formats: label_to_idx/idx_to_label (legacy) and label_to_id/id_to_label (mmBERT)
type PIIMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"` // Legacy format
	IdxToLabel map[string]string `json:"idx_to_label"` // Legacy format
	// mmBERT format (alternative field names)
	LabelToID map[string]int    `json:"label_to_id,omitempty"` // mmBERT format
	IDToLabel map[string]string `json:"id_to_label,omitempty"` // mmBERT format
}

// JailbreakMapping holds the mapping between indices and jailbreak types
// Supports: label_to_idx/idx_to_label (legacy), label_to_id/id_to_label (mmBERT), id2label (HuggingFace)
type JailbreakMapping struct {
	LabelToIdx map[string]int    `json:"label_to_idx"` // Legacy format
	IdxToLabel map[string]string `json:"idx_to_label"` // Legacy format
	// mmBERT format (alternative field names)
	LabelToID map[string]int    `json:"label_to_id,omitempty"` // mmBERT format
	IDToLabel map[string]string `json:"id_to_label,omitempty"` // mmBERT format
	// HuggingFace config convention (no underscore)
	Id2Label map[string]string `json:"id2label,omitempty"`
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
// Supports both formats: label_to_idx/idx_to_label (legacy) and label_to_id/id_to_label (mmBERT)
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

	// Normalize: if mmBERT format (label_to_id/id_to_label) is present but legacy format is not, copy it
	if len(mapping.LabelToIdx) == 0 && len(mapping.LabelToID) > 0 {
		mapping.LabelToIdx = mapping.LabelToID
	}
	if len(mapping.IdxToLabel) == 0 && len(mapping.IDToLabel) > 0 {
		mapping.IdxToLabel = mapping.IDToLabel
	}

	return &mapping, nil
}

// LoadJailbreakMapping loads the jailbreak mapping from a JSON file.
// Supports: label_to_idx/idx_to_label (legacy), label_to_id/id_to_label (mmBERT), id2label (HuggingFace).
// If the file uses an unknown structure, falls back to parsing raw JSON for id2label, label2id, or labels array.
// If path contains "mom-mmbert32k-" and the file is missing, tries the same path with "mmbert32k-" (no mom prefix) as fallback.
func LoadJailbreakMapping(path string) (*JailbreakMapping, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		// Path fallback: if using mom- prefix and file not found, try without mom- (e.g. model downloaded to old path)
		if os.IsNotExist(err) && strings.Contains(path, "mom-mmbert32k-") {
			fallback := strings.Replace(path, "mom-mmbert32k-", "mmbert32k-", 1)
			if fallback != path {
				data, err = os.ReadFile(fallback)
			}
		}
		if err != nil {
			return nil, fmt.Errorf("failed to read jailbreak mapping file: %w", err)
		}
	}

	// Parse the JSON data - will populate whichever fields match
	var mapping JailbreakMapping
	if err := json.Unmarshal(data, &mapping); err != nil {
		return nil, fmt.Errorf("failed to parse jailbreak mapping JSON: %w", err)
	}

	// Normalize: id2label (HuggingFace) -> IdxToLabel
	if len(mapping.IdxToLabel) == 0 && len(mapping.Id2Label) > 0 {
		mapping.IdxToLabel = mapping.Id2Label
	}
	// Normalize: id_to_label (mmBERT) -> IdxToLabel
	if len(mapping.IdxToLabel) == 0 && len(mapping.IDToLabel) > 0 {
		mapping.IdxToLabel = mapping.IDToLabel
	}
	// Normalize: label_to_id -> LabelToIdx
	if len(mapping.LabelToIdx) == 0 && len(mapping.LabelToID) > 0 {
		mapping.LabelToIdx = mapping.LabelToID
	}

	// Build label_to_idx from idx_to_label if missing
	if len(mapping.LabelToIdx) == 0 && len(mapping.IdxToLabel) > 0 {
		mapping.LabelToIdx = make(map[string]int)
		for idxStr, label := range mapping.IdxToLabel {
			var idx int
			if _, err := fmt.Sscanf(idxStr, "%d", &idx); err == nil {
				mapping.LabelToIdx[label] = idx
			}
		}
	}

	// Fallback: file may use different keys or nested structure (e.g. HuggingFace config)
	if mapping.GetJailbreakTypeCount() == 0 {
		populated := tryRawJailbreakFormats(data, &mapping)
		if !populated {
			// Try path fallback for reading: if we used mom- path and got empty, re-read from non-mom path
			if strings.Contains(path, "mom-mmbert32k-") {
				fallback := strings.Replace(path, "mom-mmbert32k-", "mmbert32k-", 1)
				if fallback != path {
					if data2, err := os.ReadFile(fallback); err == nil {
						populated = tryRawJailbreakFormats(data2, &mapping)
					}
				}
			}
		}
	}

	if mapping.GetJailbreakTypeCount() == 0 {
		absPath, _ := filepath.Abs(path)
		return nil, fmt.Errorf("jailbreak mapping has 0 types (file: %s); expected JSON with one of: id2label, id_to_label, idx_to_label, label_to_idx, label_to_id, label2id, or labels array (need at least 2 classes)", absPath)
	}

	return &mapping, nil
}

// tryRawJailbreakFormats tries to populate mapping from raw JSON (various key names and nested config). Returns true if any format worked.
func tryRawJailbreakFormats(data []byte, mapping *JailbreakMapping) bool {
	var raw map[string]interface{}
	if err := json.Unmarshal(data, &raw); err != nil {
		return false
	}
	// Collect candidate objects to search for id2label/label2id (top-level and nested)
	var toCheck []map[string]interface{}
	toCheck = append(toCheck, raw)
	if c, ok := raw["config"].(map[string]interface{}); ok {
		toCheck = append(toCheck, c)
	}
	if p, ok := raw["preprocessor_config"].(map[string]interface{}); ok {
		toCheck = append(toCheck, p)
	}
	for _, m := range toCheck {
		// id2label / id_to_label / idx_to_label (index -> label)
		for _, key := range []string{"id2label", "id_to_label", "idx_to_label"} {
			if v, ok := m[key].(map[string]interface{}); ok && len(v) > 0 {
				mapping.IdxToLabel = make(map[string]string)
				for k, val := range v {
					var label string
					switch t := val.(type) {
					case string:
						label = t
					case float64:
						label = fmt.Sprintf("%.0f", t)
					default:
						continue
					}
					mapping.IdxToLabel[k] = label
				}
				if len(mapping.LabelToIdx) == 0 && len(mapping.IdxToLabel) > 0 {
					mapping.LabelToIdx = make(map[string]int)
					for idxStr, label := range mapping.IdxToLabel {
						var idx int
						if _, err := fmt.Sscanf(idxStr, "%d", &idx); err == nil {
							mapping.LabelToIdx[label] = idx
						}
					}
				}
				return mapping.GetJailbreakTypeCount() >= 2
			}
		}
		// label2id (label -> id): build IdxToLabel by reversing
		if v, ok := m["label2id"].(map[string]interface{}); ok && len(v) > 0 {
			mapping.LabelToIdx = make(map[string]int)
			for label, val := range v {
				var id int
				switch t := val.(type) {
				case float64:
					id = int(t)
				case int:
					id = t
				default:
					continue
				}
				mapping.LabelToIdx[label] = id
			}
			if len(mapping.IdxToLabel) == 0 && len(mapping.LabelToIdx) > 0 {
				mapping.IdxToLabel = make(map[string]string)
				for label, id := range mapping.LabelToIdx {
					mapping.IdxToLabel[fmt.Sprintf("%d", id)] = label
				}
			}
			return mapping.GetJailbreakTypeCount() >= 2
		}
		// labels: array of strings ["safe", "jailbreak", ...]
		if v, ok := m["labels"].([]interface{}); ok && len(v) >= 2 {
			mapping.IdxToLabel = make(map[string]string)
			mapping.LabelToIdx = make(map[string]int)
			for i, val := range v {
				if s, ok := val.(string); ok {
					idxStr := fmt.Sprintf("%d", i)
					mapping.IdxToLabel[idxStr] = s
					mapping.LabelToIdx[s] = i
				}
			}
			return mapping.GetJailbreakTypeCount() >= 2
		}
	}
	return false
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

// GetJailbreakTypeCount returns the number of jailbreak types in the mapping.
// Uses whichever mapping is populated (label_to_idx, idx_to_label, label_to_id, id_to_label, id2label).
func (jm *JailbreakMapping) GetJailbreakTypeCount() int {
	return max(len(jm.LabelToIdx), len(jm.IdxToLabel), len(jm.LabelToID), len(jm.IDToLabel), len(jm.Id2Label))
}
