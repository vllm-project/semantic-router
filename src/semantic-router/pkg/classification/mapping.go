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
	if err := validateCategoryMapping(path, &mapping); err != nil {
		return nil, err
	}

	return &mapping, nil
}

// validateCategoryMapping requires both directions to describe the same
// zero-based contiguous bijection. Remote scores are aligned through the
// category->index direction, while routing resolves the winning index through
// index->category; accepting disagreement between them can silently route a
// correct score to the wrong category.
func validateCategoryMapping(path string, mapping *CategoryMapping) error {
	if len(mapping.CategoryToIdx) != len(mapping.IdxToCategory) {
		return fmt.Errorf("category mapping %s: category_to_idx and idx_to_category must have the same size", path)
	}
	if err := validateCategoryMappingIndexes(path, mapping); err != nil {
		return err
	}
	if err := validateCategoryMappingLabels(path, mapping); err != nil {
		return err
	}
	return nil
}

func validateCategoryMappingIndexes(path string, mapping *CategoryMapping) error {
	for idx := 0; idx < len(mapping.CategoryToIdx); idx++ {
		if _, ok := mapping.IdxToCategory[strconv.Itoa(idx)]; !ok {
			return fmt.Errorf("missing label for index %d in category mapping %s", idx, path)
		}
	}
	for label, idx := range mapping.CategoryToIdx {
		if idx < 0 || idx >= len(mapping.CategoryToIdx) {
			return fmt.Errorf("category mapping %s: index for %q must be contiguous from 0, got %d", path, label, idx)
		}
		reverse, ok := mapping.IdxToCategory[strconv.Itoa(idx)]
		if !ok || reverse != label {
			return fmt.Errorf("category mapping %s: category_to_idx and idx_to_category disagree for %q", path, label)
		}
	}
	return nil
}

func validateCategoryMappingLabels(path string, mapping *CategoryMapping) error {
	for index, label := range mapping.IdxToCategory {
		idx, err := strconv.Atoi(index)
		if err != nil || idx < 0 || idx >= len(mapping.CategoryToIdx) {
			return fmt.Errorf("category mapping %s: index->category key %q is not contiguous from 0", path, index)
		}
		if mapped, ok := mapping.CategoryToIdx[label]; !ok || mapped != idx {
			return fmt.Errorf("category mapping %s: idx_to_category and category_to_idx disagree for index %q", path, index)
		}
	}
	return nil
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

	// Collapse whichever of the four accepted label maps the file declared into
	// one agreed label<->index set, so every consumer reads the same thing.
	if err := canonicalizeJailbreakMapping(path, &mapping); err != nil {
		return nil, err
	}

	// A configured label equal to the on_error: block sentinel
	// (JailbreakClassificationErrorType) would make a genuine detection of
	// that label indistinguishable from a classify failure - see @adaamko's
	// review on #2918/#2930. Reject it here, once, rather than at every
	// place that compares against the sentinel.
	//
	// This deliberately probes via GetIndexForJailbreakType rather than
	// indexing LabelToIdx directly: it is the same lookup the runtime uses to
	// resolve a label, so the guard cannot cover fewer shapes than the code it
	// guards. Canonicalization above already unified the shapes, but that is
	// what makes the probe safe rather than a reason to drop it.
	if _, collides := mapping.GetIndexForJailbreakType(JailbreakClassificationErrorType); collides {
		return nil, fmt.Errorf(
			"jailbreak mapping %s: label %q is reserved for the on_error: block sentinel and cannot be a configured label",
			path, JailbreakClassificationErrorType)
	}

	return &mapping, nil
}

// canonicalizeJailbreakMapping collapses the four accepted label maps into one
// agreed label<->index set on LabelToIdx/IdxToLabel, or fails.
//
// The formats let a file declare any subset of label_to_idx, label_to_id,
// idx_to_label and id_to_label. Consumers disagree about which they read:
// LabelCount() reads the label->index map, GetJailbreakTypeFromIndex reads the
// index->label maps, and GetIndexForJailbreakType falls back to scanning them.
// alignScoresToMapping sizes the http_classify probability distribution from
// LabelCount() and indexes it by resolved label index, so any disagreement
// silently truncates that distribution: a class that resolves but has no slot
// makes a response omitting it pass the completeness guard, and the missing
// class - possibly the positive one - reads as probability 0.0. Deriving one
// canonical set at load makes the count authoritative for every consumer, and
// keeps the reverse scan off the request path.
//
// Indices must be contiguous from 0 for the same reason: the distribution has
// exactly LabelCount() slots, so a 1-based or sparse mapping would reject every
// response at request time instead of at load.
func canonicalizeJailbreakMapping(path string, mapping *JailbreakMapping) error {
	labelToIdx, err := collectJailbreakLabelIndices(path, mapping)
	if err != nil {
		return err
	}
	idxToLabel, err := invertJailbreakLabelIndices(path, labelToIdx)
	if err != nil {
		return err
	}
	if err := checkJailbreakIndicesContiguous(path, idxToLabel); err != nil {
		return err
	}

	mapping.LabelToIdx = labelToIdx
	mapping.IdxToLabel = idxToLabel
	return nil
}

// collectJailbreakLabelIndices unions every label->index pair the file declared,
// in either naming convention and either direction.
func collectJailbreakLabelIndices(path string, mapping *JailbreakMapping) (map[string]int, error) {
	labelToIdx := make(map[string]int)
	for _, src := range []map[string]int{mapping.LabelToIdx, mapping.LabelToID} {
		for label, idx := range src {
			if err := addJailbreakLabel(path, labelToIdx, label, idx); err != nil {
				return nil, err
			}
		}
	}
	for _, src := range []map[string]string{mapping.IdxToLabel, mapping.IDToLabel} {
		for indexStr, label := range src {
			idx, err := strconv.Atoi(indexStr)
			if err != nil {
				return nil, fmt.Errorf(
					"jailbreak mapping %s: index->label key %q is not a number", path, indexStr)
			}
			if err := addJailbreakLabel(path, labelToIdx, label, idx); err != nil {
				return nil, err
			}
		}
	}
	return labelToIdx, nil
}

// addJailbreakLabel records one label->index pair, rejecting a label that two
// of the source maps disagree about.
func addJailbreakLabel(path string, labelToIdx map[string]int, label string, idx int) error {
	if existing, seen := labelToIdx[label]; seen && existing != idx {
		return fmt.Errorf(
			"jailbreak mapping %s: inconsistent label maps - label %q is index %d in one map and %d in another",
			path, label, existing, idx)
	}
	labelToIdx[label] = idx
	return nil
}

// invertJailbreakLabelIndices builds the index->label direction, rejecting an
// index two labels both claim.
func invertJailbreakLabelIndices(path string, labelToIdx map[string]int) (map[string]string, error) {
	idxToLabel := make(map[string]string, len(labelToIdx))
	for label, idx := range labelToIdx {
		key := strconv.Itoa(idx)
		if existing, taken := idxToLabel[key]; taken {
			return nil, fmt.Errorf(
				"jailbreak mapping %s: inconsistent label maps - index %d is claimed by both %q and %q",
				path, idx, existing, label)
		}
		idxToLabel[key] = label
	}
	return idxToLabel, nil
}

// checkJailbreakIndicesContiguous requires indices 0..n-1, the slots
// alignScoresToMapping allocates.
func checkJailbreakIndicesContiguous(path string, idxToLabel map[string]string) error {
	for idx := 0; idx < len(idxToLabel); idx++ {
		if _, ok := idxToLabel[strconv.Itoa(idx)]; !ok {
			return fmt.Errorf(
				"jailbreak mapping %s: label indices must be contiguous from 0, but %d of %d is missing",
				path, idx, len(idxToLabel))
		}
	}
	return nil
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
