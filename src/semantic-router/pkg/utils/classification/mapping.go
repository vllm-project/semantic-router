// Copyright 2025 The vLLM Semantic Router Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package classification

import (
	"encoding/json"
	"fmt"
	"os"
)

// CategoryMapping holds the mapping between indices and domain categories
type CategoryMapping struct {
	CategoryToIdx map[string]int    `json:"category_to_idx"`
	IdxToCategory map[string]string `json:"idx_to_category"`
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

// GetJailbreakTypeFromIndex converts a class index to jailbreak type name using the mapping
func (jm *JailbreakMapping) GetJailbreakTypeFromIndex(classIndex int) (string, bool) {
	jailbreakType, ok := jm.IdxToLabel[fmt.Sprintf("%d", classIndex)]
	return jailbreakType, ok
}

// GetCategoryCount returns the number of categories in the mapping
func (cm *CategoryMapping) GetCategoryCount() int {
	return len(cm.CategoryToIdx)
}

// GetPIITypeCount returns the number of PII types in the mapping
func (pm *PIIMapping) GetPIITypeCount() int {
	return len(pm.LabelToIdx)
}

// GetJailbreakTypeCount returns the number of jailbreak types in the mapping
func (jm *JailbreakMapping) GetJailbreakTypeCount() int {
	return len(jm.LabelToIdx)
}
