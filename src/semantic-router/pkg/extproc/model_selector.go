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

package extproc

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability"
)

// classifyAndSelectBestModel chooses best models based on category classification and model quality and expected TTFT
func (r *OpenAIRouter) classifyAndSelectBestModel(query string) string {
	return r.Classifier.ClassifyAndSelectBestModel(query)
}

// findCategoryForClassification determines the category for the given text using classification
func (r *OpenAIRouter) findCategoryForClassification(query string) string {
	if len(r.CategoryDescriptions) == 0 {
		return ""
	}

	categoryName, _, err := r.Classifier.ClassifyCategory(query)
	if err != nil {
		observability.Errorf("Category classification error: %v", err)
		return ""
	}

	return categoryName
}
