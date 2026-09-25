/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package modelselection

// VSR's 14 categories for one-hot encoding
// IMPORTANT: This order MUST match Python training: src/training/model_selection/ml_model_selection/data_loader.py
var VSRCategories = []string{
	"math",
	"physics",
	"chemistry",
	"biology",
	"computer science",
	"history",
	"economics",
	"business",
	"law",
	"health",
	"psychology",
	"philosophy",
	"other",
	"unknown",
}

// CategoryToIndex maps category name to one-hot index
var CategoryToIndex = func() map[string]int {
	m := make(map[string]int)
	for i, cat := range VSRCategories {
		m[cat] = i
	}
	return m
}()

// CategoryToOneHot converts a category name to a one-hot encoded vector
func CategoryToOneHot(category string) []float64 {
	oneHot := make([]float64, len(VSRCategories))
	if idx, ok := CategoryToIndex[category]; ok {
		oneHot[idx] = 1.0
	} else {
		// Default to "other" if unknown category
		oneHot[CategoryToIndex["other"]] = 1.0
	}
	return oneHot
}

// CombineEmbeddingWithCategory creates the full feature vector
// Feature = [QueryEmbedding] + [CategoryOneHot], with the fixed domain order above.
func CombineEmbeddingWithCategory(embedding []float64, category string) []float64 {
	categoryOneHot := CategoryToOneHot(category)
	combined := make([]float64, len(embedding)+len(categoryOneHot))
	copy(combined, embedding)
	copy(combined[len(embedding):], categoryOneHot)
	return combined
}
