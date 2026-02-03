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

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// MultiModelCategory represents a category/decision with multiple models configured
type MultiModelCategory struct {
	Name   string   // Decision/category name
	Models []string // List of model names
}

// ConfigAnalysisResult contains the analysis of a config file for model selection
type ConfigAnalysisResult struct {
	// Categories that have multiple models (need model selection)
	MultiModelCategories []MultiModelCategory

	// All unique models that need training
	ModelsNeedingTraining []string

	// Categories with single model (no selection needed)
	SingleModelCategories []string

	// Total decisions analyzed
	TotalDecisions int
}

// AnalyzeConfigForModelSelection reads a VSR config file and identifies
// which categories have multiple models configured (requiring model selection)
func AnalyzeConfigForModelSelection(configPath string) (*ConfigAnalysisResult, error) {
	// Read config file
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	// Parse YAML
	var cfg config.RouterConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return AnalyzeConfig(&cfg), nil
}

// AnalyzeConfig analyzes a RouterConfig to find multi-model categories
func AnalyzeConfig(cfg *config.RouterConfig) *ConfigAnalysisResult {
	result := &ConfigAnalysisResult{
		MultiModelCategories:  []MultiModelCategory{},
		ModelsNeedingTraining: []string{},
		SingleModelCategories: []string{},
		TotalDecisions:        0,
	}

	// Track unique models
	uniqueModels := make(map[string]bool)

	// Analyze decisions
	for _, decision := range cfg.Decisions {
		result.TotalDecisions++

		// Extract model names from ModelRefs
		var modelNames []string
		for _, ref := range decision.ModelRefs {
			if ref.Model != "" {
				modelNames = append(modelNames, ref.Model)
			}
		}

		if len(modelNames) > 1 {
			// Multiple models - needs selection
			result.MultiModelCategories = append(result.MultiModelCategories, MultiModelCategory{
				Name:   decision.Name,
				Models: modelNames,
			})

			// Add to unique models
			for _, m := range modelNames {
				uniqueModels[m] = true
			}

			logging.Infof("✅ Category '%s': %d models → Model selection needed", decision.Name, len(modelNames))
		} else if len(modelNames) == 1 {
			// Single model - no selection needed
			result.SingleModelCategories = append(result.SingleModelCategories, decision.Name)
			logging.Debugf("⏭️  Category '%s': 1 model → No selection needed", decision.Name)
		} else {
			logging.Warnf("⚠️  Category '%s': No models configured", decision.Name)
		}
	}

	// Convert unique models map to slice
	for model := range uniqueModels {
		result.ModelsNeedingTraining = append(result.ModelsNeedingTraining, model)
	}

	return result
}

// PrintAnalysisSummary prints a summary of the config analysis
func (r *ConfigAnalysisResult) PrintAnalysisSummary() {
	fmt.Println("\n" + "=" + repeatString("=", 59))
	fmt.Println("MODEL SELECTION CONFIG ANALYSIS")
	fmt.Println(repeatString("=", 60))

	fmt.Printf("\n📊 Total decisions analyzed: %d\n", r.TotalDecisions)
	fmt.Printf("   • Multi-model categories: %d (need selection)\n", len(r.MultiModelCategories))
	fmt.Printf("   • Single-model categories: %d (no selection needed)\n", len(r.SingleModelCategories))

	if len(r.MultiModelCategories) > 0 {
		fmt.Println("\n🎯 Categories requiring model selection:")
		for _, cat := range r.MultiModelCategories {
			fmt.Printf("   • %s: %v\n", cat.Name, cat.Models)
		}

		fmt.Printf("\n🤖 Unique models to train: %d\n", len(r.ModelsNeedingTraining))
		for _, model := range r.ModelsNeedingTraining {
			fmt.Printf("   • %s\n", model)
		}
	} else {
		fmt.Println("\n⚠️  No multi-model categories found!")
		fmt.Println("   Model selection training is not needed.")
		fmt.Println("   Each category has only one model configured.")
	}

	fmt.Println(repeatString("=", 60))
}

// NeedsModelSelectionTraining returns true if there are categories
// with multiple models that would benefit from model selection
func (r *ConfigAnalysisResult) NeedsModelSelectionTraining() bool {
	return len(r.MultiModelCategories) > 0
}

// GetModelEndpoints returns endpoint information for a model from config
func GetModelEndpoints(cfg *config.RouterConfig, modelName string) []string {
	var endpoints []string

	// Check model_config for preferred endpoints
	if modelCfg, ok := cfg.ModelConfig[modelName]; ok {
		endpoints = append(endpoints, modelCfg.PreferredEndpoints...)
	}

	// If no preferred endpoints, return all vLLM endpoints
	if len(endpoints) == 0 {
		for _, ep := range cfg.VLLMEndpoints {
			endpoints = append(endpoints, fmt.Sprintf("%s:%d", ep.Address, ep.Port))
		}
	}

	return endpoints
}

// repeatString repeats a string n times
func repeatString(s string, n int) string {
	result := ""
	for i := 0; i < n; i++ {
		result += s
	}
	return result
}
