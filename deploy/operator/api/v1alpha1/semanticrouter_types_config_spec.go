/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package v1alpha1

// ConfigSpec defines the semantic router configuration
type ConfigSpec struct {
	// Embedding models configuration (qwen3, gemma, mmbert)
	// +optional
	EmbeddingModels *EmbeddingModelsConfig `json:"embedding_models,omitempty"`

	// Semantic cache configuration
	// +optional
	SemanticCache *SemanticCacheConfig `json:"semantic_cache,omitempty"`

	// Tools configuration
	// +optional
	Tools *ToolsConfig `json:"tools,omitempty"`

	// Prompt guard configuration
	// +optional
	PromptGuard *PromptGuardConfig `json:"prompt_guard,omitempty"`

	// Classifier configuration
	// +optional
	Classifier *ClassifierConfig `json:"classifier,omitempty"`

	// Complexity rules for complexity-aware routing
	// +optional
	ComplexityRules []ComplexityRulesConfig `json:"complexity_rules,omitempty"`

	// Decision routing strategy ("priority" for priority-based matching)
	// +kubebuilder:validation:Enum=priority
	// +optional
	Strategy string `json:"strategy,omitempty" yaml:"strategy,omitempty"`

	// Routing decisions based on signals (domain, complexity, etc.)
	// +optional
	Decisions []DecisionConfig `json:"decisions,omitempty"`

	// Reasoning families
	// +optional
	ReasoningFamilies map[string]ReasoningFamily `json:"reasoning_families,omitempty"`

	// Default reasoning effort
	// +kubebuilder:validation:Enum=low;medium;high
	// +optional
	DefaultReasoningEffort string `json:"default_reasoning_effort,omitempty"`

	// API configuration
	// +optional
	API *APIConfig `json:"api,omitempty"`

	// Observability configuration
	// +optional
	Observability *ObservabilityConfig `json:"observability,omitempty"`
}
