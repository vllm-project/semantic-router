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

import (
	"k8s.io/apimachinery/pkg/runtime"
)

// ComplexityRulesConfig defines complexity-based signal classification
type ComplexityRulesConfig struct {
	// Name of the complexity rule (e.g., "code-complexity", "reasoning-complexity")
	Name string `json:"name"`

	// Description of what this rule classifies
	// +optional
	Description string `json:"description,omitempty"`

	// Threshold for difficulty classification (0.0-1.0). Stored as string to avoid float precision issues.
	// Queries scoring above this threshold are classified as "hard"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`

	// Hard candidates represent complex/difficult examples
	Hard ComplexityCandidates `json:"hard"`

	// Easy candidates represent simple/easy examples
	Easy ComplexityCandidates `json:"easy"`

	// Composer allows filtering based on other signals (e.g., only apply this rule if domain:medical)
	// +optional
	Composer *RuleComposition `json:"composer,omitempty"`
}

// ComplexityCandidates defines candidate examples for complexity classification
type ComplexityCandidates struct {
	// List of candidate phrases or examples
	Candidates []string `json:"candidates"`
}

// RuleComposition defines how to compose/filter rules based on other signals
type RuleComposition struct {
	// Operator for combining conditions (AND, OR, NOT). NOT is strictly unary and negates its single child.
	// +kubebuilder:validation:Enum=AND;OR;NOT
	Operator string `json:"operator"`

	// List of conditions that must be met
	Conditions []CompositionCondition `json:"conditions"`
}

// CompositionCondition defines a single composition condition
type CompositionCondition struct {
	// Type of signal to check (e.g., "domain", "language", "category")
	Type string `json:"type"`

	// Name of the specific signal/rule value to match
	Name string `json:"name"`
}

// DecisionConfig defines a routing decision
type DecisionConfig struct {
	// Name is the unique identifier for this decision
	Name string `json:"name" yaml:"name"`

	// Description provides information about what this decision handles
	// +optional
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// Priority is used for decision ordering - higher priority decisions are evaluated first
	// +optional
	Priority int `json:"priority,omitempty" yaml:"priority,omitempty"`

	// Rules defines the combination of conditions using AND/OR logic
	Rules RuleCombinationConfig `json:"rules" yaml:"rules"`

	// ModelRefs contains model references for this decision
	// +optional
	ModelRefs []ModelRefConfig `json:"modelRefs,omitempty" yaml:"modelRefs,omitempty"`

	// PreferredEndpoints specifies which vLLM endpoints to prefer for this decision
	// +optional
	PreferredEndpoints []string `json:"preferred_endpoints,omitempty" yaml:"preferred_endpoints,omitempty"`

	// Plugins contains policy configurations applied after rule matching
	// +optional
	Plugins []runtime.RawExtension `json:"plugins,omitempty" yaml:"plugins,omitempty"`
}

// RuleCombinationConfig defines how to combine multiple rule conditions
type RuleCombinationConfig struct {
	// Operator specifies how to combine conditions: "AND", "OR", or "NOT". NOT is strictly unary: it takes
	// exactly one child condition and negates its result. Compose NOR/NAND by nesting NOT around OR/AND.
	// +kubebuilder:validation:Enum=AND;OR;NOT
	Operator string `json:"operator" yaml:"operator"`

	// Conditions is the list of rule references to evaluate
	Conditions []RuleConditionConfig `json:"conditions" yaml:"conditions"`
}

// RuleConditionConfig references a specific rule by type and name
type RuleConditionConfig struct {
	// Type specifies the rule type: "keyword", "embedding", "domain", "complexity", or "fact_check"
	// +kubebuilder:validation:Enum=keyword;embedding;domain;complexity;fact_check;context
	Type string `json:"type" yaml:"type"`

	// Name is the name of the rule to reference
	Name string `json:"name" yaml:"name"`
}

// ModelRefConfig defines a model reference for routing
type ModelRefConfig struct {
	// Model name to route to
	Model string `json:"model" yaml:"model"`

	// LoRAName is the optional LoRA adapter name
	// +optional
	LoRAName string `json:"lora_name,omitempty" yaml:"lora_name,omitempty"`

	// UseReasoning enables reasoning mode for this model
	// +optional
	UseReasoning *bool `json:"use_reasoning,omitempty" yaml:"use_reasoning,omitempty"`

	// ReasoningEffort specifies the reasoning effort level (low, medium, high)
	// +optional
	ReasoningEffort string `json:"reasoning_effort,omitempty" yaml:"reasoning_effort,omitempty"`
}
