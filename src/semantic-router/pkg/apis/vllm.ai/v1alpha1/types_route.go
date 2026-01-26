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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// IntelligentRouteSpec defines the desired state of IntelligentRoute
type IntelligentRouteSpec struct {
	// Signals defines signal extraction rules for routing decisions
	// +optional
	Signals Signals `json:"signals,omitempty" yaml:"signals,omitempty"`

	// Decisions defines the routing decisions based on signal combinations
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=100
	Decisions []Decision `json:"decisions" yaml:"decisions"`
}

// Signals defines signal extraction rules
type Signals struct {
	// Keywords defines keyword-based signal extraction rules
	// +optional
	// +kubebuilder:validation:MaxItems=100
	Keywords []KeywordSignal `json:"keywords,omitempty" yaml:"keywords,omitempty"`

	// Embeddings defines embedding-based signal extraction rules
	// +optional
	// +kubebuilder:validation:MaxItems=100
	Embeddings []EmbeddingSignal `json:"embeddings,omitempty" yaml:"embeddings,omitempty"`

	// Domains defines MMLU domain categories for classification
	// +optional
	// +kubebuilder:validation:MaxItems=14
	Domains []DomainSignal `json:"domains,omitempty" yaml:"domains,omitempty"`

	// FactCheckRules defines fact-check rules for signal classification
	// Similar to KeywordRules and EmbeddingRules, but based on ML model classification
	// Each rule has a name that can be referenced in decision conditions
	// +optional
	ContextRules []ContextRule `json:"contextRules,omitempty" yaml:"context_rules,omitempty"`

	// +optional
	FactCheckRules []FactCheckRule `json:"factCheckRules,omitempty" yaml:"fact_check_rules,omitempty"`
}

// FactCheckRule defines a rule for fact-check signal classification
// The classifier determines if a query needs fact verification based on the ML model
// Predefined signal names: "needs_fact_check", "no_fact_check_needed"
// Threshold is read from hallucination_mitigation.fact_check_model.threshold
type FactCheckRule struct {
	// Name is the signal name that can be referenced in decision rules
	// e.g., "needs_fact_check" or "no_fact_check_needed"
	Name string `json:"name" yaml:"name"`

	// Description provides human-readable explanation of when this signal is triggered
	// +optional
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

// ContextRule defines a rule for context-based (token count) classification
type ContextRule struct {
	// Name is the signal name (e.g., "high_token_count")
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// MinTokens is the minimum token count (supports K/M suffixes)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[0-9]+(\.[0-9]+)?[KMkm]?$`
	MinTokens string `json:"minTokens" yaml:"min_tokens"`

	// MaxTokens is the maximum token count (supports K/M suffixes)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[0-9]+(\.[0-9]+)?[KMkm]?$`
	MaxTokens string `json:"maxTokens" yaml:"max_tokens"`

	// Description provides human-readable explanation
	// +optional
	// +kubebuilder:validation:MaxLength=500
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

// DomainSignal defines a domain category for classification
type DomainSignal struct {
	// Name is the unique identifier for this domain
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// Description provides a human-readable description of this domain
	// +optional
	// +kubebuilder:validation:MaxLength=500
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

// KeywordSignal defines a keyword-based signal extraction rule
type KeywordSignal struct {
	// Name is the unique identifier for this rule (also used as category name)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// Operator defines the logical operator for keywords (AND/OR)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=AND;OR
	Operator string `json:"operator" yaml:"operator"`

	// Keywords is the list of keywords to match
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=100
	Keywords []string `json:"keywords" yaml:"keywords"`

	// CaseSensitive specifies whether keyword matching is case-sensitive
	// +optional
	// +kubebuilder:default=false
	CaseSensitive bool `json:"caseSensitive" yaml:"caseSensitive"`
}

// EmbeddingSignal defines an embedding-based signal extraction rule
type EmbeddingSignal struct {
	// Name is the unique identifier for this signal
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// Threshold is the similarity threshold for matching (0.0-1.0)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	Threshold float32 `json:"threshold" yaml:"threshold"`

	// Candidates is the list of candidate phrases for semantic matching
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=100
	Candidates []string `json:"candidates" yaml:"candidates"`

	// AggregationMethod defines how to aggregate multiple candidate similarities
	// +optional
	// +kubebuilder:validation:Enum=mean;max;any
	// +kubebuilder:default=max
	AggregationMethod string `json:"aggregationMethod,omitempty" yaml:"aggregationMethod,omitempty"`
}

// Decision defines a routing decision based on rule combinations
type Decision struct {
	// Name is the unique identifier for this decision
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// Priority defines the priority of this decision (higher values = higher priority)
	// Used when strategy is "priority"
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1000
	// +kubebuilder:default=0
	Priority int32 `json:"priority" yaml:"priority"`

	// Description provides a human-readable description of this decision
	// +optional
	// +kubebuilder:validation:MaxLength=500
	Description string `json:"description,omitempty" yaml:"description,omitempty"`

	// Signals defines the signal combination logic
	// +kubebuilder:validation:Required
	Signals SignalCombination `json:"signals" yaml:"signals"`

	// ModelRefs defines the model references for this decision (currently only one model is supported)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=1
	ModelRefs []ModelRef `json:"modelRefs" yaml:"modelRefs"`

	// Plugins defines the plugins to apply for this decision
	// +optional
	// +kubebuilder:validation:MaxItems=10
	Plugins []DecisionPlugin `json:"plugins,omitempty" yaml:"plugins,omitempty"`
}

// SignalCombination defines how to combine multiple signals
type SignalCombination struct {
	// Operator defines the logical operator for combining conditions (AND/OR)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=AND;OR
	Operator string `json:"operator" yaml:"operator"`

	// Conditions defines the list of signal conditions
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=50
	Conditions []SignalCondition `json:"conditions" yaml:"conditions"`
}

// SignalCondition defines a single signal condition
type SignalCondition struct {
	// Type defines the type of signal (keyword/embedding/domain/fact_check/context)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=keyword;embedding;domain;fact_check;context
	Type string `json:"type" yaml:"type"`

	// Name is the name of the signal to reference
	// For fact_check type, use "needs_fact_check" to match queries that need fact verification
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`
}

// ModelRef defines a model reference without score
type ModelRef struct {
	// Model is the name of the model (must exist in IntelligentPool)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Model string `json:"model" yaml:"model"`

	// LoRAName is the name of the LoRA adapter to use (must exist in the model's LoRAs)
	// +optional
	// +kubebuilder:validation:MaxLength=100
	LoRAName string `json:"loraName,omitempty" yaml:"loraName,omitempty"`

	// UseReasoning specifies whether to enable reasoning mode for this model
	// +optional
	// +kubebuilder:default=false
	UseReasoning bool `json:"useReasoning" yaml:"useReasoning"`

	// ReasoningDescription provides context for when to use reasoning
	// +optional
	// +kubebuilder:validation:MaxLength=500
	ReasoningDescription string `json:"reasoningDescription,omitempty" yaml:"reasoningDescription,omitempty"`

	// ReasoningEffort defines the reasoning effort level (low/medium/high)
	// +optional
	// +kubebuilder:validation:Enum=low;medium;high
	ReasoningEffort string `json:"reasoningEffort,omitempty" yaml:"reasoningEffort,omitempty"`
}

// DecisionPlugin defines a plugin configuration for a decision
type DecisionPlugin struct {
	// Type is the plugin type (semantic-cache, jailbreak, pii, system_prompt, header_mutation, hallucination)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=semantic-cache;jailbreak;pii;system_prompt;header_mutation;hallucination
	Type string `json:"type" yaml:"type"`

	// Configuration is the plugin-specific configuration as a raw JSON object
	// +optional
	// +kubebuilder:pruning:PreserveUnknownFields
	// +kubebuilder:validation:Schemaless
	Configuration *runtime.RawExtension `json:"configuration,omitempty" yaml:"configuration,omitempty"`
}

// ModelScore defines the model selection score (deprecated, use ModelRef instead)
// This type is kept for backward compatibility with existing CRDs
type ModelScore struct {
	// Model is the name of the model (must exist in IntelligentPool)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Model string `json:"model" yaml:"model"`

	// Score is the selection score for this model (0.0-1.0)
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	Score float32 `json:"score" yaml:"score"`

	// LoRAName is the name of the LoRA adapter to use (must exist in the model's LoRAs)
	// +optional
	// +kubebuilder:validation:MaxLength=100
	LoRAName string `json:"loraName,omitempty" yaml:"loraName,omitempty"`

	// UseReasoning specifies whether to enable reasoning mode for this model
	// +optional
	// +kubebuilder:default=false
	UseReasoning bool `json:"useReasoning" yaml:"useReasoning"`

	// ReasoningDescription provides context for when to use reasoning
	// +optional
	// +kubebuilder:validation:MaxLength=500
	ReasoningDescription string `json:"reasoningDescription,omitempty" yaml:"reasoningDescription,omitempty"`

	// ReasoningEffort defines the reasoning effort level (low/medium/high)
	// +optional
	// +kubebuilder:validation:Enum=low;medium;high
	ReasoningEffort string `json:"reasoningEffort,omitempty" yaml:"reasoningEffort,omitempty"`
}

// IntelligentRouteStatus defines the observed state of IntelligentRoute
type IntelligentRouteStatus struct {
	// Conditions represent the latest available observations of the IntelligentRoute's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty" yaml:"conditions,omitempty"`

	// ObservedGeneration reflects the generation of the most recently observed IntelligentRoute
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty" yaml:"observedGeneration,omitempty"`

	// Statistics provides statistics about configured decisions and signals
	// +optional
	Statistics *RouteStatistics `json:"statistics,omitempty" yaml:"statistics,omitempty"`
}

// RouteStatistics provides statistics about the IntelligentRoute configuration
type RouteStatistics struct {
	// Decisions indicates the number of decisions
	Decisions int32 `json:"decisions" yaml:"decisions"`

	// Keywords indicates the number of keyword signals
	Keywords int32 `json:"keywords" yaml:"keywords"`

	// Embeddings indicates the number of embedding signals
	Embeddings int32 `json:"embeddings" yaml:"embeddings"`

	// Domains indicates the number of domain signals
	Domains int32 `json:"domains" yaml:"domains"`
}

// IntelligentRouteList contains a list of IntelligentRoute
// +kubebuilder:object:root=true
type IntelligentRouteList struct {
	metav1.TypeMeta `json:",inline" yaml:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" yaml:"metadata,omitempty"`
	Items           []IntelligentRoute `json:"items" yaml:"items"`
}
