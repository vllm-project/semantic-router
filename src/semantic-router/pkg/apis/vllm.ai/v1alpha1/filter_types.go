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

// PIIDetectionConfig defines the configuration for PII detection filter
type PIIDetectionConfig struct {
	// AllowByDefault defines whether PII is allowed by default
	// +optional
	// +kubebuilder:default=false
	AllowByDefault *bool `json:"allowByDefault,omitempty"`

	// PIITypesAllowed defines the list of PII types that are allowed
	// +optional
	// +kubebuilder:validation:MaxItems=50
	PIITypesAllowed []string `json:"pii_types_allowed,omitempty"`

	// Threshold defines the confidence threshold for PII detection (0.0-1.0)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	// +kubebuilder:default=0.7
	Threshold *float64 `json:"threshold,omitempty"`

	// Action defines what to do when PII is detected
	// +optional
	// +kubebuilder:validation:Enum=block;mask;allow
	// +kubebuilder:default=block
	Action *string `json:"action,omitempty"`
}

// PromptGuardConfig defines the configuration for prompt guard filter
type PromptGuardConfig struct {
	// Threshold defines the confidence threshold for jailbreak detection (0.0-1.0)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	// +kubebuilder:default=0.7
	Threshold *float64 `json:"threshold,omitempty"`

	// Action defines what to do when a jailbreak attempt is detected
	// +optional
	// +kubebuilder:validation:Enum=block;warn;allow
	// +kubebuilder:default=block
	Action *string `json:"action,omitempty"`

	// CustomRules defines additional custom security rules
	// +optional
	// +kubebuilder:validation:MaxItems=100
	CustomRules []SecurityRule `json:"customRules,omitempty"`
}

// SecurityRule defines a custom security rule
type SecurityRule struct {
	// Name defines the name of the security rule
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name"`

	// Pattern defines the regex pattern to match
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=1000
	Pattern string `json:"pattern"`

	// Action defines the action to take when this rule matches
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=block;warn;allow
	Action string `json:"action"`

	// Description provides an optional description of this rule
	// +optional
	// +kubebuilder:validation:MaxLength=500
	Description string `json:"description,omitempty"`
}

// SemanticCacheConfig defines the configuration for semantic cache filter
type SemanticCacheConfig struct {
	// SimilarityThreshold defines the similarity threshold for cache hits (0.0-1.0)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	// +kubebuilder:default=0.8
	SimilarityThreshold *float64 `json:"similarityThreshold,omitempty"`

	// MaxEntries defines the maximum number of cache entries
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=1000000
	// +kubebuilder:default=1000
	MaxEntries *int32 `json:"maxEntries,omitempty"`

	// TTLSeconds defines the time-to-live for cache entries in seconds
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=86400
	// +kubebuilder:default=3600
	TTLSeconds *int32 `json:"ttlSeconds,omitempty"`

	// Backend defines the cache backend type
	// +optional
	// +kubebuilder:validation:Enum=memory;redis;milvus
	// +kubebuilder:default=memory
	Backend *string `json:"backend,omitempty"`

	// BackendConfig defines backend-specific configuration
	// +optional
	BackendConfig map[string]string `json:"backendConfig,omitempty"`
}

// ReasoningControlConfig defines the configuration for reasoning control filter
type ReasoningControlConfig struct {
	// ReasonFamily defines the reasoning family to use
	// +optional
	// +kubebuilder:validation:Enum=gpt-oss;deepseek;qwen3;claude
	ReasonFamily *string `json:"reasonFamily,omitempty"`

	// EnableReasoning defines whether reasoning mode is enabled
	// +optional
	// +kubebuilder:default=true
	EnableReasoning *bool `json:"enableReasoning,omitempty"`

	// ReasoningEffort defines the reasoning effort level
	// +optional
	// +kubebuilder:validation:Enum=low;medium;high
	// +kubebuilder:default=medium
	ReasoningEffort *string `json:"reasoningEffort,omitempty"`

	// MaxReasoningSteps defines the maximum number of reasoning steps
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=10
	MaxReasoningSteps *int32 `json:"maxReasoningSteps,omitempty"`

	// ReasoningTimeout defines the timeout for reasoning in seconds
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=300
	// +kubebuilder:default=30
	ReasoningTimeout *int32 `json:"reasoningTimeout,omitempty"`
}

// ToolSelectionConfig defines the configuration for automatic tool selection filter
type ToolSelectionConfig struct {
	// TopK defines the number of top tools to select based on similarity
	// +optional
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=20
	// +kubebuilder:default=3
	TopK *int32 `json:"topK,omitempty"`

	// SimilarityThreshold defines the similarity threshold for tool selection (0.0-1.0)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	// +kubebuilder:default=0.2
	SimilarityThreshold *float64 `json:"similarityThreshold,omitempty"`

	// ToolsDBPath defines the path to the tools database file
	// +optional
	// +kubebuilder:default="config/tools_db.json"
	ToolsDBPath *string `json:"toolsDBPath,omitempty"`

	// FallbackToEmpty defines whether to return empty tools on failure
	// +optional
	// +kubebuilder:default=true
	FallbackToEmpty *bool `json:"fallbackToEmpty,omitempty"`

	// Categories defines the tool categories to include in selection
	// +optional
	// +kubebuilder:validation:MaxItems=20
	Categories []string `json:"categories,omitempty"`

	// Tags defines the tool tags to include in selection
	// +optional
	// +kubebuilder:validation:MaxItems=50
	Tags []string `json:"tags,omitempty"`
}

// FilterCondition defines a condition for applying filters
type FilterCondition struct {
	// Type defines the condition type
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=Always;Never;OnMatch;OnNoMatch
	Type FilterConditionType `json:"type"`

	// Value defines the condition value (used with OnMatch/OnNoMatch)
	// +optional
	Value string `json:"value,omitempty"`
}

// FilterConditionType defines the supported filter condition types
// +kubebuilder:validation:Enum=Always;Never;OnMatch;OnNoMatch
type FilterConditionType string

const (
	// FilterConditionAlways means the filter is always applied
	FilterConditionAlways FilterConditionType = "Always"
	// FilterConditionNever means the filter is never applied
	FilterConditionNever FilterConditionType = "Never"
	// FilterConditionOnMatch means the filter is applied when a condition matches
	FilterConditionOnMatch FilterConditionType = "OnMatch"
	// FilterConditionOnNoMatch means the filter is applied when a condition doesn't match
	FilterConditionOnNoMatch FilterConditionType = "OnNoMatch"
)
