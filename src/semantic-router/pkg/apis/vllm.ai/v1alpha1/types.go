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

// SemanticRoute defines a semantic routing rule for LLM requests
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=sr
// +kubebuilder:printcolumn:name="Rules",type="integer",JSONPath=".spec.rules",description="Number of routing rules"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type SemanticRoute struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   SemanticRouteSpec   `json:"spec,omitempty"`
	Status SemanticRouteStatus `json:"status,omitempty"`
}

// SemanticRouteSpec defines the desired state of SemanticRoute
type SemanticRouteSpec struct {
	// Rules defines the routing rules to be applied
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=100
	Rules []RouteRule `json:"rules"`
}

// SemanticRouteStatus defines the observed state of SemanticRoute
type SemanticRouteStatus struct {
	// Conditions represent the latest available observations of the SemanticRoute's current state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ObservedGeneration reflects the generation of the most recently observed SemanticRoute
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// ActiveRules indicates the number of currently active routing rules
	// +optional
	ActiveRules int32 `json:"activeRules,omitempty"`
}

// RouteRule defines a single routing rule
type RouteRule struct {
	// Intents defines the intent categories that this rule should match
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=50
	Intents []Intent `json:"intents"`

	// ModelRefs defines the target models for this routing rule
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=10
	ModelRefs []ModelRef `json:"modelRefs"`

	// Filters defines the optional filters to be applied to requests matching this rule
	// +optional
	// +kubebuilder:validation:MaxItems=20
	Filters []Filter `json:"filters,omitempty"`

	// DefaultModel defines the fallback model if no modelRefs are available
	// +optional
	DefaultModel *ModelRef `json:"defaultModel,omitempty"`
}

// Intent defines an intent category for routing
type Intent struct {
	// Category defines the intent category name (e.g., "math", "computer science", "creative")
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	// +kubebuilder:validation:Pattern=^[a-zA-Z0-9\s\-_]+$
	Category string `json:"category"`

	// Description provides an optional description of this intent category
	// +optional
	// +kubebuilder:validation:MaxLength=500
	Description string `json:"description,omitempty"`

	// Threshold defines the confidence threshold for this intent (0.0-1.0)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1
	// +kubebuilder:default=0.7
	Threshold *float64 `json:"threshold,omitempty"`
}

// ModelRef defines a reference to a model endpoint
type ModelRef struct {
	// ModelName defines the name of the model
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	ModelName string `json:"modelName"`

	// Address defines the endpoint address
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=255
	Address string `json:"address"`

	// Port defines the endpoint port
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=1
	// +kubebuilder:validation:Maximum=65535
	Port int32 `json:"port"`

	// Weight defines the traffic weight for this model (0-100)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=100
	// +kubebuilder:default=100
	Weight *int32 `json:"weight,omitempty"`

	// Priority defines the priority of this model reference (higher values = higher priority)
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=1000
	Priority *int32 `json:"priority,omitempty"`
}

// Filter defines a filter to be applied to requests
type Filter struct {
	// Type defines the filter type
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Enum=PIIDetection;PromptGuard;SemanticCache;ReasoningControl
	Type FilterType `json:"type"`

	// Config defines the filter-specific configuration
	// +optional
	Config *runtime.RawExtension `json:"config,omitempty"`

	// Enabled defines whether this filter is enabled
	// +optional
	// +kubebuilder:default=true
	Enabled *bool `json:"enabled,omitempty"`
}

// FilterType defines the supported filter types
// +kubebuilder:validation:Enum=PIIDetection;PromptGuard;SemanticCache;ReasoningControl;ToolSelection
type FilterType string

const (
	// FilterTypePIIDetection enables PII detection and filtering
	FilterTypePIIDetection FilterType = "PIIDetection"
	// FilterTypePromptGuard enables prompt security and jailbreak detection
	FilterTypePromptGuard FilterType = "PromptGuard"
	// FilterTypeSemanticCache enables semantic caching for performance optimization
	FilterTypeSemanticCache FilterType = "SemanticCache"
	// FilterTypeReasoningControl enables reasoning mode control
	FilterTypeReasoningControl FilterType = "ReasoningControl"
	// FilterTypeToolSelection enables automatic tool selection based on semantic similarity
	FilterTypeToolSelection FilterType = "ToolSelection"
)

// SemanticRouteList contains a list of SemanticRoute
// +kubebuilder:object:root=true
type SemanticRouteList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []SemanticRoute `json:"items"`
}
