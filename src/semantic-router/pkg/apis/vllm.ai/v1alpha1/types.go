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
)

// IntelligentPool defines a pool of models with their configurations
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=ipool
// +kubebuilder:printcolumn:name="Default Model",type="string",JSONPath=".spec.defaultModel",description="Default model name"
// +kubebuilder:printcolumn:name="Models",type="integer",JSONPath=".status.modelCount",description="Number of models"
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].status",description="Ready status"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type IntelligentPool struct {
	metav1.TypeMeta   `json:",inline" yaml:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" yaml:"metadata,omitempty"`

	Spec   IntelligentPoolSpec   `json:"spec,omitempty" yaml:"spec,omitempty"`
	Status IntelligentPoolStatus `json:"status,omitempty" yaml:"status,omitempty"`
}

// IntelligentPoolSpec defines the desired state of IntelligentPool
type IntelligentPoolSpec struct {
	// DefaultModel specifies the default model to use when no specific model is selected
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	DefaultModel string `json:"defaultModel" yaml:"defaultModel"`

	// Models defines the list of available models in this pool
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinItems=1
	// +kubebuilder:validation:MaxItems=100
	Models []ModelConfig `json:"models" yaml:"models"`
}

// ModelConfig defines the configuration for a single model
type ModelConfig struct {
	// Name is the unique identifier for this model
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// ReasoningFamily specifies the reasoning syntax family (e.g., "qwen3", "deepseek")
	// Must be defined in the global static configuration's ReasoningFamilies
	// +optional
	// +kubebuilder:validation:MaxLength=50
	ReasoningFamily string `json:"reasoningFamily,omitempty" yaml:"reasoningFamily,omitempty"`

	// Pricing defines the cost structure for this model
	// +optional
	Pricing *ModelPricing `json:"pricing,omitempty" yaml:"pricing,omitempty"`

	// LoRAs defines the list of LoRA adapters available for this model
	// +optional
	// +kubebuilder:validation:MaxItems=50
	LoRAs []LoRAConfig `json:"loras,omitempty" yaml:"loras,omitempty"`
}

// ModelPricing defines the pricing structure for a model
type ModelPricing struct {
	// InputTokenPrice is the cost per input token
	// +optional
	// +kubebuilder:validation:Minimum=0
	InputTokenPrice float64 `json:"inputTokenPrice,omitempty" yaml:"inputTokenPrice,omitempty"`

	// OutputTokenPrice is the cost per output token
	// +optional
	// +kubebuilder:validation:Minimum=0
	OutputTokenPrice float64 `json:"outputTokenPrice,omitempty" yaml:"outputTokenPrice,omitempty"`
}

// LoRAConfig defines a LoRA adapter configuration
type LoRAConfig struct {
	// Name is the unique identifier for this LoRA adapter
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:MaxLength=100
	Name string `json:"name" yaml:"name"`

	// Description provides a human-readable description of this LoRA adapter
	// +optional
	// +kubebuilder:validation:MaxLength=500
	Description string `json:"description,omitempty" yaml:"description,omitempty"`
}

// IntelligentPoolStatus defines the observed state of IntelligentPool
type IntelligentPoolStatus struct {
	// Conditions represent the latest available observations of the IntelligentPool's state
	// +optional
	Conditions []metav1.Condition `json:"conditions,omitempty"`

	// ObservedGeneration reflects the generation of the most recently observed IntelligentPool
	// +optional
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`

	// ModelCount indicates the number of models in the pool
	// +optional
	ModelCount int32 `json:"modelCount,omitempty"`
}

// IntelligentPoolList contains a list of IntelligentPool
// +kubebuilder:object:root=true
type IntelligentPoolList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []IntelligentPool `json:"items"`
}

// IntelligentRoute defines intelligent routing rules and decisions
// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:scope=Namespaced,shortName=iroute
// +kubebuilder:printcolumn:name="Decisions",type="integer",JSONPath=".status.statistics.decisions",description="Number of decisions"
// +kubebuilder:printcolumn:name="Keywords",type="integer",JSONPath=".status.statistics.keywords",description="Number of keyword signals"
// +kubebuilder:printcolumn:name="Embeddings",type="integer",JSONPath=".status.statistics.embeddings",description="Number of embedding signals"
// +kubebuilder:printcolumn:name="Domains",type="integer",JSONPath=".status.statistics.domains",description="Number of domain signals"
// +kubebuilder:printcolumn:name="Status",type="string",JSONPath=".status.conditions[?(@.type=='Ready')].status",description="Ready status"
// +kubebuilder:printcolumn:name="Age",type="date",JSONPath=".metadata.creationTimestamp"
type IntelligentRoute struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   IntelligentRouteSpec   `json:"spec,omitempty"`
	Status IntelligentRouteStatus `json:"status,omitempty"`
}
