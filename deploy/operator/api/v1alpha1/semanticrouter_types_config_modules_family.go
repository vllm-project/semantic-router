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

// ToolsConfig defines tools configuration
type ToolsConfig struct {
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default=3
	// +optional
	TopK int `json:"top_k,omitempty"`
	// Similarity threshold for tool selection (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.2"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	SimilarityThreshold string `json:"similarity_threshold,omitempty"`
	// +kubebuilder:default="config/tools_db.json"
	// +optional
	ToolsDBPath string `json:"tools_db_path,omitempty"`
	// +kubebuilder:default=true
	// +optional
	FallbackToEmpty bool `json:"fallback_to_empty,omitempty"`
}

// PromptGuardConfig defines prompt guard configuration
type PromptGuardConfig struct {
	// +kubebuilder:default=true
	// +optional
	Enabled bool `json:"enabled,omitempty"`
	// +kubebuilder:default=false
	// +optional
	UseModernBERT bool `json:"use_modernbert,omitempty"`
	// +kubebuilder:default="models/mmbert32k-jailbreak-detector-merged"
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// Jailbreak detection threshold (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:default="0.7"
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +kubebuilder:default=true
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
	// +optional
	JailbreakMappingPath string `json:"jailbreak_mapping_path,omitempty"`
}

// ClassifierConfig defines classifier configuration
type ClassifierConfig struct {
	// +optional
	CategoryModel *CategoryModelConfig `json:"category_model,omitempty"`
	// +optional
	PIIModel *PIIModelConfig `json:"pii_model,omitempty"`
}

// CategoryModelConfig defines category model configuration
type CategoryModelConfig struct {
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// +optional
	UseModernBERT bool `json:"use_modernbert,omitempty"`
	// Classification threshold (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
	// +optional
	CategoryMappingPath string `json:"category_mapping_path,omitempty"`
}

// PIIModelConfig defines PII model configuration
type PIIModelConfig struct {
	// +optional
	ModelID string `json:"model_id,omitempty"`
	// +optional
	UseModernBERT bool `json:"use_modernbert,omitempty"`
	// Detection threshold (0.0-1.0). Stored as string to avoid float precision issues.
	// +kubebuilder:validation:Pattern=`^0(\.[0-9]+)?$|^1(\.0+)?$`
	// +optional
	Threshold string `json:"threshold,omitempty"`
	// +optional
	UseCPU bool `json:"use_cpu,omitempty"`
	// +optional
	PIIMappingPath string `json:"pii_mapping_path,omitempty"`
}
