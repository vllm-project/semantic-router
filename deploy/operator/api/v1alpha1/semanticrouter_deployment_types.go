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
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
)

// PodDisruptionBudgetSpec defines the managed Router disruption guard.
type PodDisruptionBudgetSpec struct {
	// Enabled overrides the mode-aware default. Managed mode defaults to true;
	// standalone mode defaults to false.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// MinAvailable is the minimum number of Router Pods kept available.
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=0
	// +optional
	MinAvailable *int32 `json:"minAvailable,omitempty"`
}

// TopologySpreadSpec defines one portable Router topology constraint.
type TopologySpreadSpec struct {
	// Enabled overrides the mode-aware default. Managed mode defaults to true;
	// standalone mode defaults to false.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// MaxSkew is the maximum permitted imbalance between topology domains.
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	// +optional
	MaxSkew int32 `json:"maxSkew,omitempty"`

	// TopologyKey selects the node label that defines a failure domain.
	// +kubebuilder:default="kubernetes.io/hostname"
	// +kubebuilder:validation:MinLength=1
	// +optional
	TopologyKey string `json:"topologyKey,omitempty"`

	// WhenUnsatisfiable controls scheduling when the constraint cannot be met.
	// +kubebuilder:default="ScheduleAnyway"
	// +kubebuilder:validation:Enum=DoNotSchedule;ScheduleAnyway
	// +optional
	WhenUnsatisfiable corev1.UnsatisfiableConstraintAction `json:"whenUnsatisfiable,omitempty"`
}

// NetworkPolicySpec defines listener-specific ingress peers. Omitted peer
// families stay denied when the policy is enabled.
type NetworkPolicySpec struct {
	// Enabled overrides the mode-aware default. Managed mode defaults to true;
	// standalone mode defaults to false.
	// +optional
	Enabled *bool `json:"enabled,omitempty"`

	// InferencePeers may reach the public ExtProc and sidecar inference ports.
	// +optional
	InferencePeers []networkingv1.NetworkPolicyPeer `json:"inferencePeers,omitempty"`

	// ManagementPeers may reach only the private Management listener.
	// +optional
	ManagementPeers []networkingv1.NetworkPolicyPeer `json:"managementPeers,omitempty"`

	// MetricsPeers may scrape only the metrics listener.
	// +optional
	MetricsPeers []networkingv1.NetworkPolicyPeer `json:"metricsPeers,omitempty"`
}

// MigrationStatus reports the explicit managed schema Job state.
type MigrationStatus struct {
	// JobName is the content-addressed migration Job for this bootstrap and
	// Router image.
	JobName string `json:"jobName"`

	// State is Pending, Running, Succeeded, or Failed.
	// +kubebuilder:validation:Enum=Pending;Running;Succeeded;Failed
	State string `json:"state"`
}
