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

// DecisionReliabilityConfig is a decision's reliability block, with the
// router configuration's field names and meaning. Durations are Go durations
// such as "30s"; "0s" turns a timeout off.
type DecisionReliabilityConfig struct {
	// TotalTimeout bounds the whole call: every attempt and the response
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	TotalTimeout string `json:"total_timeout,omitempty" yaml:"total_timeout,omitempty"`

	// PerTryTimeout bounds each attempt until its response starts
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	PerTryTimeout string `json:"per_try_timeout,omitempty" yaml:"per_try_timeout,omitempty"`

	// IdleTimeout bounds the wait for more of a streamed response
	// (standalone mode only)
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	IdleTimeout string `json:"idle_timeout,omitempty" yaml:"idle_timeout,omitempty"`

	// FirstByteTimeout bounds the wait for the first response byte
	// (standalone mode only)
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	FirstByteTimeout string `json:"first_byte_timeout,omitempty" yaml:"first_byte_timeout,omitempty"`

	// RetryCount is the number of retries after the first attempt
	// +optional
	// +kubebuilder:validation:Minimum=0
	// +kubebuilder:validation:Maximum=5
	RetryCount *int32 `json:"retry_count,omitempty" yaml:"retry_count,omitempty"`

	// RetryOn adds retry conditions, with Envoy's names (5xx, reset, ...)
	// +optional
	RetryOn string `json:"retry_on,omitempty" yaml:"retry_on,omitempty"`

	// RetriableStatusCodes adds statuses retried under retriable-status-codes
	// +optional
	// +kubebuilder:validation:items:Minimum=100
	// +kubebuilder:validation:items:Maximum=599
	RetriableStatusCodes []int32 `json:"retriable_status_codes,omitempty" yaml:"retriable_status_codes,omitempty"`

	// RetryBackOffBase is the base of the randomized exponential wait between
	// retries (standalone mode only)
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	RetryBackOffBase string `json:"retry_back_off_base,omitempty" yaml:"retry_back_off_base,omitempty"`

	// RetryBackOffMax caps that wait (standalone mode only)
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	RetryBackOffMax string `json:"retry_back_off_max,omitempty" yaml:"retry_back_off_max,omitempty"`

	// RetryAfterMax honors a response's Retry-After up to this bound
	// (standalone mode only)
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	RetryAfterMax string `json:"retry_after_max,omitempty" yaml:"retry_after_max,omitempty"`
}

// DecisionFallbackConfig is a decision's fallback block, with the router
// configuration's field names and meaning. A field left out keeps the
// recipe's or the router's value.
type DecisionFallbackConfig struct {
	// Enabled turns cross-model fallback on or off for this decision
	// +optional
	Enabled *bool `json:"enabled,omitempty" yaml:"enabled,omitempty"`

	// MaxAttempts bounds the candidates tried, the first one included
	// +optional
	// +kubebuilder:validation:Minimum=0
	MaxAttempts int32 `json:"max_attempts,omitempty" yaml:"max_attempts,omitempty"`

	// TotalTimeout bounds the whole fallback chain
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	TotalTimeout string `json:"total_timeout,omitempty" yaml:"total_timeout,omitempty"`

	// PerAttemptTimeout bounds each candidate's attempt
	// +optional
	// +kubebuilder:validation:Pattern=`^([0-9]+(\.[0-9]+)?(ns|us|ms|s|m|h))+$`
	PerAttemptTimeout string `json:"per_attempt_timeout,omitempty" yaml:"per_attempt_timeout,omitempty"`

	// RetryableStatusCodes are the statuses that move on to the next candidate
	// +optional
	// +kubebuilder:validation:items:Minimum=100
	// +kubebuilder:validation:items:Maximum=599
	RetryableStatusCodes []int32 `json:"retryable_status_codes,omitempty" yaml:"retryable_status_codes,omitempty"`
}
