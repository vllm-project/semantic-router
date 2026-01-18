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

package controllers

const (
	// SemanticRouterFinalizer is the finalizer for SemanticRouter resources
	SemanticRouterFinalizer = "semanticrouter.vllm.ai/finalizer"
)

// Default values for SemanticRouter resources
const (
	// DefaultImage is the default container image for semantic router
	DefaultImage = "ghcr.io/vllm-project/semantic-router/extproc:latest"

	// DefaultReplicas is the default number of replicas
	DefaultReplicas = int32(1)

	// DefaultPVCSize is the default PVC storage size
	DefaultPVCSize = "10Gi"

	// Port numbers
	DefaultGRPCPort    = int32(50051)
	DefaultAPIPort     = int32(8080)
	DefaultMetricsPort = int32(9190)

	// Probe defaults
	DefaultStartupProbePeriod           = int32(10)
	DefaultStartupProbeTimeout          = int32(5)
	DefaultStartupProbeFailureThreshold = int32(360)

	DefaultLivenessProbePeriod           = int32(30)
	DefaultLivenessProbeTimeout          = int32(10)
	DefaultLivenessProbeFailureThreshold = int32(5)
	DefaultLivenessProbeInitialDelay     = int32(30)

	DefaultReadinessProbePeriod           = int32(30)
	DefaultReadinessProbeTimeout          = int32(10)
	DefaultReadinessProbeFailureThreshold = int32(5)
	DefaultReadinessProbeInitialDelay     = int32(30)

	// HPA defaults
	DefaultHPAMinReplicas = int32(1)
	DefaultHPAMaxReplicas = int32(10)

	// Security defaults
	DefaultRunAsUser    = int64(1000)
	DefaultFSGroup      = int64(1000)
	DefaultRunAsNonRoot = true
	DefaultAllowPrivEsc = false
)
