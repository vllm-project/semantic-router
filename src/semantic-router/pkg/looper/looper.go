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

// Package looper provides multi-model execution strategies for LLM routing.
// It enables executing requests against multiple models with various algorithms
// (confidence, ratings, cost-aware) and aggregating the results.
package looper

import (
	"context"
	"fmt"
	"io"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Request contains the input for looper execution
type Request struct {
	Grounding *GroundingBackends
	// OriginalRequest is the OpenAI chat completion request from the client
	OriginalRequest *openai.ChatCompletionNewParams

	// BaseContextTokens is the Router's conservative estimate for the original
	// request. Generated Looper stages add their own prompt growth before every
	// backend dispatch and re-check the target model's context window.
	BaseContextTokens int

	// ModelRefs contains the list of models to potentially use, ordered by preference
	ModelRefs []config.ModelRef

	// ModelParams maps model names to their ModelParams configuration
	// Used to lookup access_key and param_size for confidence routing
	ModelParams map[string]config.ModelParams

	// Algorithm defines the execution strategy
	Algorithm *config.AlgorithmConfig

	// IsStreaming indicates if the client expects a streaming response
	IsStreaming bool

	// DecisionName is the name of the decision that triggered this looper execution
	// Used by extproc to lookup decision configuration and apply plugins
	DecisionName string

	// RecipeName is the routing namespace that owns DecisionName. It is
	// propagated on router-generated model calls so internal requests retain
	// the parent request's recipe scope.
	RecipeName config.RecipeName

	// OutputContract is the decision-scoped final response contract. The looper
	// merges it with any output format already present in the original request.
	OutputContract string

	// OutputContractSpec is the typed router-executable contract for output
	// normalization and post-processing. OutputContract remains prompt text.
	OutputContractSpec *config.OutputContractSpec

	// Fusion carries optional call-level configuration for direct Looper
	// callers. A recipe-owned Fusion algorithm accepts only its trace-visibility
	// fields; an algorithm-free internal call may use the complete configuration.
	Fusion *config.FusionRequestConfig

	// CachedPanel, when non-nil, replaces live Fusion analysis-model calls. Its
	// entries still undergo the normal usability and quorum checks before every
	// arm synthesizes from the same retained panel (see bench/grounded_fusion).
	// Nil in production; only the fusioneval driver sets it.
	CachedPanel []*ModelResponse
}

// Response contains the output from looper execution
type Response struct {
	// Body is the response body (JSON for non-streaming, SSE for streaming)
	Body []byte

	// ContentType is "application/json" or "text/event-stream"
	ContentType string

	// Model is the name of the model that produced the final response
	Model string

	// ModelsUsed tracks all models that were called during execution
	ModelsUsed []string

	// Iterations indicates how many model calls were made
	Iterations int

	// AlgorithmType indicates which algorithm was used
	AlgorithmType string

	// Logprobs contains the logprobs from the final response (if available)
	Logprobs []float64

	// IntermediateResponses contains intermediate responses from multi-round algorithms (e.g., ReMoM)
	// This is used for visualization in the dashboard
	IntermediateResponses interface{} `json:"intermediate_responses,omitempty"`

	// Usage is the aggregated token usage across all model calls made during
	// this execution. It mirrors the usage block embedded in Body so callers
	// (extproc, dashboard, metrics) can read totals without re-parsing the body.
	Usage TokenUsage `json:"usage,omitempty"`

	// LatencyMs is the wall-clock latency, in milliseconds, of the full
	// looper execution (all model calls plus algorithm overhead). It is set
	// by ExecuteWithLatency rather than by individual Looper implementations,
	// so it reflects real elapsed time regardless of whether the algorithm
	// dispatches its model calls sequentially or concurrently.
	LatencyMs int64 `json:"latency_ms,omitempty"`

	// ExecutionTrace is bounded, content-free diagnostic evidence for tracing
	// and Router Replay. It is not included in the client response body.
	ExecutionTrace ExecutionTrace `json:"-"`
}

// Looper defines the interface for multi-model execution strategies
type Looper interface {
	// Execute runs the looper algorithm and returns an aggregated response
	Execute(ctx context.Context, req *Request) (*Response, error)
}

// ManagedLooper owns the resources created by Factory and must be closed by
// its caller. Loopers built with FactoryWithClient borrow the supplied client.
type ManagedLooper interface {
	Looper
	io.Closer
}

// UnsupportedAlgorithmError reports an algorithm that cannot be constructed
// by the Looper runtime.
type UnsupportedAlgorithmError struct {
	AlgorithmType string
}

func (e *UnsupportedAlgorithmError) Error() string {
	return fmt.Sprintf("unsupported Looper algorithm %q", e.AlgorithmType)
}

type algorithmConstructor func(*config.LooperConfig, clientBinding) ManagedLooper

var algorithmConstructors = map[string]algorithmConstructor{
	config.DecisionAlgorithmConfidence: func(cfg *config.LooperConfig, binding clientBinding) ManagedLooper {
		return newConfidenceLooper(cfg, binding)
	},
	config.DecisionAlgorithmFusion: func(cfg *config.LooperConfig, binding clientBinding) ManagedLooper {
		return newFusionLooper(cfg, binding)
	},
	config.DecisionAlgorithmRatings: func(cfg *config.LooperConfig, binding clientBinding) ManagedLooper {
		return newRatingsLooper(cfg, binding)
	},
	config.DecisionAlgorithmReMoM: func(cfg *config.LooperConfig, binding clientBinding) ManagedLooper {
		return newReMoMLooper(cfg, binding)
	},
	config.DecisionAlgorithmWorkflows: func(cfg *config.LooperConfig, binding clientBinding) ManagedLooper {
		return newWorkflowsLooper(cfg, binding)
	},
}

// Factory creates a Looper instance based on the authoritative config catalog.
func Factory(cfg *config.LooperConfig, algorithmType string) (ManagedLooper, error) {
	constructor, err := constructorFor(algorithmType)
	if err != nil {
		return nil, err
	}
	client, err := NewConnectorClient(cfg)
	if err != nil {
		return nil, err
	}
	return constructor(cfg, ownClient(client)), nil
}

// FactoryWithClient creates a Looper that reuses the supplied client.
func FactoryWithClient(cfg *config.LooperConfig, algorithmType string, client *Client) (Looper, error) {
	constructor, err := constructorFor(algorithmType)
	if err != nil {
		return nil, err
	}
	if client == nil {
		return nil, fmt.Errorf("looper client is required")
	}
	return constructor(cfg, borrowClient(client)), nil
}

func constructorFor(algorithmType string) (algorithmConstructor, error) {
	constructor, ok := algorithmConstructors[algorithmType]
	if !config.IsLooperAlgorithmType(algorithmType) || !ok {
		return nil, &UnsupportedAlgorithmError{AlgorithmType: algorithmType}
	}
	return constructor, nil
}
