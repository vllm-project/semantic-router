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

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

// Request contains the input for looper execution
type Request struct {
	// OriginalRequest is the OpenAI chat completion request from the client
	OriginalRequest *openai.ChatCompletionNewParams

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

	// OutputContract is the decision-scoped final response contract. The looper
	// merges it with any output format already present in the original request.
	OutputContract string

	// OutputContractSpec is the typed router-executable contract for output
	// normalization and post-processing. OutputContract remains prompt text.
	OutputContractSpec *config.OutputContractSpec

	// Fusion carries request-level plugins[].id=fusion overrides.
	Fusion *config.FusionRequestConfig

	// CachedPanel, when non-nil, is used verbatim as the fusion panel instead of
	// calling the analysis models. It exists for paired multi-arm evaluation where
	// every arm must synthesize from a byte-identical panel (see
	// bench/grounded_fusion). Nil in production; only the fusioneval driver sets it.
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
}

// Looper defines the interface for multi-model execution strategies
type Looper interface {
	// Execute runs the looper algorithm and returns an aggregated response
	Execute(ctx context.Context, req *Request) (*Response, error)
}

// Factory creates a Looper instance based on the algorithm type
func Factory(cfg *config.LooperConfig, algorithmType string) Looper {
	return FactoryWithSelectionRegistry(cfg, algorithmType, nil)
}

// FactoryWithSelectionRegistry creates a Looper using runtime-owned model
// selectors when an algorithm needs selector state.
func FactoryWithSelectionRegistry(
	cfg *config.LooperConfig,
	algorithmType string,
	selectorRegistry *selection.Registry,
) Looper {
	switch algorithmType {
	case "confidence":
		return NewConfidenceLooper(cfg)
	case "ratings":
		return NewRatingsLooper(cfg)
	case "remom":
		return NewReMoMLooper(cfg)
	case "fusion":
		return NewFusionLooper(cfg)
	case "workflows":
		return NewWorkflowsLooper(cfg)
	case "rl_driven":
		return NewRLDrivenLooperWithSelectionRegistry(cfg, selectorRegistry)
	default:
		// Default to simple looper that just calls models sequentially
		return NewBaseLooper(cfg)
	}
}
