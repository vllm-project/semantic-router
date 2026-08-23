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
	"bytes"
	"context"
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/protocolcodec"
)

// NewRequestFromSemantic constructs a Looper request from the Router's neutral
// contract. Wire shaping is delegated to the codec registry rather than
// reproduced in ExtProc.
func NewRequestFromSemantic(semantic *llmprotocol.Request) (*Request, error) {
	if semantic == nil {
		return nil, fmt.Errorf("neutral looper request is required")
	}
	encoded, err := protocolcodec.NewBuiltinEngine().EncodeRequest(
		llmprotocol.OpenAIChatV1,
		*semantic,
		llmprotocol.Envelope{},
	)
	if err != nil {
		return nil, err
	}
	var request openai.ChatCompletionNewParams
	if err := json.Unmarshal(encoded.Body, &request); err != nil {
		return nil, fmt.Errorf("decode looper execution request: %w", err)
	}
	return &Request{
		SemanticRequest:  cloneSemanticRequest(semantic),
		executionRequest: &request,
	}, nil
}

// Request contains the input for looper execution
type Request struct {
	// SemanticRequest is the protocol-neutral Router contract. It is immutable
	// after construction and is the only request representation exposed across
	// the Looper package boundary.
	SemanticRequest *llmprotocol.Request

	// executionRequest is a private model-client DTO used by the existing
	// Looper algorithms. It never crosses ingress, routing, settlement, replay,
	// or response encoding boundaries.
	executionRequest *openai.ChatCompletionNewParams

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

	// Fusion carries request-level plugins[].id=fusion overrides.
	Fusion *config.FusionRequestConfig

	// CachedPanel, when non-nil, is used verbatim as the fusion panel instead of
	// calling the analysis models. It exists for paired multi-arm evaluation where
	// every arm must synthesize from a byte-identical panel (see
	// bench/grounded_fusion). Nil in production; only the fusioneval driver sets it.
	CachedPanel []*ModelResponse
}

func cloneSemanticRequest(request *llmprotocol.Request) *llmprotocol.Request {
	if request == nil {
		return nil
	}
	cloned := *request
	cloned.Instructions = append([]llmprotocol.InstructionBlock(nil), request.Instructions...)
	for index := range cloned.Instructions {
		cloned.Instructions[index].Content = append(
			[]llmprotocol.Content(nil), request.Instructions[index].Content...,
		)
	}
	cloned.Messages = append([]llmprotocol.Message(nil), request.Messages...)
	for index := range cloned.Messages {
		cloned.Messages[index].Content = append([]llmprotocol.Content(nil), request.Messages[index].Content...)
	}
	cloned.Tools = append([]llmprotocol.Tool(nil), request.Tools...)
	cloned.Sampling.Stop = append([]string(nil), request.Sampling.Stop...)
	if bytes.Equal(bytes.TrimSpace(cloned.OutputFormat.Schema), []byte("null")) {
		cloned.OutputFormat.Schema = nil
	}
	if request.Metadata != nil {
		cloned.Metadata = make(map[string]string, len(request.Metadata))
		for key, value := range request.Metadata {
			cloned.Metadata[key] = value
		}
	}
	return &cloned
}

// Response contains the output from looper execution
type Response struct {
	// Semantic is the only response representation exposed outside Looper.
	// Public wire encoding belongs to the shared protocol codec engine.
	Semantic *llmprotocol.Response

	// Streaming preserves the client's requested delivery mode. Looper itself
	// remains buffered; ExtProc renders Semantic as semantic stream events.
	Streaming bool

	// IncludeUsage is a protocol-neutral delivery preference for streaming
	// responses. Accounting always uses the backend response terminal; this flag
	// controls only whether the synthesized client stream exposes aggregate
	// usage.
	IncludeUsage bool

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

	// Usage is the aggregate reported usage across all model calls made during
	// this execution. It is presentation metadata only; dispatch settlement is
	// driven by BackendInvoker response terminals.
	Usage TokenUsage `json:"usage,omitempty"`

	// LatencyMs is the wall-clock latency, in milliseconds, of the full
	// looper execution (all model calls plus algorithm overhead). It is set
	// by ExecuteWithLatency rather than by individual Looper implementations,
	// so it reflects real elapsed time regardless of whether the algorithm
	// dispatches its model calls sequentially or concurrently.
	LatencyMs int64 `json:"latency_ms,omitempty"`
}

// Looper defines the interface for multi-model execution strategies
type Looper interface {
	// Execute runs the looper algorithm and returns an aggregated response
	Execute(ctx context.Context, req *Request) (*Response, error)
}

// UnsupportedAlgorithmError reports an algorithm that cannot be constructed
// by the Looper runtime.
type UnsupportedAlgorithmError struct {
	AlgorithmType string
}

func (e *UnsupportedAlgorithmError) Error() string {
	return fmt.Sprintf("unsupported Looper algorithm %q", e.AlgorithmType)
}

type algorithmConstructor func(*config.LooperConfig) Looper

var algorithmConstructors = map[string]algorithmConstructor{
	config.DecisionAlgorithmConfidence: func(cfg *config.LooperConfig) Looper {
		return NewConfidenceLooper(cfg)
	},
	config.DecisionAlgorithmFusion: func(cfg *config.LooperConfig) Looper {
		return NewFusionLooper(cfg)
	},
	config.DecisionAlgorithmRatings: func(cfg *config.LooperConfig) Looper {
		return NewRatingsLooper(cfg)
	},
	config.DecisionAlgorithmReMoM: func(cfg *config.LooperConfig) Looper {
		return NewReMoMLooper(cfg)
	},
	config.DecisionAlgorithmWorkflows: func(cfg *config.LooperConfig) Looper {
		return NewWorkflowsLooper(cfg)
	},
}

// Factory creates a Looper instance based on the authoritative config catalog.
func Factory(cfg *config.LooperConfig, algorithmType string) (Looper, error) {
	constructor, ok := algorithmConstructors[algorithmType]
	if !config.IsLooperAlgorithmType(algorithmType) || !ok {
		return nil, &UnsupportedAlgorithmError{AlgorithmType: algorithmType}
	}
	return constructor(cfg), nil
}
