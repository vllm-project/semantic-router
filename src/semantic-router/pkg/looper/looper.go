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
	"sync"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Request contains the input for looper execution
type Request struct {
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

// WorkflowStateService is an opaque handle to a shared workflow tool-state
// store. Create one per router generation with NewWorkflowStateService and pass
// it to FactoryWithWorkflowState so independent HTTP turns share pause/resume
// state. Safe for concurrent use.
type WorkflowStateService struct {
	store  workflowToolStateStore
	wg     sync.WaitGroup
	mu     sync.RWMutex
	closed bool
}

// Acquire tries to get a read lease on the service. Returns false if closed.
//
// Safety invariant: wg.Add(1) is called while holding RLock. This is safe
// because Close() sets s.closed = true under a write lock *before* calling
// wg.Wait(). Once closed is true, no new Add(1) can happen, so Wait() will
// observe a stable counter. Do not add a second Close() codepath without
// preserving this ordering.
func (s *WorkflowStateService) Acquire() bool {
	if s == nil {
		return false
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.closed {
		return false
	}
	s.wg.Add(1)
	return true
}

// Release releases a read lease on the service.
func (s *WorkflowStateService) Release() {
	if s != nil {
		s.wg.Done()
	}
}

// Store returns the underlying state store. Safe to use only while holding a lease.
func (s *WorkflowStateService) Store() workflowToolStateStore {
	if s == nil {
		return nil
	}
	return s.store
}

// NewWorkflowStateService creates a shared workflow state store from the
// looper configuration. The returned service should be stored on the router
// and passed into every FactoryWithWorkflowState call.
func NewWorkflowStateService(cfg *config.LooperConfig) *WorkflowStateService {
	if cfg == nil {
		return nil
	}
	return &WorkflowStateService{
		store: newWorkflowToolStateStoreFromConfig(workflowFlowRuntimeConfig(cfg)),
	}
}

// Close releases resources held by the state service (e.g. Redis connections).
func (s *WorkflowStateService) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return nil
	}
	s.closed = true
	s.mu.Unlock()

	s.wg.Wait()
	if s.store != nil {
		return s.store.Close()
	}
	return nil
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
	return FactoryWithWorkflowState(cfg, algorithmType, nil)
}

// FactoryWithWorkflowState creates a Looper and, for workflows, shares the
// generation-owned tool-state store across independent requests.
func FactoryWithWorkflowState(
	cfg *config.LooperConfig,
	algorithmType string,
	stateService *WorkflowStateService,
) (Looper, error) {
	if !config.IsLooperAlgorithmType(algorithmType) {
		return nil, &UnsupportedAlgorithmError{AlgorithmType: algorithmType}
	}
	if algorithmType == config.DecisionAlgorithmWorkflows {
		return newWorkflowsLooperWithService(cfg, stateService), nil
	}
	constructor, ok := algorithmConstructors[algorithmType]
	if !ok {
		return nil, &UnsupportedAlgorithmError{AlgorithmType: algorithmType}
	}
	return constructor(cfg), nil
}
