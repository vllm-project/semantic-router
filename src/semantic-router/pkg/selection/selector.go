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

// Package selection implements the selectors, which choose one of a
// decision's candidate models before it is called: static, random,
// multi_factor and systemone, and the session-aware policy Router Learning
// applies over them.
package selection

import (
	"context"
	"errors"
	"io"
	"sync"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selectiontrace"
)

// SelectionMethod defines the type of model selection algorithm
type SelectionMethod string

const (
	// MethodStatic uses static scores from configuration (default behavior)
	MethodStatic SelectionMethod = "static"

	// MethodMultiFactor combines quality/latency/cost/load signals via a
	// weighted score with optional SLO ceilings. Issue #37.
	MethodMultiFactor SelectionMethod = "multi_factor"

	// MethodRandom picks uniformly from the eligible candidates. No learned state.
	MethodRandom SelectionMethod = "random"

	// MethodSystemOne asks a System One model to choose one declared
	// candidate; its probabilities are the candidate scores.
	MethodSystemOne SelectionMethod = "systemone"

	// MethodSessionAware wraps a base selector with agentic session policy:
	// it keeps tool loops and hot multi-turn continuations on the current model
	// unless the switch benefit clears the explicit handoff and prefix-cache cost.
	MethodSessionAware SelectionMethod = "session_aware"
)

// AlgorithmTier classifies algorithms by production readiness
type AlgorithmTier string

const (
	// TierSupported indicates a production-ready algorithm with no surprise dependencies
	TierSupported AlgorithmTier = "supported"

	// TierExperimental indicates a research-grade algorithm that may require
	// external services or has limited production validation
	TierExperimental AlgorithmTier = "experimental"
)

// DependencyType classifies external dependencies
type DependencyType string

const (
	// DependencyExternalService requires a separate running service (HTTP server, etc.)
	DependencyExternalService DependencyType = "external_service"
)

// Dependency describes an external dependency required by an algorithm
type Dependency struct {
	// Name is a human-readable name (e.g., "Decision model runtime")
	Name string

	// Type classifies the dependency
	Type DependencyType

	// Description explains what the dependency is used for
	Description string

	// HealthURL is an optional URL to check at startup (for external services)
	HealthURL string

	// Required indicates the algorithm degrades without this dependency
	Required bool
}

// SelectionContext provides context for model selection decisions
type SelectionContext struct {
	// Query is the user's input query text
	Query string

	// ConversationHistory provides prior messages for context-aware selection
	ConversationHistory []string

	// DecisionName is the name of the matched decision for category-specific selection
	DecisionName string

	// RecipeName identifies the isolated routing profile that owns DecisionName.
	// Selectors are instantiated per recipe; shared learning/lookup components
	// use both fields as their state namespace.
	RecipeName config.RecipeName

	// CategoryName is the detected domain category (e.g., "physics", "math")
	// Used by the static selector's category scores and the session policy
	CategoryName string

	// CandidateModels is the list of models to select from
	CandidateModels []config.ModelRef

	// CandidateIterations carries bounded DSL FOR ... IN metadata for selectors
	// that opt into session-aware candidate policy evaluation.
	CandidateIterations []config.CandidateIterationConfig

	// InputTokens is the request-context token estimate available before model
	// selection. ExpectedOutputTokens is the caller's output budget when present.
	// Cost-aware selectors use both to compare request-shaped estimated cost.
	InputTokens          int
	ExpectedOutputTokens int
	// CandidateDemands contains provider-rendered budgets for automatic output.
	CandidateDemands map[string]CandidateDemand

	// UserID identifies the user for session telemetry and learning traces (optional)
	UserID string

	// SessionID identifies the conversation session for multi-turn context (optional)
	// Used to track within-session model performance
	SessionID string

	// SessionStateKey is the canonical recipe-scoped router memory key. It is
	// separate from SessionID so client identity text cannot be mistaken for an
	// encoded session/conversation tuple. An empty key uses the raw SessionID.
	SessionStateKey string

	// AgenticSession carries request-time session facts used by
	// session_aware selection. The flat SessionID remains the shared
	// correlation key for selectors that do not need richer session facts.
	AgenticSession *AgenticSessionContext

	// CacheAffinityCtx carries request-time session signals for cache-affinity estimation.
	CacheAffinityCtx *CacheAffinityContext
}

// ScopedRoutingName namespaces recipe-local task-family names for shared
// lookup-table state while keeping default-recipe keys unscoped.
func (c *SelectionContext) ScopedRoutingName(localName string) string {
	if c == nil {
		return ""
	}
	return config.RoutingNamespaceKey(c.RecipeName, localName)
}

// RoutingScope returns the lookup-table namespace for this recipe. The
// default recipe keeps unscoped keys.
func (c *SelectionContext) RoutingScope() string {
	if c == nil {
		return ""
	}
	return config.RoutingNamespaceScope(c.RecipeName)
}

// SelectionResult contains the result of a model selection decision
type SelectionResult struct {
	// SelectedModel is the name of the selected model
	SelectedModel string

	// SelectedCandidate preserves the exact winning ModelRef when a model appears
	// more than once with different candidate-level settings.
	SelectedCandidate *config.ModelRef

	// LoRAName is the LoRA adapter name to use (if applicable)
	LoRAName string

	// Score is the selection score for the chosen model
	Score float64

	// Confidence indicates how confident the selector is in this choice
	Confidence float64

	// Method indicates which selection method was used
	Method SelectionMethod

	// Tier indicates the production readiness of the algorithm that made this selection
	Tier AlgorithmTier

	// Reasoning provides human-readable explanation for the selection
	Reasoning string

	// EligibleModels is the selector's explicit policy envelope. Nil means the
	// selector only ranks candidates; a non-nil set contains every model still
	// allowed after hard filters (or the explicitly configured fallback).
	// Later learning and provider rerouting must not expand this set.
	EligibleModels []config.ModelRef

	// CandidateScores is the typed input for composition and policy. AllScores
	// is its compatibility/diagnostic projection, never a candidate identity.
	CandidateScores CandidateScores
	ScoreDirection  ScoreDirection
	AllScores       map[string]float64

	// MultiFactor records the base objective stages and their eligible survivors.
	// Later learning/session policy may choose among them; this is not a final
	// dispatch trace. Weighted objectives omit it.
	MultiFactor *selectiontrace.MultiFactorObjective

	// SessionPolicy records the session-aware stay/switch policy trace when
	// Method is session_aware.
	SessionPolicy *SessionPolicyTrace
}

// Selector is the interface for model selection algorithms
type Selector interface {
	// Select chooses the best model from candidates based on the selection context
	Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error)

	// Method returns the selection method type
	Method() SelectionMethod

	// Tier returns the production readiness classification of this algorithm
	Tier() AlgorithmTier

	// ExternalDependencies returns the list of external dependencies this algorithm requires
	ExternalDependencies() []Dependency
}

// Registry maintains available selection methods and their configurations
type Registry struct {
	selectors map[SelectionMethod]Selector
	mu        sync.RWMutex
}

// NewRegistry creates a new selector registry
func NewRegistry() *Registry {
	return &Registry{
		selectors: make(map[SelectionMethod]Selector),
	}
}

// Register adds a selector to the registry
func (r *Registry) Register(method SelectionMethod, selector Selector) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.selectors[method] = selector
}

// Get retrieves a selector by method type
func (r *Registry) Get(method SelectionMethod) (Selector, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	s, ok := r.selectors[method]
	return s, ok
}

func (r *Registry) Close() error {
	if r == nil {
		return nil
	}

	r.mu.RLock()
	closers := make([]io.Closer, 0, len(r.selectors))
	for _, selector := range r.selectors {
		if closer, ok := selector.(io.Closer); ok {
			closers = append(closers, closer)
		}
	}
	r.mu.RUnlock()

	var errs []error
	for _, closer := range closers {
		if err := closer.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Join(errs...)
}

var globalRegistry atomic.Pointer[Registry]

func init() {
	globalRegistry.Store(NewRegistry())
}

// GetGlobalRegistry returns the process-wide default selection registry.
func GetGlobalRegistry() *Registry {
	return globalRegistry.Load()
}

// SetGlobalRegistry replaces the process-wide default selection registry.
func SetGlobalRegistry(registry *Registry) {
	globalRegistry.Store(registry)
}

// Select uses the specified method to select a model
func Select(ctx context.Context, method SelectionMethod, selCtx *SelectionContext) (*SelectionResult, error) {
	if err := ValidateSelectionContext(selCtx); err != nil {
		return nil, err
	}

	registry := GetGlobalRegistry()
	selector, ok := registry.Get(method)
	if !ok {
		// Default to static selection when the requested method is not registered.
		selector, _ = registry.Get(MethodStatic)
	}
	if selector == nil {
		// Last-resort default: return the first configured candidate.
		return &SelectionResult{
			SelectedModel: selCtx.CandidateModels[0].Model,
			LoRAName:      selCtx.CandidateModels[0].LoRAName,
			Score:         1.0,
			Confidence:    1.0,
			Method:        MethodStatic,
			Tier:          TierSupported,
			Reasoning:     "No selector available, using first candidate",
		}, nil
	}
	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		return nil, err
	}
	if err := ValidateSelectionResult(selCtx, result); err != nil {
		return nil, err
	}
	result.Tier = selector.Tier()
	return result, nil
}
