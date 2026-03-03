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

package selection

import (
	"context"
	"fmt"
	"math"
	"strings"
	"sync"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// PRISM integration modes define when legitimacy verification occurs relative to model selection.
// Reference: PRISM — Protocol for Routed Intelligent Specialized Models (Zenodo: 10.5281/zenodo.18750029)
const (
	// PRISMModeFineFilter applies verification after model selection (Option 1).
	// The selected model must prove it is legitimate to respond.
	// Low integration complexity; re-routing possible on refusal.
	PRISMModeFineFilter = "fine_filter"

	// PRISMModeCoarseFilter applies verification before model selection (Option 2).
	// Only pre-qualified models enter the selection pool.
	// No GPU resources consumed on illegitimate requests.
	PRISMModeCoarseFilter = "coarse_filter"

	// PRISMModeHybrid applies verification both before and after selection (Option 3).
	// Double guarantee: coarse filter + fine filter.
	// Maximum legitimacy assurance for critical deployments.
	PRISMModeHybrid = "hybrid"
)

// PRISM refusal policies control behavior when a model fails legitimacy verification.
const (
	// PRISMRefusalReroute tries the next best candidate on refusal (default).
	PRISMRefusalReroute = "reroute"

	// PRISMRefusalReject returns a refusal response immediately.
	PRISMRefusalReject = "reject"
)

// PRISMConfig configures the PRISM 153-key legitimacy verification layer.
// PRISM (Protocol for Routed Intelligent Specialized Models) ensures that each model
// explicitly declares its domain boundaries and formally refuses out-of-scope queries.
type PRISMConfig struct {
	// Enabled controls whether PRISM verification is active
	Enabled bool `yaml:"enabled"`

	// Mode controls where PRISM verification is applied:
	// "fine_filter" (Option 1): After model selection — validates the selected model
	// "coarse_filter" (Option 2): Before model selection — pre-qualifies candidates
	// "hybrid" (Option 3): Both before and after model selection
	Mode string `yaml:"mode"`

	// DomainThreshold is the minimum similarity score required for a model
	// to be considered legitimate for a query (0.0–1.0, default: 0.3)
	DomainThreshold float64 `yaml:"domain_threshold"`

	// RefusalPolicy controls behavior when a model is refused:
	// "reroute" (default): Try the next best candidate
	// "reject": Return a refusal response
	RefusalPolicy string `yaml:"refusal_policy"`

	// MaxRerouteAttempts limits rerouting attempts when refusal_policy is "reroute" (default: 3)
	MaxRerouteAttempts int `yaml:"max_reroute_attempts"`
}

// DefaultPRISMConfig returns the default PRISM configuration
func DefaultPRISMConfig() *PRISMConfig {
	return &PRISMConfig{
		Enabled:            false,
		Mode:               PRISMModeFineFilter,
		DomainThreshold:    0.3,
		RefusalPolicy:      PRISMRefusalReroute,
		MaxRerouteAttempts: 3,
	}
}

// PRISMLegitimacyResult represents the outcome of a PRISM 153-key verification
type PRISMLegitimacyResult struct {
	// Legitimate indicates whether the model passed verification
	Legitimate bool

	// Score is the domain alignment score (0.0–1.0)
	Score float64

	// Reason provides a human-readable explanation
	Reason string

	// ModelName is the model that was verified
	ModelName string
}

// PRISMValidator implements the PRISM 153-key legitimacy verification protocol.
// It ensures that each model operates within its declared domain boundaries
// by checking query–model alignment using keyword matching and embedding similarity.
type PRISMValidator struct {
	config *PRISMConfig

	// modelDomains maps model names to their declared domain keywords
	modelDomains map[string][]string

	// modelDescriptions maps model names to their description text
	modelDescriptions map[string]string

	// embeddingFunc computes embeddings for similarity comparison
	embeddingFunc func(string) ([]float32, error)

	mu sync.RWMutex
}

// NewPRISMValidator creates a new PRISM legitimacy validator
func NewPRISMValidator(cfg *PRISMConfig) *PRISMValidator {
	if cfg == nil {
		cfg = DefaultPRISMConfig()
	}
	return &PRISMValidator{
		config:            cfg,
		modelDomains:      make(map[string][]string),
		modelDescriptions: make(map[string]string),
	}
}

// SetEmbeddingFunc sets the embedding function for similarity-based verification
func (v *PRISMValidator) SetEmbeddingFunc(fn func(string) ([]float32, error)) {
	v.mu.Lock()
	defer v.mu.Unlock()
	v.embeddingFunc = fn
}

// InitializeFromConfig populates model domain boundaries from model configuration.
// Each model's Description and Capabilities fields define its declared domain.
func (v *PRISMValidator) InitializeFromConfig(modelConfig map[string]config.ModelParams) {
	v.mu.Lock()
	defer v.mu.Unlock()

	for name, params := range modelConfig {
		if params.Description != "" {
			v.modelDescriptions[name] = params.Description
			// Extract domain keywords from description
			v.modelDomains[name] = extractDomainKeywords(params.Description)
		}
		if len(params.Capabilities) > 0 {
			// Capabilities directly serve as domain boundary declarations
			existing := v.modelDomains[name]
			for _, cap := range params.Capabilities {
				existing = append(existing, strings.ToLower(cap))
			}
			v.modelDomains[name] = existing
		}
	}

	logging.Infof("[PRISM] Initialized domain boundaries for %d models", len(v.modelDomains))
}

// Validate performs the PRISM 153-key legitimacy check for a model against a query.
// Returns a legitimacy result indicating whether the model is authorized to respond.
func (v *PRISMValidator) Validate(ctx context.Context, modelName string, query string) *PRISMLegitimacyResult {
	v.mu.RLock()
	defer v.mu.RUnlock()

	// If no domain boundaries declared, model is considered legitimate by default
	domains, hasDomains := v.modelDomains[modelName]
	description := v.modelDescriptions[modelName]

	if !hasDomains && description == "" {
		return &PRISMLegitimacyResult{
			Legitimate: true,
			Score:      1.0,
			Reason:     "No domain boundaries declared — model accepted by default",
			ModelName:  modelName,
		}
	}

	queryLower := strings.ToLower(query)
	var bestScore float64

	// Phase 1: Keyword-based domain matching
	if len(domains) > 0 {
		keywordScore := v.computeKeywordScore(queryLower, domains)
		if keywordScore > bestScore {
			bestScore = keywordScore
		}
	}

	// Phase 2: Embedding-based similarity (if available)
	if v.embeddingFunc != nil && description != "" {
		embeddingScore := v.computeEmbeddingScore(query, description)
		if embeddingScore > bestScore {
			bestScore = embeddingScore
		}
	}

	legitimate := bestScore >= v.config.DomainThreshold

	var reason string
	if legitimate {
		reason = fmt.Sprintf("PRISM 153-key: ACCEPTED — domain alignment score %.3f >= threshold %.3f",
			bestScore, v.config.DomainThreshold)
	} else {
		reason = fmt.Sprintf("PRISM 153-key: REFUSED — domain alignment score %.3f < threshold %.3f (model %s is not legitimate for this query)",
			bestScore, v.config.DomainThreshold, modelName)
	}

	return &PRISMLegitimacyResult{
		Legitimate: legitimate,
		Score:      bestScore,
		Reason:     reason,
		ModelName:  modelName,
	}
}

// FilterCandidates applies coarse PRISM filtering to remove illegitimate models
// from the candidate pool before selection. Returns the filtered list.
func (v *PRISMValidator) FilterCandidates(ctx context.Context, candidates []config.ModelRef, query string) []config.ModelRef {
	if !v.config.Enabled {
		return candidates
	}

	var legitimate []config.ModelRef
	for _, candidate := range candidates {
		result := v.Validate(ctx, candidate.Model, query)
		if result.Legitimate {
			legitimate = append(legitimate, candidate)
			logging.Debugf("[PRISM] Coarse filter: %s ACCEPTED (score=%.3f)", candidate.Model, result.Score)
		} else {
			logging.Infof("[PRISM] Coarse filter: %s REFUSED — %s", candidate.Model, result.Reason)
		}
	}

	// If all models were filtered out, return the original list to avoid deadlock
	if len(legitimate) == 0 {
		logging.Warnf("[PRISM] All %d candidates were refused — bypassing coarse filter", len(candidates))
		return candidates
	}

	return legitimate
}

// ValidateSelection applies fine PRISM filtering after model selection.
// If the selected model is refused and rerouting is enabled, it tries alternatives.
func (v *PRISMValidator) ValidateSelection(ctx context.Context, selected *SelectionResult, candidates []config.ModelRef, query string, selector Selector, selCtx *SelectionContext) (*SelectionResult, *PRISMLegitimacyResult) {
	if !v.config.Enabled || selected == nil {
		return selected, &PRISMLegitimacyResult{Legitimate: true, Score: 1.0, ModelName: ""}
	}

	result := v.Validate(ctx, selected.SelectedModel, query)
	if result.Legitimate {
		logging.Debugf("[PRISM] Fine filter: %s ACCEPTED (score=%.3f)", selected.SelectedModel, result.Score)
		return selected, result
	}

	logging.Infof("[PRISM] Fine filter: %s REFUSED — %s", selected.SelectedModel, result.Reason)

	// If reject policy, return the refusal immediately
	if v.config.RefusalPolicy == PRISMRefusalReject {
		return nil, result
	}

	// Reroute: try other candidates
	tried := map[string]bool{selected.SelectedModel: true}
	for attempt := 0; attempt < v.config.MaxRerouteAttempts; attempt++ {
		// Build reduced candidate list excluding tried models
		var remaining []config.ModelRef
		for _, c := range candidates {
			if !tried[c.Model] {
				remaining = append(remaining, c)
			}
		}
		if len(remaining) == 0 {
			break
		}

		// Re-select from remaining candidates
		rerouteCtx := &SelectionContext{
			Query:           selCtx.Query,
			DecisionName:    selCtx.DecisionName,
			CategoryName:    selCtx.CategoryName,
			CandidateModels: remaining,
			CostWeight:      selCtx.CostWeight,
			QualityWeight:   selCtx.QualityWeight,
		}
		newResult, err := selector.Select(ctx, rerouteCtx)
		if err != nil {
			logging.Warnf("[PRISM] Reroute attempt %d failed: %v", attempt+1, err)
			break
		}

		tried[newResult.SelectedModel] = true
		recheck := v.Validate(ctx, newResult.SelectedModel, query)
		if recheck.Legitimate {
			logging.Infof("[PRISM] Reroute succeeded: %s ACCEPTED on attempt %d (score=%.3f)",
				newResult.SelectedModel, attempt+1, recheck.Score)
			newResult.Reasoning = fmt.Sprintf("%s (PRISM rerouted from %s)", newResult.Reasoning, selected.SelectedModel)
			return newResult, recheck
		}

		logging.Infof("[PRISM] Reroute attempt %d: %s also REFUSED", attempt+1, newResult.SelectedModel)
	}

	// All reroute attempts failed — fall back to original selection
	logging.Warnf("[PRISM] All reroute attempts exhausted — accepting original selection %s", selected.SelectedModel)
	return selected, result
}

// computeKeywordScore calculates domain alignment using keyword overlap
func (v *PRISMValidator) computeKeywordScore(queryLower string, domains []string) float64 {
	if len(domains) == 0 {
		return 0.0
	}

	matches := 0
	for _, keyword := range domains {
		if strings.Contains(queryLower, keyword) {
			matches++
		}
	}

	return float64(matches) / float64(len(domains))
}

// computeEmbeddingScore calculates domain alignment using cosine similarity
func (v *PRISMValidator) computeEmbeddingScore(query, description string) float64 {
	queryEmb, err := v.embeddingFunc(query)
	if err != nil {
		logging.Warnf("[PRISM] Failed to embed query: %v", err)
		return 0.0
	}

	descEmb, err := v.embeddingFunc(description)
	if err != nil {
		logging.Warnf("[PRISM] Failed to embed description: %v", err)
		return 0.0
	}

	return cosineSimilarity(queryEmb, descEmb)
}

// cosineSimilarity computes the cosine similarity between two vectors
func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range a {
		dotProduct += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	denominator := math.Sqrt(normA) * math.Sqrt(normB)
	if denominator == 0 {
		return 0.0
	}

	return dotProduct / denominator
}

// extractDomainKeywords extracts meaningful keywords from a model description.
// These keywords form the model's declared domain boundaries per the PRISM 153-key protocol.
func extractDomainKeywords(description string) []string {
	// Split description into words and filter common stop words
	words := strings.Fields(strings.ToLower(description))
	stopWords := map[string]bool{
		"a": true, "an": true, "the": true, "is": true, "are": true,
		"was": true, "were": true, "be": true, "been": true, "being": true,
		"have": true, "has": true, "had": true, "do": true, "does": true,
		"did": true, "will": true, "would": true, "could": true, "should": true,
		"may": true, "might": true, "shall": true, "can": true,
		"and": true, "or": true, "but": true, "if": true, "then": true,
		"for": true, "of": true, "with": true, "at": true, "by": true,
		"from": true, "in": true, "on": true, "to": true, "into": true,
		"that": true, "this": true, "it": true, "its": true,
		"not": true, "no": true, "nor": true,
		"model": true, "language": true, "large": true,
	}

	var keywords []string
	seen := make(map[string]bool)
	for _, word := range words {
		// Clean punctuation
		word = strings.Trim(word, ".,;:!?()[]{}\"'")
		if len(word) < 3 || stopWords[word] || seen[word] {
			continue
		}
		seen[word] = true
		keywords = append(keywords, word)
	}

	return keywords
}
