package promptcompression

import (
	"fmt"
	"path/filepath"
	"sync"
)

// PipelineSelector maps decision/domain names to compression Pipelines.
//
// This is used exclusively for Use Case 2 (pre-inference compression):
// by the time the selector is called, the domain classifier has already
// run and the winning decision name is known, so there is no chicken-and-egg
// problem. The selector simply looks up the pre-loaded Pipeline for that
// domain and returns it (falling back to a default if no specific pipeline
// is configured for that decision).
//
//	Use Case 1 — pre-classification  (compression before signal evaluation)
//	  Domain is NOT known yet.
//	  Use the global PromptCompressionConfig with a single default pipeline.
//	  See: req_filter_classification.go → performDecisionEvaluation.
//
//	Use Case 2 — pre-inference  (compression before sending to vLLM)
//	  Domain IS known (from the DecisionResult already computed in UC1).
//	  Use PipelineSelector.Select(decisionName) to get the domain pipeline.
//	  See: processor_req_body_routing.go → modifyRequestBodyForAutoRouting.
//
// Pipelines are loaded once at startup (from YAML files) and cached in the
// selector. The selector itself is immutable after construction.
type PipelineSelector struct {
	mu         sync.RWMutex
	byDecision map[string]Pipeline
	def        Pipeline
}

// Select returns the Pipeline registered for decisionName.
// Falls back to the default Pipeline if no specific mapping exists.
// Safe for concurrent use.
func (s *PipelineSelector) Select(decisionName string) Pipeline {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if p, ok := s.byDecision[decisionName]; ok {
		return p
	}
	return s.def
}

// Register adds or replaces the Pipeline for a decision name.
// Typically called at startup; safe for concurrent use.
func (s *PipelineSelector) Register(decisionName string, p Pipeline) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.byDecision[decisionName] = p
}

// SelectorConfig is the YAML schema for building a PipelineSelector.
//
// In the router config YAML this appears under prompt_compression:
//
//	prompt_compression:
//	  # Use Case 1 (existing)
//	  enabled: true
//	  max_tokens: 512
//
//	  # Use Case 2 (new)
//	  inference_enabled: true
//	  inference_max_tokens: 4096   # per-pipeline overrides this
//	  default_pipeline: ./compression/config_default.yaml
//	  domain_pipelines:
//	    coding:   ./compression/config_coding.yaml
//	    medical:  ./compression/config_medical.yaml
//	    security: ./compression/config_security.yaml
type SelectorConfig struct {
	// DefaultPipeline is the path to the YAML pipeline config used when no
	// domain-specific pipeline matches the decision name.
	DefaultPipeline string `yaml:"default_pipeline"`

	// DomainPipelines maps decision/domain names (as returned by the
	// DecisionResult) to YAML pipeline config paths.
	DomainPipelines map[string]string `yaml:"domain_pipelines"`

	// BaseDir is prepended to relative paths when resolving YAML files.
	// Typically the directory of the router config file. Not a YAML field;
	// set programmatically after unmarshalling.
	BaseDir string `yaml:"-"`
}

// NewPipelineSelector builds a PipelineSelector from a SelectorConfig,
// loading all referenced YAML pipeline files.
func NewPipelineSelector(cfg SelectorConfig) (*PipelineSelector, error) {
	s := &PipelineSelector{
		byDecision: make(map[string]Pipeline, len(cfg.DomainPipelines)),
	}

	// Load the default pipeline.
	if cfg.DefaultPipeline == "" {
		// No default file: use the built-in DefaultPipeline equivalent.
		// max_tokens=0 means "pass-through" — callers should override.
		s.def = DefaultPipeline(0)
	} else {
		p, err := LoadPipeline(resolvePath(cfg.BaseDir, cfg.DefaultPipeline))
		if err != nil {
			return nil, fmt.Errorf("promptcompression: loading default pipeline %q: %w",
				cfg.DefaultPipeline, err)
		}
		s.def = p
	}

	// Load domain-specific pipelines.
	for domain, path := range cfg.DomainPipelines {
		p, err := LoadPipeline(resolvePath(cfg.BaseDir, path))
		if err != nil {
			return nil, fmt.Errorf("promptcompression: loading pipeline for domain %q (%s): %w",
				domain, path, err)
		}
		s.byDecision[domain] = p
	}

	return s, nil
}

// NewPipelineSelectorFromMap builds a PipelineSelector directly from
// pre-loaded Pipelines. Useful in tests or when the caller already holds
// parsed Pipeline objects.
func NewPipelineSelectorFromMap(def Pipeline, byDecision map[string]Pipeline) *PipelineSelector {
	if byDecision == nil {
		byDecision = map[string]Pipeline{}
	}
	return &PipelineSelector{
		byDecision: byDecision,
		def:        def,
	}
}

func resolvePath(baseDir, path string) string {
	if filepath.IsAbs(path) || baseDir == "" {
		return path
	}
	return filepath.Join(baseDir, path)
}
