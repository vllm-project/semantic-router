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
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// ModelSelectionConfig represents the configuration for model selection
type ModelSelectionConfig struct {
	// Method specifies the selection algorithm to use
	Method string `yaml:"method"`

	// Elo configuration (used when method is "elo")
	Elo *EloConfig `yaml:"elo,omitempty"`

	// RouterDC configuration (used when method is "router_dc")
	RouterDC *RouterDCConfig `yaml:"router_dc,omitempty"`

	// AutoMix configuration (used when method is "automix")
	AutoMix *AutoMixConfig `yaml:"automix,omitempty"`

	// Hybrid configuration (used when method is "hybrid")
	Hybrid *HybridConfig `yaml:"hybrid,omitempty"`
}

// DefaultModelSelectionConfig returns the default configuration
func DefaultModelSelectionConfig() *ModelSelectionConfig {
	return &ModelSelectionConfig{
		Method: string(MethodStatic),
	}
}

// Factory creates and initializes selectors based on configuration
type Factory struct {
	cfg           *ModelSelectionConfig
	modelConfig   map[string]config.ModelParams
	categories    []config.Category
	embeddingFunc func(string) ([]float32, error)
}

// NewFactory creates a new selector factory
func NewFactory(cfg *ModelSelectionConfig) *Factory {
	if cfg == nil {
		cfg = DefaultModelSelectionConfig()
	}
	return &Factory{
		cfg: cfg,
	}
}

// WithModelConfig sets the model configuration
func (f *Factory) WithModelConfig(modelConfig map[string]config.ModelParams) *Factory {
	f.modelConfig = modelConfig
	return f
}

// WithCategories sets the category configuration
func (f *Factory) WithCategories(categories []config.Category) *Factory {
	f.categories = categories
	return f
}

// WithEmbeddingFunc sets the embedding function for RouterDC
func (f *Factory) WithEmbeddingFunc(fn func(string) ([]float32, error)) *Factory {
	f.embeddingFunc = fn
	return f
}

// Create creates and initializes a selector based on the configured method
func (f *Factory) Create() Selector {
	method := SelectionMethod(f.cfg.Method)

	var selector Selector

	switch method {
	case MethodElo:
		eloSelector := NewEloSelector(f.cfg.Elo)
		if f.modelConfig != nil {
			eloSelector.InitializeFromConfig(f.modelConfig, f.categories)
		}
		selector = eloSelector

	case MethodRouterDC:
		routerDCSelector := NewRouterDCSelector(f.cfg.RouterDC)
		if f.embeddingFunc != nil {
			routerDCSelector.SetEmbeddingFunc(f.embeddingFunc)
		}
		// Initialize model embeddings from descriptions in model config
		if f.modelConfig != nil {
			if err := routerDCSelector.InitializeFromConfig(f.modelConfig); err != nil {
				logging.Errorf("[SelectionFactory] RouterDC initialization failed: %v", err)
			}
		}
		selector = routerDCSelector

	case MethodAutoMix:
		autoMixSelector := NewAutoMixSelector(f.cfg.AutoMix)
		if f.modelConfig != nil {
			autoMixSelector.InitializeFromConfig(f.modelConfig)
		}
		selector = autoMixSelector

	case MethodHybrid:
		hybridSelector := NewHybridSelector(f.cfg.Hybrid)
		if f.modelConfig != nil {
			hybridSelector.InitializeFromConfig(f.modelConfig, f.categories)
		}
		if f.embeddingFunc != nil && hybridSelector.routerDCSelector != nil {
			hybridSelector.routerDCSelector.SetEmbeddingFunc(f.embeddingFunc)
		}
		selector = hybridSelector

	default:
		// Default to static selector
		staticSelector := NewStaticSelector(DefaultStaticConfig())
		if f.categories != nil {
			staticSelector.InitializeFromConfig(f.categories)
		}
		selector = staticSelector
	}

	logging.Infof("[SelectionFactory] Created selector: method=%s", method)
	return selector
}

// CreateAll creates all available selectors and registers them
func (f *Factory) CreateAll() *Registry {
	// Initialize metrics for model selection tracking
	InitializeMetrics()

	registry := NewRegistry()

	// Always create static selector
	staticSelector := NewStaticSelector(DefaultStaticConfig())
	if f.categories != nil {
		staticSelector.InitializeFromConfig(f.categories)
	}
	registry.Register(MethodStatic, staticSelector)

	// Create Elo selector
	eloCfg := f.cfg.Elo
	if eloCfg == nil {
		eloCfg = DefaultEloConfig()
	}
	eloSelector := NewEloSelector(eloCfg)
	if f.modelConfig != nil {
		eloSelector.InitializeFromConfig(f.modelConfig, f.categories)
	}
	registry.Register(MethodElo, eloSelector)

	// Create RouterDC selector
	routerDCCfg := f.cfg.RouterDC
	if routerDCCfg == nil {
		routerDCCfg = DefaultRouterDCConfig()
	}
	routerDCSelector := NewRouterDCSelector(routerDCCfg)
	if f.embeddingFunc != nil {
		routerDCSelector.SetEmbeddingFunc(f.embeddingFunc)
	}
	// Initialize model embeddings from descriptions in model config
	if f.modelConfig != nil {
		if err := routerDCSelector.InitializeFromConfig(f.modelConfig); err != nil {
			logging.Errorf("[SelectionFactory] RouterDC initialization failed: %v", err)
		}
	}
	registry.Register(MethodRouterDC, routerDCSelector)

	// Create AutoMix selector
	autoMixCfg := f.cfg.AutoMix
	if autoMixCfg == nil {
		autoMixCfg = DefaultAutoMixConfig()
	}
	autoMixSelector := NewAutoMixSelector(autoMixCfg)
	if f.modelConfig != nil {
		autoMixSelector.InitializeFromConfig(f.modelConfig)
	}
	registry.Register(MethodAutoMix, autoMixSelector)

	// Create Hybrid selector with component references
	hybridCfg := f.cfg.Hybrid
	if hybridCfg == nil {
		hybridCfg = DefaultHybridConfig()
	}
	hybridSelector := NewHybridSelectorWithComponents(hybridCfg, eloSelector, routerDCSelector, autoMixSelector)
	if f.modelConfig != nil {
		hybridSelector.InitializeFromConfig(f.modelConfig, f.categories)
	}
	registry.Register(MethodHybrid, hybridSelector)

	logging.Infof("[SelectionFactory] Created all selectors: static, elo, router_dc, automix, hybrid")
	return registry
}

// Initialize sets up the global registry with all selectors
func Initialize(cfg *ModelSelectionConfig, modelConfig map[string]config.ModelParams, categories []config.Category, embeddingFunc func(string) ([]float32, error)) {
	factory := NewFactory(cfg).
		WithModelConfig(modelConfig).
		WithCategories(categories).
		WithEmbeddingFunc(embeddingFunc)

	// Create all selectors and register globally
	GlobalRegistry = factory.CreateAll()

	logging.Infof("[Selection] Initialized global selector registry")
}

// GetSelector returns a selector for the specified method from global registry
func GetSelector(method SelectionMethod) Selector {
	selector, ok := GlobalRegistry.Get(method)
	if !ok {
		// Fallback to static
		selector, _ = GlobalRegistry.Get(MethodStatic)
	}
	return selector
}
