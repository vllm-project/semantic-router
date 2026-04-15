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
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection/lookuptable"
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

	// ML configuration (used for knn, kmeans, svm methods)
	ML *MLSelectorConfig `yaml:"ml,omitempty"`

	// RLDriven configuration (used when method is "rl_driven")
	// Implements Router-R1 reward structure for RL training
	RLDriven *RLDrivenConfig `yaml:"rl_driven,omitempty"`

	// GMTRouter configuration (used when method is "gmtrouter")
	// Implements heterogeneous graph learning for personalized routing
	GMTRouter *GMTRouterConfig `yaml:"gmtrouter,omitempty"`
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
	lookupTable   lookuptable.LookupTable
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

// WithLookupTable sets the lookup table used by selectors that support data-driven
// constant resolution (e.g. HybridSelector's quality-gap threshold).
func (f *Factory) WithLookupTable(lt lookuptable.LookupTable) *Factory {
	f.lookupTable = lt
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
		if f.lookupTable != nil {
			hybridSelector.SetLookupTable(f.lookupTable)
		}
		selector = hybridSelector

	case MethodGMTRouter:
		gmtRouterSelector := NewGMTRouterSelector(f.cfg.GMTRouter)
		if f.modelConfig != nil {
			gmtRouterSelector.InitializeFromConfig(f.modelConfig)
		}
		if f.embeddingFunc != nil {
			gmtRouterSelector.SetEmbeddingFunc(f.embeddingFunc)
		}
		selector = gmtRouterSelector

	case MethodRLDriven:
		rlDrivenSelector := NewRLDrivenSelector(f.cfg.RLDriven)
		if f.modelConfig != nil {
			rlDrivenSelector.InitializeFromConfig(f.modelConfig, f.categories)
		}
		selector = rlDrivenSelector

	case MethodLatencyAware:
		selector = NewLatencyAwareSelector(nil)

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
	if f.lookupTable != nil {
		hybridSelector.SetLookupTable(f.lookupTable)
	}
	registry.Register(MethodHybrid, hybridSelector)

	// Create ML-based selectors (KNN, KMeans, SVM)
	mlCfg := f.cfg.ML
	if mlCfg == nil {
		mlCfg = DefaultMLSelectorConfig()
	}

	// Create KNN selector
	knnAdapter, err := CreateKNNSelector(mlCfg, f.embeddingFunc)
	if err != nil {
		logging.Warnf("[SelectionFactory] Failed to create KNN selector: %v", err)
	} else {
		registry.Register(MethodKNN, knnAdapter)
	}

	// Create KMeans selector
	kmeansAdapter, err := CreateKMeansSelector(mlCfg, f.embeddingFunc)
	if err != nil {
		logging.Warnf("[SelectionFactory] Failed to create KMeans selector: %v", err)
	} else {
		registry.Register(MethodKMeans, kmeansAdapter)
	}

	// Create SVM selector
	svmAdapter, err := CreateSVMSelector(mlCfg, f.embeddingFunc)
	if err != nil {
		logging.Warnf("[SelectionFactory] Failed to create SVM selector: %v", err)
	} else {
		registry.Register(MethodSVM, svmAdapter)
	}

	// Create MLP selector (GPU-accelerated via Candle)
	mlpAdapter, err := CreateMLPSelector(mlCfg, f.embeddingFunc)
	if err != nil {
		logging.Warnf("[SelectionFactory] Failed to create MLP selector: %v", err)
	} else {
		registry.Register(MethodMLP, mlpAdapter)
	}

	// Create RL-Driven selector
	rlDrivenCfg := f.cfg.RLDriven
	if rlDrivenCfg == nil {
		rlDrivenCfg = DefaultRLDrivenConfig()
	}
	rlDrivenSelector := NewRLDrivenSelector(rlDrivenCfg)
	if f.modelConfig != nil {
		rlDrivenSelector.InitializeFromConfig(f.modelConfig, f.categories)
	}
	registry.Register(MethodRLDriven, rlDrivenSelector)

	// Create GMTRouter selector
	gmtRouterCfg := f.cfg.GMTRouter
	if gmtRouterCfg == nil {
		gmtRouterCfg = DefaultGMTRouterConfig()
	}
	gmtRouterSelector := NewGMTRouterSelector(gmtRouterCfg)
	if f.modelConfig != nil {
		gmtRouterSelector.InitializeFromConfig(f.modelConfig)
	}
	if f.embeddingFunc != nil {
		gmtRouterSelector.SetEmbeddingFunc(f.embeddingFunc)
	}
	registry.Register(MethodGMTRouter, gmtRouterSelector)

	// Create LatencyAware selector
	latencyAwareSelector := NewLatencyAwareSelector(nil)
	registry.Register(MethodLatencyAware, latencyAwareSelector)

	LogRegisteredAlgorithms(registry)
	logging.ComponentEvent("selection", "selection_factory_initialized", map[string]interface{}{
		"selector_count": len(registry.selectors),
	})
	return registry
}

// LogRegisteredAlgorithms logs the tier and dependencies of each registered algorithm
func LogRegisteredAlgorithms(registry *Registry) {
	registry.mu.RLock()
	defer registry.mu.RUnlock()

	for method, selector := range registry.selectors {
		deps := selector.ExternalDependencies()
		if len(deps) == 0 {
			logging.Infof("[Selection] Registered algorithm: %s (tier=%s, dependencies=none)", method, selector.Tier())
		} else {
			depNames := make([]string, len(deps))
			for i, dep := range deps {
				depNames[i] = fmt.Sprintf("%s (%s)", dep.Name, dep.Type)
			}
			logging.Infof("[Selection] Registered algorithm: %s (tier=%s, dependencies=[%s])",
				method, selector.Tier(), strings.Join(depNames, ", "))
		}
	}
}

// WarnExperimentalAlgorithms logs prominent warnings for experimental algorithms
// that are actually configured in operator decisions
func WarnExperimentalAlgorithms(registry *Registry, configuredMethods []SelectionMethod) {
	for _, method := range configuredMethods {
		selector, ok := registry.Get(method)
		if !ok {
			continue
		}
		if selector.Tier() != TierExperimental {
			continue
		}

		deps := selector.ExternalDependencies()
		logging.Warnf("[Selection] WARNING: Algorithm %q is EXPERIMENTAL and not recommended for production use", method)
		for _, dep := range deps {
			if dep.HealthURL != "" {
				logging.Warnf("[Selection]   External dependency: %s (%s)", dep.Name, dep.HealthURL)
			} else {
				logging.Warnf("[Selection]   Dependency: %s — %s", dep.Name, dep.Description)
			}
		}
	}
}

// CheckDependencyHealth checks reachability of external service dependencies
// for the given algorithms. Logs results but never fails.
func CheckDependencyHealth(registry *Registry, configuredMethods []SelectionMethod) {
	for _, method := range configuredMethods {
		selector, ok := registry.Get(method)
		if !ok {
			continue
		}

		for _, dep := range selector.ExternalDependencies() {
			if dep.Type != DependencyExternalService || dep.HealthURL == "" {
				continue
			}

			ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			client := &http.Client{Timeout: 5 * time.Second}
			req, err := http.NewRequestWithContext(ctx, "GET", dep.HealthURL, nil)
			if err != nil {
				logging.Warnf("[Selection] Dependency check: %s — UNREACHABLE (bad URL: %v)", dep.Name, err)
				cancel()
				continue
			}

			resp, err := client.Do(req)
			cancel()
			if err != nil {
				logging.Warnf("[Selection] Dependency check: %s at %s — UNREACHABLE (will degrade at runtime)", dep.Name, dep.HealthURL)
			} else {
				_ = resp.Body.Close()
				if resp.StatusCode == http.StatusOK {
					logging.Infof("[Selection] Dependency check: %s at %s — OK", dep.Name, dep.HealthURL)
				} else {
					logging.Warnf("[Selection] Dependency check: %s at %s — unhealthy (status %d)", dep.Name, dep.HealthURL, resp.StatusCode)
				}
			}
		}
	}
}

// Initialize sets up the global registry with all selectors
func Initialize(cfg *ModelSelectionConfig, modelConfig map[string]config.ModelParams, categories []config.Category, embeddingFunc func(string) ([]float32, error)) {
	factory := NewFactory(cfg).
		WithModelConfig(modelConfig).
		WithCategories(categories).
		WithEmbeddingFunc(embeddingFunc)

	// Create all selectors and register globally
	GlobalRegistry = factory.CreateAll()

	logging.ComponentEvent("selection", "selection_registry_initialized", map[string]interface{}{
		"selector_count": len(GlobalRegistry.selectors),
	})
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
