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
)

// Factory creates the shared selectors of a routing recipe: static, random
// and the default multi_factor. A decision with its own multi_factor block,
// and a systemone selector, are built for that decision.
type Factory struct {
	modelConfig map[string]config.ModelParams
	categories  []config.Category
}

// NewFactory creates a new selector factory
func NewFactory() *Factory {
	return &Factory{}
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

func (f *Factory) staticSelector() *StaticSelector {
	selector := NewStaticSelector(DefaultStaticConfig())
	if f.categories != nil {
		selector.InitializeFromConfig(f.categories)
	}
	return selector
}

func (f *Factory) multiFactorSelector() *MultiFactorSelector {
	selector := NewMultiFactorSelector(nil)
	if f.modelConfig != nil {
		selector.InitializeFromConfig(f.modelConfig)
	}
	return selector
}

// CreateAll creates all available selectors and registers them
func (f *Factory) CreateAll() *Registry {
	// Initialize metrics for model selection tracking
	InitializeMetrics()

	registry := NewRegistry()
	registry.Register(MethodStatic, f.staticSelector())
	registry.Register(MethodRandom, NewRandomSelector())
	registry.Register(MethodMultiFactor, f.multiFactorSelector())

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

// GetSelector returns a selector for the specified method from global registry
func GetSelector(method SelectionMethod) Selector {
	registry := GetGlobalRegistry()
	selector, ok := registry.Get(method)
	if !ok {
		// Fallback to static
		selector, _ = registry.Get(MethodStatic)
	}
	return selector
}
