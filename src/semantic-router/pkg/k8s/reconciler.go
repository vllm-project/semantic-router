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

package k8s

import (
	"context"
	"fmt"
	"sync"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	"sigs.k8s.io/controller-runtime/pkg/metrics/server"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/apis/vllm.ai/v1alpha1"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Reconciler reconciles IntelligentPool and IntelligentRoute CRDs
type Reconciler struct {
	client         client.Client
	scheme         *runtime.Scheme
	namespace      string
	converter      *CRDConverter
	staticConfig   *config.RouterConfig
	onConfigUpdate func(*config.RouterConfig) error
	mu             sync.RWMutex
	lastPool       *v1alpha1.IntelligentPool
	lastRoute      *v1alpha1.IntelligentRoute
}

// ReconcilerConfig holds configuration for the reconciler
type ReconcilerConfig struct {
	Namespace      string
	Kubeconfig     string // Optional: if empty, uses in-cluster config
	StaticConfig   *config.RouterConfig
	OnConfigUpdate func(*config.RouterConfig) error
}

// NewReconciler creates a new reconciler with controller-runtime
func NewReconciler(cfg ReconcilerConfig) (*Reconciler, error) {
	// Build REST config
	var restConfig *rest.Config
	var err error
	if cfg.Kubeconfig != "" {
		restConfig, err = clientcmd.BuildConfigFromFlags("", cfg.Kubeconfig)
	} else {
		restConfig, err = rest.InClusterConfig()
	}
	if err != nil {
		return nil, fmt.Errorf("failed to build REST config: %w", err)
	}

	// Create scheme and register our types
	scheme := runtime.NewScheme()
	err = v1alpha1.AddToScheme(scheme)
	if err != nil {
		return nil, fmt.Errorf("failed to add v1alpha1 to scheme: %w", err)
	}

	// Create manager options
	options := manager.Options{
		Scheme: scheme,
		Cache: cache.Options{
			DefaultNamespaces: map[string]cache.Config{
				cfg.Namespace: {},
			},
		},
		// Disable metrics server to avoid port conflicts
		Metrics: server.Options{
			BindAddress: "0", // "0" disables the metrics server
		},
	}

	// Create manager
	mgr, err := manager.New(restConfig, options)
	if err != nil {
		return nil, fmt.Errorf("failed to create manager: %w", err)
	}

	reconciler := &Reconciler{
		client:         mgr.GetClient(),
		scheme:         scheme,
		namespace:      cfg.Namespace,
		converter:      NewCRDConverter(),
		staticConfig:   cfg.StaticConfig,
		onConfigUpdate: cfg.OnConfigUpdate,
	}

	// Start the manager in a goroutine
	go func() {
		if err := mgr.Start(context.Background()); err != nil {
			logging.Errorf("Failed to start manager: %v", err)
		}
	}()

	// Wait for cache to sync
	if !mgr.GetCache().WaitForCacheSync(context.Background()) {
		return nil, fmt.Errorf("failed to wait for cache sync")
	}

	return reconciler, nil
}

// Start starts watching for CRD changes
func (r *Reconciler) Start(ctx context.Context) error {
	logging.Infof("Starting Kubernetes reconciler in namespace %s", r.namespace)

	// Initial sync
	if err := r.reconcile(ctx); err != nil {
		logging.Warnf("Initial reconciliation failed (will retry on CRD changes): %v", err)
	}

	// Start watch loops
	go r.watchLoop(ctx)

	return nil
}

// Stop stops the reconciler
func (r *Reconciler) Stop() {
	logging.Infof("Stopping Kubernetes reconciler")
}

// watchLoop continuously watches for CRD changes using informers
func (r *Reconciler) watchLoop(ctx context.Context) {
	// Use a ticker to periodically check for changes
	// The cache will automatically update via informers
	ticker := time.NewTicker(5 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			if err := r.reconcile(ctx); err != nil {
				logging.Debugf("Reconciliation check: %v", err)
			}
		}
	}
}

// reconcile performs the reconciliation logic
func (r *Reconciler) reconcile(ctx context.Context) error {
	// Get IntelligentPool
	pool, err := r.getIntelligentPool(ctx)
	if err != nil {
		return fmt.Errorf("failed to get IntelligentPool: %w", err)
	}

	// Get IntelligentRoute
	route, err := r.getIntelligentRoute(ctx)
	if err != nil {
		return fmt.Errorf("failed to get IntelligentRoute: %w", err)
	}

	// Check if anything changed
	r.mu.RLock()
	poolChanged := r.lastPool == nil || pool.Generation != r.lastPool.Generation
	routeChanged := r.lastRoute == nil || route.Generation != r.lastRoute.Generation
	r.mu.RUnlock()

	if !poolChanged && !routeChanged {
		return nil // No changes
	}

	logging.Infof("CRD changes detected, reconciling configuration")

	// Validate and update
	if err := r.validateAndUpdate(ctx, pool, route); err != nil {
		return fmt.Errorf("validation/update failed: %w", err)
	}

	// Update last seen versions
	r.mu.Lock()
	r.lastPool = pool.DeepCopy()
	r.lastRoute = route.DeepCopy()
	r.mu.Unlock()

	return nil
}

// getIntelligentPool retrieves the IntelligentPool from the namespace
func (r *Reconciler) getIntelligentPool(ctx context.Context) (*v1alpha1.IntelligentPool, error) {
	poolList := &v1alpha1.IntelligentPoolList{}
	if err := r.client.List(ctx, poolList, client.InNamespace(r.namespace)); err != nil {
		return nil, fmt.Errorf("failed to list IntelligentPools: %w", err)
	}

	if len(poolList.Items) == 0 {
		return nil, fmt.Errorf("no IntelligentPool found in namespace %s", r.namespace)
	}
	if len(poolList.Items) > 1 {
		return nil, fmt.Errorf("multiple IntelligentPools found in namespace %s, expected exactly 1", r.namespace)
	}

	return &poolList.Items[0], nil
}

// getIntelligentRoute retrieves the IntelligentRoute from the namespace
func (r *Reconciler) getIntelligentRoute(ctx context.Context) (*v1alpha1.IntelligentRoute, error) {
	routeList := &v1alpha1.IntelligentRouteList{}
	if err := r.client.List(ctx, routeList, client.InNamespace(r.namespace)); err != nil {
		return nil, fmt.Errorf("failed to list IntelligentRoutes: %w", err)
	}

	if len(routeList.Items) == 0 {
		return nil, fmt.Errorf("no IntelligentRoute found in namespace %s", r.namespace)
	}
	if len(routeList.Items) > 1 {
		return nil, fmt.Errorf("multiple IntelligentRoutes found in namespace %s, expected exactly 1", r.namespace)
	}

	return &routeList.Items[0], nil
}

// validateAndUpdate validates CRDs and updates configuration
func (r *Reconciler) validateAndUpdate(ctx context.Context, pool *v1alpha1.IntelligentPool, route *v1alpha1.IntelligentRoute) error {
	// Validate
	if err := r.validate(pool, route); err != nil {
		// Update status to Invalid
		r.updatePoolStatus(ctx, pool, metav1.ConditionFalse, "ValidationFailed", err.Error())
		r.updateRouteStatus(ctx, route, metav1.ConditionFalse, "ValidationFailed", err.Error())
		return err
	}

	// Convert to internal config
	backendModels, err := r.converter.ConvertIntelligentPool(pool)
	if err != nil {
		return fmt.Errorf("failed to convert IntelligentPool: %w", err)
	}

	intelligentRouting, err := r.converter.ConvertIntelligentRoute(route)
	if err != nil {
		return fmt.Errorf("failed to convert IntelligentRoute: %w", err)
	}

	// Create new config by merging with static config
	newConfig := *r.staticConfig
	newConfig.BackendModels = *backendModels

	// Copy IntelligentRouting fields explicitly (since it's embedded with ,inline in YAML)
	// Assigning the whole struct doesn't work correctly with embedded structs
	newConfig.KeywordRules = intelligentRouting.KeywordRules
	newConfig.EmbeddingRules = intelligentRouting.EmbeddingRules
	newConfig.Categories = intelligentRouting.Categories
	newConfig.Decisions = intelligentRouting.Decisions
	newConfig.Strategy = intelligentRouting.Strategy
	newConfig.ReasoningConfig = intelligentRouting.ReasoningConfig

	// Call update callback
	if r.onConfigUpdate != nil {
		if err := r.onConfigUpdate(&newConfig); err != nil {
			r.updatePoolStatus(ctx, pool, metav1.ConditionFalse, "UpdateFailed", err.Error())
			r.updateRouteStatus(ctx, route, metav1.ConditionFalse, "UpdateFailed", err.Error())
			return fmt.Errorf("config update failed: %w", err)
		}
	}

	// Update status to Ready
	r.updatePoolStatus(ctx, pool, metav1.ConditionTrue, "Ready", "Configuration applied successfully")
	r.updateRouteStatus(ctx, route, metav1.ConditionTrue, "Ready", "Configuration applied successfully")

	logging.Infof("Configuration updated successfully from CRDs")
	return nil
}

// validate validates the CRDs
func (r *Reconciler) validate(pool *v1alpha1.IntelligentPool, route *v1alpha1.IntelligentRoute) error {
	// Build model map
	modelMap := make(map[string]*v1alpha1.ModelConfig)
	for i := range pool.Spec.Models {
		model := &pool.Spec.Models[i]
		modelMap[model.Name] = model
	}

	// Build signal name sets
	keywordSignalNames := make(map[string]bool)
	embeddingSignalNames := make(map[string]bool)
	domainSignalNames := make(map[string]bool)

	for _, signal := range route.Spec.Signals.Keywords {
		if keywordSignalNames[signal.Name] {
			return fmt.Errorf("duplicate keyword signal name: %s", signal.Name)
		}
		keywordSignalNames[signal.Name] = true
	}

	for _, signal := range route.Spec.Signals.Embeddings {
		if embeddingSignalNames[signal.Name] {
			return fmt.Errorf("duplicate embedding signal name: %s", signal.Name)
		}
		embeddingSignalNames[signal.Name] = true
	}

	// Domains is now an array of DomainSignal with name and description
	for _, domain := range route.Spec.Signals.Domains {
		if domainSignalNames[domain.Name] {
			return fmt.Errorf("duplicate domain signal name: %s", domain.Name)
		}
		domainSignalNames[domain.Name] = true
	}

	// Validate decisions
	for _, decision := range route.Spec.Decisions {
		// Validate signal references
		for _, condition := range decision.Signals.Conditions {
			switch condition.Type {
			case "keyword":
				if !keywordSignalNames[condition.Name] {
					return fmt.Errorf("decision %s references unknown keyword signal: %s", decision.Name, condition.Name)
				}
			case "embedding":
				if !embeddingSignalNames[condition.Name] {
					return fmt.Errorf("decision %s references unknown embedding signal: %s", decision.Name, condition.Name)
				}
			case "domain":
				if !domainSignalNames[condition.Name] {
					return fmt.Errorf("decision %s references unknown domain signal: %s", decision.Name, condition.Name)
				}
			}
		}

		// Validate model scores
		for _, ms := range decision.ModelRefs {
			model, ok := modelMap[ms.Model]
			if !ok {
				return fmt.Errorf("decision %s references unknown model: %s", decision.Name, ms.Model)
			}

			// Validate LoRA reference
			if ms.LoRAName != "" {
				found := false
				for _, lora := range model.LoRAs {
					if lora.Name == ms.LoRAName {
						found = true
						break
					}
				}
				if !found {
					return fmt.Errorf("decision %s references unknown LoRA %s for model %s", decision.Name, ms.LoRAName, ms.Model)
				}
			}
		}
	}

	// Validate reasoning families
	if r.staticConfig != nil && r.staticConfig.ReasoningFamilies != nil {
		for _, model := range pool.Spec.Models {
			if model.ReasoningFamily != "" {
				if _, ok := r.staticConfig.ReasoningFamilies[model.ReasoningFamily]; !ok {
					return fmt.Errorf("model %s references unknown reasoning family: %s", model.Name, model.ReasoningFamily)
				}
			}
		}
	}

	return nil
}

// updatePoolStatus updates the status of IntelligentPool
func (r *Reconciler) updatePoolStatus(ctx context.Context, pool *v1alpha1.IntelligentPool, status metav1.ConditionStatus, reason, message string) {
	// Create a copy to update
	poolCopy := pool.DeepCopy()

	// Update conditions
	condition := metav1.Condition{
		Type:               "Ready",
		Status:             status,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: poolCopy.Generation,
	}

	// Find and update existing condition or append new one
	found := false
	for i, c := range poolCopy.Status.Conditions {
		if c.Type == "Ready" {
			poolCopy.Status.Conditions[i] = condition
			found = true
			break
		}
	}
	if !found {
		poolCopy.Status.Conditions = append(poolCopy.Status.Conditions, condition)
	}

	poolCopy.Status.ObservedGeneration = poolCopy.Generation
	poolCopy.Status.ModelCount = int32(len(poolCopy.Spec.Models)) //nolint:gosec // Model count is unlikely to overflow int32

	// Update status subresource
	if err := r.client.Status().Update(ctx, poolCopy); err != nil {
		logging.Errorf("Failed to update IntelligentPool status: %v", err)
	}
}

// updateRouteStatus updates the status of IntelligentRoute
func (r *Reconciler) updateRouteStatus(ctx context.Context, route *v1alpha1.IntelligentRoute, status metav1.ConditionStatus, reason, message string) {
	// Create a copy to update
	routeCopy := route.DeepCopy()

	// Update conditions
	condition := metav1.Condition{
		Type:               "Ready",
		Status:             status,
		Reason:             reason,
		Message:            message,
		LastTransitionTime: metav1.Now(),
		ObservedGeneration: routeCopy.Generation,
	}

	// Find and update existing condition or append new one
	found := false
	for i, c := range routeCopy.Status.Conditions {
		if c.Type == "Ready" {
			routeCopy.Status.Conditions[i] = condition
			found = true
			break
		}
	}
	if !found {
		routeCopy.Status.Conditions = append(routeCopy.Status.Conditions, condition)
	}

	routeCopy.Status.ObservedGeneration = routeCopy.Generation

	// Update statistics
	routeCopy.Status.Statistics = &v1alpha1.RouteStatistics{
		Decisions:  int32(len(routeCopy.Spec.Decisions)),          //nolint:gosec // Decision count is unlikely to overflow int32
		Keywords:   int32(len(routeCopy.Spec.Signals.Keywords)),   //nolint:gosec // Keyword count is unlikely to overflow int32
		Embeddings: int32(len(routeCopy.Spec.Signals.Embeddings)), //nolint:gosec // Embedding count is unlikely to overflow int32
		Domains:    int32(len(routeCopy.Spec.Signals.Domains)),    //nolint:gosec // Domain count is unlikely to overflow int32
	}

	// Update status subresource
	if err := r.client.Status().Update(ctx, routeCopy); err != nil {
		logging.Errorf("Failed to update IntelligentRoute status: %v", err)
	}
}
