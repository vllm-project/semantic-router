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
	"errors"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	yamlv3 "gopkg.in/yaml.v3"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
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
	runtimeManager manager.Manager
	namespace      string
	converter      *CRDConverter
	staticConfig   *config.RouterConfig
	onConfigUpdate func(context.Context, *config.RouterConfig) error
	mu             sync.RWMutex
	lastPool       *v1alpha1.IntelligentPool
	lastRoute      *v1alpha1.IntelligentRoute
}

// ReconcilerConfig holds configuration for the reconciler
type ReconcilerConfig struct {
	Namespace      string
	Kubeconfig     string // Optional: if empty, uses in-cluster config
	StaticConfig   *config.RouterConfig
	OnConfigUpdate func(context.Context, *config.RouterConfig) error
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
		runtimeManager: mgr,
		namespace:      cfg.Namespace,
		converter:      NewCRDConverter(),
		staticConfig:   cfg.StaticConfig,
		onConfigUpdate: cfg.OnConfigUpdate,
	}

	return reconciler, nil
}

// Start starts watching for CRD changes
func (r *Reconciler) Start(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	logging.Infof("Starting Kubernetes reconciler in namespace %s", r.namespace)
	runCtx, cancel := context.WithCancel(ctx)
	var workers sync.WaitGroup
	defer func() {
		cancel()
		workers.Wait()
	}()

	managerDone := make(chan error, 1)
	workers.Add(1)
	go func() {
		defer workers.Done()
		err := r.runtimeManager.Start(runCtx)
		cancel()
		managerDone <- err
	}()
	cacheSynced := make(chan bool, 1)
	workers.Add(1)
	go func() {
		defer workers.Done()
		cacheSynced <- r.runtimeManager.GetCache().WaitForCacheSync(runCtx)
	}()
	select {
	case <-ctx.Done():
		return nil
	case err := <-managerDone:
		if ctx.Err() != nil {
			return nil
		}
		return runtimeManagerExitError(err)
	case synced := <-cacheSynced:
		if !synced {
			if ctx.Err() != nil {
				return nil
			}
			return errors.New("failed to wait for cache sync")
		}
	}

	// Reconciliation may wait for model preparation and activation. Run it
	// beside the manager monitor so a terminal watcher error cancels that work.
	workers.Add(1)
	go func() {
		defer workers.Done()
		if err := r.reconcile(runCtx); err != nil && runCtx.Err() == nil {
			logging.Warnf("Initial reconciliation failed (will retry): %v", err)
		}
		r.watchLoop(runCtx)
	}()

	select {
	case <-ctx.Done():
		return nil
	case err := <-managerDone:
		if ctx.Err() != nil {
			return nil
		}
		return runtimeManagerExitError(err)
	}
}

func runtimeManagerExitError(err error) error {
	if err == nil {
		return errors.New("kubernetes runtime manager stopped unexpectedly")
	}
	return fmt.Errorf("kubernetes runtime manager: %w", err)
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
	// The reconciler emits a single canonical config and so expects exactly
	// one pool and one route. When several coexist, short-circuiting would
	// leave every CR with an empty status indefinitely (#1908); instead mark
	// each conflicting CR Ready=False so the conflict is observable.
	poolList := &v1alpha1.IntelligentPoolList{}
	if err := r.client.List(ctx, poolList, client.InNamespace(r.namespace)); err != nil {
		return fmt.Errorf("failed to list IntelligentPools: %w", err)
	}
	routeList := &v1alpha1.IntelligentRouteList{}
	if err := r.client.List(ctx, routeList, client.InNamespace(r.namespace)); err != nil {
		return fmt.Errorf("failed to list IntelligentRoutes: %w", err)
	}

	if len(poolList.Items) == 0 {
		return fmt.Errorf("no IntelligentPool found in namespace %s", r.namespace)
	}
	if len(routeList.Items) == 0 {
		return fmt.Errorf("no IntelligentRoute found in namespace %s", r.namespace)
	}

	if len(poolList.Items) > 1 || len(routeList.Items) > 1 {
		return r.reportConflict(ctx, poolList.Items, routeList.Items)
	}

	pool := &poolList.Items[0]
	route := &routeList.Items[0]

	// Check if anything changed
	r.mu.RLock()
	poolChanged := r.lastPool == nil || pool.Generation != r.lastPool.Generation || pool.UID != r.lastPool.UID || client.ObjectKeyFromObject(pool) != client.ObjectKeyFromObject(r.lastPool)
	routeChanged := r.lastRoute == nil || route.Generation != r.lastRoute.Generation || route.UID != r.lastRoute.UID || client.ObjectKeyFromObject(route) != client.ObjectKeyFromObject(r.lastRoute)
	r.mu.RUnlock()

	if !poolChanged && !routeChanged {
		// Runtime activation already succeeded. Retry failed status writes and
		// repair external status changes without rebuilding the generation.
		r.updatePoolStatus(ctx, pool, metav1.ConditionTrue, "Ready", "Configuration is active on this router")
		r.updateRouteStatus(ctx, route, metav1.ConditionTrue, "Ready", "Configuration is active on this router")
		return nil
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

// reportConflict marks every IntelligentPool and IntelligentRoute in the
// namespace Ready=False/Reason=Conflict and returns an error so the watch
// loop logs the condition without silently dropping all CRs' status.
func (r *Reconciler) reportConflict(ctx context.Context, pools []v1alpha1.IntelligentPool, routes []v1alpha1.IntelligentRoute) error {
	var msgs []string

	if len(pools) > 1 {
		msg := fmt.Sprintf("found %d IntelligentPools in namespace %s, expected exactly 1; resolve the conflict so a single configuration can be applied", len(pools), r.namespace)
		msgs = append(msgs, msg)
		for i := range pools {
			r.updatePoolStatus(ctx, &pools[i], metav1.ConditionFalse, "Conflict", msg)
		}
	}

	if len(routes) > 1 {
		msg := fmt.Sprintf("found %d IntelligentRoutes in namespace %s, expected exactly 1; resolve the conflict so a single configuration can be applied", len(routes), r.namespace)
		msgs = append(msgs, msg)
		for i := range routes {
			r.updateRouteStatus(ctx, &routes[i], metav1.ConditionFalse, "Conflict", msg)
		}
	}

	return errors.New(strings.Join(msgs, "; "))
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

	canonicalBase := config.CanonicalStaticConfigFromRouterConfig(r.staticConfig)
	canonicalCfg, err := r.converter.Convert(pool, route, &canonicalBase)
	if err != nil {
		return fmt.Errorf("failed to convert CRDs to canonical config: %w", err)
	}

	canonicalBytes, err := yamlv3.Marshal(canonicalCfg)
	if err != nil {
		return fmt.Errorf("failed to marshal canonical config: %w", err)
	}

	newConfig, err := config.ParseYAMLBytes(canonicalBytes)
	if err != nil {
		r.updatePoolStatus(ctx, pool, metav1.ConditionFalse, "ValidationFailed", err.Error())
		r.updateRouteStatus(ctx, route, metav1.ConditionFalse, "ValidationFailed", err.Error())
		return fmt.Errorf("failed to normalize canonical config: %w", err)
	}
	newConfig.ConfigSource = config.ConfigSourceKubernetes

	// Parsing already validates static global settings. Now validate the complete
	// routing graph from CRDs before publishing the candidate to the runtime.
	if err := config.ValidateKubernetesConfigContracts(newConfig); err != nil {
		r.updatePoolStatus(ctx, pool, metav1.ConditionFalse, "ValidationFailed", err.Error())
		r.updateRouteStatus(ctx, route, metav1.ConditionFalse, "ValidationFailed", err.Error())
		return fmt.Errorf("kubernetes config validation failed: %w", err)
	}

	// Status publication is best-effort; it must never prevent runtime activation.
	r.updatePoolStatus(ctx, pool, metav1.ConditionFalse, "Activating", "Waiting for runtime activation")
	r.updateRouteStatus(ctx, route, metav1.ConditionFalse, "Activating", "Waiting for runtime activation")
	var activationErr error
	if r.onConfigUpdate == nil {
		activationErr = errors.New("runtime activation callback is not configured")
	} else {
		activationErr = r.onConfigUpdate(ctx, newConfig)
	}
	if activationErr != nil {
		r.updatePoolStatus(ctx, pool, metav1.ConditionFalse, "ActivationFailed", activationErr.Error())
		r.updateRouteStatus(ctx, route, metav1.ConditionFalse, "ActivationFailed", activationErr.Error())
		return fmt.Errorf("config activation failed: %w", activationErr)
	}

	// Update status to Ready
	r.updatePoolStatus(ctx, pool, metav1.ConditionTrue, "Ready", "Configuration is active on this router")
	r.updateRouteStatus(ctx, route, metav1.ConditionTrue, "Ready", "Configuration is active on this router")

	logging.Infof("Configuration updated successfully from CRDs")
	return nil
}

// validate validates the CRDs
func (r *Reconciler) validate(pool *v1alpha1.IntelligentPool, route *v1alpha1.IntelligentRoute) error {
	var reasoningFamilies map[string]config.ReasoningFamilyConfig
	if r.staticConfig != nil {
		reasoningFamilies = r.staticConfig.ReasoningFamilies
	}
	return validatePoolRoute(pool, route, reasoningFamilies)
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

	apimeta.SetStatusCondition(&poolCopy.Status.Conditions, condition)

	poolCopy.Status.ObservedGeneration = poolCopy.Generation
	poolCopy.Status.ModelCount = int32(len(poolCopy.Spec.Models)) //nolint:gosec // Model count is unlikely to overflow int32

	if reflect.DeepEqual(pool.Status, poolCopy.Status) {
		return
	}
	if err := r.client.Status().Update(ctx, poolCopy); err != nil {
		logging.Errorf("Failed to update IntelligentPool status: %v", err)
		return
	}
	*pool = *poolCopy
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

	apimeta.SetStatusCondition(&routeCopy.Status.Conditions, condition)

	routeCopy.Status.ObservedGeneration = routeCopy.Generation

	// Update statistics
	routeCopy.Status.Statistics = &v1alpha1.RouteStatistics{
		Decisions:  int32(len(routeCopy.Spec.Decisions)),          //nolint:gosec // Decision count is unlikely to overflow int32
		Keywords:   int32(len(routeCopy.Spec.Signals.Keywords)),   //nolint:gosec // Keyword count is unlikely to overflow int32
		Embeddings: int32(len(routeCopy.Spec.Signals.Embeddings)), //nolint:gosec // Embedding count is unlikely to overflow int32
		Domains:    int32(len(routeCopy.Spec.Signals.Domains)),    //nolint:gosec // Domain count is unlikely to overflow int32
	}

	if reflect.DeepEqual(route.Status, routeCopy.Status) {
		return
	}
	if err := r.client.Status().Update(ctx, routeCopy); err != nil {
		logging.Errorf("Failed to update IntelligentRoute status: %v", err)
		return
	}
	*route = *routeCopy
}
