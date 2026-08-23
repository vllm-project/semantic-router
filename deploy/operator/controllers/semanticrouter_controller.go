/*
Copyright 2026 vLLM Semantic Router Contributors.

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

package controllers

import (
	"context"
	"sync"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1 "k8s.io/api/policy/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const bootstrapConfigMapRefIndex = ".spec.bootstrap.configMapRef.name"

// SemanticRouterReconciler reconciles a SemanticRouter object
type SemanticRouterReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// Cache for OpenShift detection
	isOpenShift     *bool
	isOpenShiftOnce sync.Once
}

// +kubebuilder:rbac:groups=vllm.ai,resources=semanticrouters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=vllm.ai,resources=semanticrouters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=vllm.ai,resources=semanticrouters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=batch,resources=jobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch
// +kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses;networkpolicies,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=policy,resources=poddisruptionbudgets,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=route.openshift.io,resources=routes,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=gateway.networking.k8s.io,resources=gateways,verbs=get;list;watch
// +kubebuilder:rbac:groups=storage.k8s.io,resources=storageclasses,verbs=get;list;watch

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *SemanticRouterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)
	r.isRunningOnOpenShift(ctx)

	semanticrouter, err := r.fetchSemanticRouter(ctx, req)
	if err != nil || semanticrouter == nil {
		return ctrl.Result{}, err
	}

	done, err := r.handleFinalizerFlow(ctx, semanticrouter)
	if err != nil || done {
		return ctrl.Result{}, err
	}

	if requeue, err := r.ensureInitialProgressingStatus(ctx, req, semanticrouter, logger); requeue || err != nil {
		if err != nil {
			return ctrl.Result{}, err
		}
		return ctrl.Result{Requeue: true}, nil
	}

	baseSR := semanticrouter.DeepCopy()

	result, reconcileErr := r.reconcileOwnedResources(ctx, semanticrouter, logger)
	if reconcileErr != nil {
		semanticrouter.Status.Phase = "Degraded"
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type:               typeDegradedSemanticRouter,
			Status:             metav1.ConditionTrue,
			Reason:             "ReconcileFailed",
			Message:            reconcileErr.Error(),
			ObservedGeneration: semanticrouter.Generation,
		})
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type:               typeAvailableSemanticRouter,
			Status:             metav1.ConditionFalse,
			Reason:             "ReconcileFailed",
			Message:            "The requested deployment contract is not ready",
			ObservedGeneration: semanticrouter.Generation,
		})
	}

	if err := r.updateStatus(ctx, semanticrouter, baseSR); err != nil {
		logger.Error(err, "Failed to update SemanticRouter status, will retry on next reconcile")
		if reconcileErr == nil {
			return ctrl.Result{}, err
		}
	}
	if reconcileErr != nil {
		return ctrl.Result{}, reconcileErr
	}
	return result, nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *SemanticRouterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if err := mgr.GetFieldIndexer().IndexField(
		context.Background(),
		&vllmv1alpha1.SemanticRouter{},
		bootstrapConfigMapRefIndex,
		func(object client.Object) []string {
			router := object.(*vllmv1alpha1.SemanticRouter)
			if router.Spec.Bootstrap.ConfigMapRef.Name == "" {
				return nil
			}
			return []string{router.Spec.Bootstrap.ConfigMapRef.Name}
		},
	); err != nil {
		return err
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&vllmv1alpha1.SemanticRouter{}).
		Owns(&appsv1.Deployment{}).
		Owns(&batchv1.Job{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.PersistentVolumeClaim{}).
		Owns(&autoscalingv2.HorizontalPodAutoscaler{}).
		Owns(&networkingv1.Ingress{}).
		Owns(&networkingv1.NetworkPolicy{}).
		Owns(&policyv1.PodDisruptionBudget{}).
		Watches(&corev1.ConfigMap{}, handler.EnqueueRequestsFromMapFunc(func(ctx context.Context, object client.Object) []reconcile.Request {
			routers := &vllmv1alpha1.SemanticRouterList{}
			if err := mgr.GetClient().List(
				ctx,
				routers,
				client.InNamespace(object.GetNamespace()),
				client.MatchingFields{bootstrapConfigMapRefIndex: object.GetName()},
			); err != nil {
				log.FromContext(ctx).Error(err, "Failed to list SemanticRouters for bootstrap ConfigMap")
				return nil
			}
			requests := make([]reconcile.Request, 0, len(routers.Items))
			for index := range routers.Items {
				requests = append(requests, reconcile.Request{NamespacedName: client.ObjectKeyFromObject(&routers.Items[index])})
			}
			return requests
		})).
		Complete(r)
}
