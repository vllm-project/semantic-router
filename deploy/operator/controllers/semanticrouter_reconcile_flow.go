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

	"github.com/go-logr/logr"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func (r *SemanticRouterReconciler) isRunningOnOpenShift(ctx context.Context) bool {
	r.isOpenShiftOnce.Do(func() {
		logger := log.FromContext(ctx)

		route := &metav1.PartialObjectMetadata{}
		route.SetGroupVersionKind(schema.GroupVersionKind{
			Group:   "route.openshift.io",
			Version: "v1",
			Kind:    "Route",
		})

		err := r.List(ctx, &metav1.PartialObjectMetadataList{
			TypeMeta: metav1.TypeMeta{
				APIVersion: "route.openshift.io/v1",
				Kind:       "Route",
			},
		}, &client.ListOptions{Limit: 1})

		isOpenShift := err == nil || !meta.IsNoMatchError(err)
		r.isOpenShift = &isOpenShift

		if isOpenShift {
			logger.Info("Detected OpenShift platform - will use OpenShift-compatible security contexts")
		} else {
			logger.Info("Detected standard Kubernetes platform - will use standard security contexts")
		}
	})

	if r.isOpenShift != nil {
		return *r.isOpenShift
	}
	return false
}

func (r *SemanticRouterReconciler) fetchSemanticRouter(ctx context.Context, req ctrl.Request) (*vllmv1alpha1.SemanticRouter, error) {
	logger := log.FromContext(ctx)

	semanticrouter := &vllmv1alpha1.SemanticRouter{}
	err := r.Get(ctx, req.NamespacedName, semanticrouter)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.Info("SemanticRouter resource not found. Ignoring since object must be deleted")
			return nil, nil
		}
		logger.Error(err, "Failed to get SemanticRouter")
		return nil, err
	}

	return semanticrouter, nil
}

func (r *SemanticRouterReconciler) handleFinalizerFlow(ctx context.Context, semanticrouter *vllmv1alpha1.SemanticRouter) (done bool, err error) {
	if semanticrouter.DeletionTimestamp.IsZero() {
		if !controllerutil.ContainsFinalizer(semanticrouter, SemanticRouterFinalizer) {
			controllerutil.AddFinalizer(semanticrouter, SemanticRouterFinalizer)
			if err := r.Update(ctx, semanticrouter); err != nil {
				return false, err
			}
		}
		return false, nil
	}

	if !controllerutil.ContainsFinalizer(semanticrouter, SemanticRouterFinalizer) {
		return true, nil
	}

	if err := r.finalizeSemanticRouter(ctx, semanticrouter); err != nil {
		return false, err
	}

	controllerutil.RemoveFinalizer(semanticrouter, SemanticRouterFinalizer)
	if err := r.Update(ctx, semanticrouter); err != nil {
		return false, err
	}
	return true, nil
}

func (r *SemanticRouterReconciler) ensureInitialProgressingStatus(
	ctx context.Context,
	req ctrl.Request,
	semanticrouter *vllmv1alpha1.SemanticRouter,
	logger logr.Logger,
) (requeue bool, err error) {
	if len(semanticrouter.Status.Conditions) != 0 {
		return false, nil
	}

	meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
		Type:    typeProgressingSemanticRouter,
		Status:  metav1.ConditionTrue,
		Reason:  "Reconciling",
		Message: "Starting reconciliation",
	})

	err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current := &vllmv1alpha1.SemanticRouter{}
		if err := r.Get(ctx, req.NamespacedName, current); err != nil {
			return err
		}
		meta.SetStatusCondition(&current.Status.Conditions, metav1.Condition{
			Type:    typeProgressingSemanticRouter,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: "Starting reconciliation",
		})
		return r.Status().Update(ctx, current)
	})
	if err != nil {
		logger.Error(err, "Failed to update initial SemanticRouter status, will retry on next reconcile")
	}
	return true, nil
}

func (r *SemanticRouterReconciler) reconcileOwnedResources(
	ctx context.Context,
	semanticrouter *vllmv1alpha1.SemanticRouter,
	logger logr.Logger,
) error {
	if err := r.reconcileServiceAccount(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile ServiceAccount")
		return err
	}

	if err := r.reconcileConfigMap(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile ConfigMap")
		return err
	}

	if err := r.reconcilePVC(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile PersistentVolumeClaim")
		return err
	}

	gatewayMode, err := reconcileGatewayIntegration(ctx, r.Client, r.Scheme, semanticrouter)
	if err != nil {
		logger.Error(err, "Gateway integration failed")
		return err
	}
	semanticrouter.Status.GatewayMode = gatewayMode
	logger.Info("Gateway mode determined", "mode", gatewayMode)

	if err := r.reconcileEnvoyConfig(ctx, semanticrouter, gatewayMode); err != nil {
		logger.Error(err, "Failed to reconcile Envoy ConfigMap")
		return err
	}

	if err := r.reconcileDeployment(ctx, semanticrouter, gatewayMode); err != nil {
		logger.Error(err, "Failed to reconcile Deployment")
		return err
	}

	if err := r.reconcileService(ctx, semanticrouter, gatewayMode); err != nil {
		logger.Error(err, "Failed to reconcile Service")
		return err
	}

	if err := r.reconcileHPA(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile HorizontalPodAutoscaler")
		return err
	}

	if err := r.reconcileIngress(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile Ingress")
		return err
	}

	isOpenShift := false
	if r.isOpenShift != nil {
		isOpenShift = *r.isOpenShift
	}
	if err := reconcileRoute(ctx, r.Client, r.Scheme, semanticrouter, isOpenShift); err != nil {
		logger.Error(err, "Route reconciliation failed")
		return err
	}

	return nil
}
