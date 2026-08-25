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
	"time"

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
) (ctrl.Result, error) {
	bootstrap, prerequisiteResult, pending, err := r.reconcileBootstrapPrerequisites(
		ctx, semanticrouter, logger,
	)
	if err != nil || pending {
		return prerequisiteResult, err
	}

	gatewayMode, err := resolveGatewayMode(ctx, r.Client, semanticrouter)
	if err != nil {
		logger.Error(err, "Gateway integration failed")
		return ctrl.Result{}, err
	}
	semanticrouter.Status.GatewayMode = gatewayMode
	logger.Info("Gateway mode determined", "mode", gatewayMode)

	if err := r.reconcileEnvoyConfig(ctx, semanticrouter, gatewayMode); err != nil {
		logger.Error(err, "Failed to reconcile Envoy ConfigMap")
		return ctrl.Result{}, err
	}

	if err := r.reconcileDeployment(ctx, semanticrouter, gatewayMode, bootstrap); err != nil {
		logger.Error(err, "Failed to reconcile Deployment")
		return ctrl.Result{}, err
	}

	if err := r.reconcileServices(ctx, semanticrouter, gatewayMode, bootstrap); err != nil {
		logger.Error(err, "Failed to reconcile Services")
		return ctrl.Result{}, err
	}

	if err := r.reconcilePodDisruptionBudget(ctx, semanticrouter, bootstrap.enablesAvailabilityDefaults()); err != nil {
		logger.Error(err, "Failed to reconcile PodDisruptionBudget")
		return ctrl.Result{}, err
	}

	if err := r.reconcileNetworkPolicy(ctx, semanticrouter, gatewayMode, bootstrap); err != nil {
		logger.Error(err, "Failed to reconcile NetworkPolicy")
		return ctrl.Result{}, err
	}

	if err := r.reconcileHPA(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile HorizontalPodAutoscaler")
		return ctrl.Result{}, err
	}

	if err := r.reconcileIngress(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile Ingress")
		return ctrl.Result{}, err
	}

	isOpenShift := false
	if r.isOpenShift != nil {
		isOpenShift = *r.isOpenShift
	}
	if err := reconcileRoute(
		ctx,
		r.Client,
		r.Scheme,
		semanticrouter,
		isOpenShift,
		gatewayMode,
		bootstrap.usesDurableState(),
	); err != nil {
		logger.Error(err, "Route reconciliation failed")
		return ctrl.Result{}, err
	}

	semanticrouter.Status.Phase = "Progressing"
	meta.RemoveStatusCondition(&semanticrouter.Status.Conditions, typeDegradedSemanticRouter)
	return ctrl.Result{}, nil
}

func (r *SemanticRouterReconciler) reconcileBootstrapPrerequisites(
	ctx context.Context,
	semanticrouter *vllmv1alpha1.SemanticRouter,
	logger logr.Logger,
) (bootstrapDeploymentContract, ctrl.Result, bool, error) {
	bootstrap, err := r.validateBootstrapForReconcile(ctx, semanticrouter, logger)
	if err != nil {
		return bootstrapDeploymentContract{}, ctrl.Result{}, false, err
	}
	if err := r.reconcileServiceAccount(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile ServiceAccount")
		return bootstrap, ctrl.Result{}, false, err
	}
	if err := r.reconcilePVC(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile PersistentVolumeClaim")
		return bootstrap, ctrl.Result{}, false, err
	}
	result, pending, err := r.reconcileMigrationPrerequisite(
		ctx, semanticrouter, bootstrap,
	)
	return bootstrap, result, pending, err
}

func (r *SemanticRouterReconciler) validateBootstrapForReconcile(
	ctx context.Context,
	semanticrouter *vllmv1alpha1.SemanticRouter,
	logger logr.Logger,
) (bootstrapDeploymentContract, error) {
	bootstrap, err := r.validateBootstrapConfigMap(ctx, semanticrouter)
	if err != nil {
		logger.Error(err, "Bootstrap ConfigMap validation failed")
		semanticrouter.Status.BootstrapRevision = ""
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type: typeBootstrapReady, Status: metav1.ConditionFalse, Reason: "ValidationFailed",
			Message: err.Error(), ObservedGeneration: semanticrouter.Generation,
		})
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type: typeMigrationReady, Status: metav1.ConditionFalse, Reason: "BootstrapInvalid",
			Message:            "Migration requirements cannot be resolved until the bootstrap is valid",
			ObservedGeneration: semanticrouter.Generation,
		})
		return bootstrapDeploymentContract{}, err
	}
	semanticrouter.Status.BootstrapRevision = bootstrap.Revision
	semanticrouter.Status.PublicService = semanticrouter.Name
	semanticrouter.Status.ManagementService = ""
	if bootstrap.exposesManagementAPI() {
		semanticrouter.Status.ManagementService = semanticrouter.Name + "-management"
	}
	meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
		Type: typeBootstrapReady, Status: metav1.ConditionTrue, Reason: "Validated",
		Message: "The immutable Router bootstrap is valid", ObservedGeneration: semanticrouter.Generation,
	})
	return bootstrap, nil
}

func (r *SemanticRouterReconciler) reconcileMigrationPrerequisite(
	ctx context.Context,
	semanticrouter *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) (ctrl.Result, bool, error) {
	if !bootstrap.usesDurableState() {
		if err := r.deleteStaleMigrationJobs(ctx, semanticrouter, ""); err != nil {
			return ctrl.Result{}, false, err
		}
		semanticrouter.Status.Migration = nil
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type: typeMigrationReady, Status: metav1.ConditionTrue, Reason: "NotRequired",
			Message: "No Management store is configured", ObservedGeneration: semanticrouter.Generation,
		})
		return ctrl.Result{}, false, nil
	}
	migration, err := r.reconcileMigrationJob(ctx, semanticrouter, bootstrap)
	semanticrouter.Status.Migration = migration
	if err != nil {
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type: typeMigrationReady, Status: metav1.ConditionFalse, Reason: "Failed",
			Message: err.Error(), ObservedGeneration: semanticrouter.Generation,
		})
		return ctrl.Result{}, false, err
	}
	if migration.State != migrationStateSucceeded {
		semanticrouter.Status.Phase = "Migrating"
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type: typeMigrationReady, Status: metav1.ConditionFalse, Reason: migration.State,
			Message: "Waiting for the Management schema migration Job", ObservedGeneration: semanticrouter.Generation,
		})
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type: typeAvailableSemanticRouter, Status: metav1.ConditionFalse, Reason: "MigrationPending",
			Message: "Router rollout is gated on schema migration", ObservedGeneration: semanticrouter.Generation,
		})
		return ctrl.Result{RequeueAfter: 5 * time.Second}, true, nil
	}
	meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
		Type: typeMigrationReady, Status: metav1.ConditionTrue, Reason: "Succeeded",
		Message: "The Management schema is current", ObservedGeneration: semanticrouter.Generation,
	})
	return ctrl.Result{}, false, nil
}
