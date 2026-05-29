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
	"fmt"
	"reflect"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func (r *SemanticRouterReconciler) reconcileServiceAccount(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if sr.Spec.ServiceAccount.Create == nil || !*sr.Spec.ServiceAccount.Create {
		return nil
	}

	saName := sr.Spec.ServiceAccount.Name
	if saName == "" {
		saName = sr.Name
	}

	sa := &corev1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:        saName,
			Namespace:   sr.Namespace,
			Annotations: sr.Spec.ServiceAccount.Annotations,
		},
	}

	if err := controllerutil.SetControllerReference(sr, sa, r.Scheme); err != nil {
		return err
	}

	found := &corev1.ServiceAccount{}
	err := r.Get(ctx, types.NamespacedName{Name: sa.Name, Namespace: sa.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, sa)
	} else if err != nil {
		return err
	}

	if !reflect.DeepEqual(found.Annotations, sa.Annotations) {
		found.Annotations = sa.Annotations
		return r.Update(ctx, found)
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcilePVC(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if sr.Spec.Persistence.Enabled == nil || !*sr.Spec.Persistence.Enabled {
		return nil
	}

	if sr.Spec.Persistence.ExistingClaim != "" {
		return nil
	}

	validatedStorageClass, err := validateStorageClass(ctx, r.Client, sr.Spec.Persistence.StorageClassName)
	if err != nil {
		return fmt.Errorf("StorageClass validation failed: %w", err)
	}

	sr.Spec.Persistence.StorageClassName = validatedStorageClass

	pvc, err := r.generatePVC(sr)
	if err != nil {
		return fmt.Errorf("failed to generate PVC: %w", err)
	}
	if err := controllerutil.SetControllerReference(sr, pvc, r.Scheme); err != nil {
		return err
	}

	found := &corev1.PersistentVolumeClaim{}
	err = r.Get(ctx, types.NamespacedName{Name: pvc.Name, Namespace: pvc.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, pvc)
	}

	return err
}

func (r *SemanticRouterReconciler) reconcileDeployment(ctx context.Context, sr *vllmv1alpha1.SemanticRouter, gatewayMode string) error {
	deployment := r.generateDeployment(sr, gatewayMode)
	if err := controllerutil.SetControllerReference(sr, deployment, r.Scheme); err != nil {
		return err
	}

	found := &appsv1.Deployment{}
	err := r.Get(ctx, types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, deployment)
	} else if err != nil {
		return err
	}

	if !reflect.DeepEqual(found.Spec, deployment.Spec) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := r.Get(ctx, types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, found); err != nil {
				return err
			}
			found.Spec = deployment.Spec
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcileService(ctx context.Context, sr *vllmv1alpha1.SemanticRouter, gatewayMode string) error {
	svc := r.generateService(sr, gatewayMode)
	if err := controllerutil.SetControllerReference(sr, svc, r.Scheme); err != nil {
		return err
	}

	found := &corev1.Service{}
	err := r.Get(ctx, types.NamespacedName{Name: svc.Name, Namespace: svc.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, svc)
	} else if err != nil {
		return err
	}

	if !reflect.DeepEqual(found.Spec.Ports, svc.Spec.Ports) ||
		found.Spec.Type != svc.Spec.Type ||
		!reflect.DeepEqual(found.Spec.Selector, svc.Spec.Selector) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := r.Get(ctx, types.NamespacedName{Name: svc.Name, Namespace: svc.Namespace}, found); err != nil {
				return err
			}
			svc.Spec.ClusterIP = found.Spec.ClusterIP
			found.Spec = svc.Spec
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcileHPA(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if sr.Spec.Autoscaling.Enabled == nil || !*sr.Spec.Autoscaling.Enabled {
		hpa := &autoscalingv2.HorizontalPodAutoscaler{}
		err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, hpa)
		if err == nil {
			return r.Delete(ctx, hpa)
		}
		return nil
	}

	hpa := r.generateHPA(sr)
	if err := controllerutil.SetControllerReference(sr, hpa, r.Scheme); err != nil {
		return err
	}

	found := &autoscalingv2.HorizontalPodAutoscaler{}
	err := r.Get(ctx, types.NamespacedName{Name: hpa.Name, Namespace: hpa.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, hpa)
	} else if err != nil {
		return err
	}

	if !reflect.DeepEqual(found.Spec, hpa.Spec) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := r.Get(ctx, types.NamespacedName{Name: hpa.Name, Namespace: hpa.Namespace}, found); err != nil {
				return err
			}
			found.Spec = hpa.Spec
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcileIngress(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if sr.Spec.Ingress.Enabled == nil || !*sr.Spec.Ingress.Enabled {
		ing := &networkingv1.Ingress{}
		err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, ing)
		if err == nil {
			return r.Delete(ctx, ing)
		}
		return nil
	}

	ing := r.generateIngress(sr)
	if err := controllerutil.SetControllerReference(sr, ing, r.Scheme); err != nil {
		return err
	}

	found := &networkingv1.Ingress{}
	err := r.Get(ctx, types.NamespacedName{Name: ing.Name, Namespace: ing.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, ing)
	} else if err != nil {
		return err
	}

	if !reflect.DeepEqual(found.Spec, ing.Spec) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := r.Get(ctx, types.NamespacedName{Name: ing.Name, Namespace: ing.Namespace}, found); err != nil {
				return err
			}
			found.Spec = ing.Spec
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) updateStatus(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		current := &vllmv1alpha1.SemanticRouter{}
		if err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, current); err != nil {
			return err
		}

		deployment := &appsv1.Deployment{}
		err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment)
		if err != nil {
			if errors.IsNotFound(err) {
				current.Status.Replicas = 0
				current.Status.ReadyReplicas = 0
				current.Status.ObservedGeneration = current.Generation
				current.Status.Phase = "Pending"
				meta.SetStatusCondition(&current.Status.Conditions, metav1.Condition{
					Type:    typeAvailableSemanticRouter,
					Status:  metav1.ConditionFalse,
					Reason:  "DeploymentNotFound",
					Message: "Deployment has not been created yet",
				})
				return r.Status().Update(ctx, current)
			}
			return err
		}

		current.Status.Replicas = deployment.Status.Replicas
		current.Status.ReadyReplicas = deployment.Status.ReadyReplicas
		current.Status.ObservedGeneration = current.Generation

		if deployment.Status.ReadyReplicas == 0 {
			current.Status.Phase = "Pending"
			meta.SetStatusCondition(&current.Status.Conditions, metav1.Condition{
				Type:    typeAvailableSemanticRouter,
				Status:  metav1.ConditionFalse,
				Reason:  "Pending",
				Message: "No replicas are ready",
			})
		} else if deployment.Status.ReadyReplicas < deployment.Status.Replicas {
			current.Status.Phase = "Progressing"
			meta.SetStatusCondition(&current.Status.Conditions, metav1.Condition{
				Type:    typeProgressingSemanticRouter,
				Status:  metav1.ConditionTrue,
				Reason:  "Progressing",
				Message: fmt.Sprintf("%d/%d replicas ready", deployment.Status.ReadyReplicas, deployment.Status.Replicas),
			})
		} else {
			current.Status.Phase = "Running"
			meta.SetStatusCondition(&current.Status.Conditions, metav1.Condition{
				Type:    typeAvailableSemanticRouter,
				Status:  metav1.ConditionTrue,
				Reason:  "AllReplicasReady",
				Message: "All replicas are ready",
			})
			meta.RemoveStatusCondition(&current.Status.Conditions, typeProgressingSemanticRouter)
		}

		return r.Status().Update(ctx, current)
	})
}

func (r *SemanticRouterReconciler) finalizeSemanticRouter(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	logger := log.FromContext(ctx)
	logger.Info("Finalizing SemanticRouter", "name", sr.Name, "namespace", sr.Namespace)

	if err := r.deleteOwnedPVCIfPresent(ctx, sr); err != nil {
		return err
	}

	logger.Info("Successfully finalized SemanticRouter")
	return nil
}

func (r *SemanticRouterReconciler) deleteOwnedPVCIfPresent(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if sr.Spec.Persistence.Enabled == nil || !*sr.Spec.Persistence.Enabled {
		return nil
	}
	if sr.Spec.Persistence.ExistingClaim != "" {
		return nil
	}

	logger := log.FromContext(ctx)
	pvcName := sr.Name + "-models"
	pvc := &corev1.PersistentVolumeClaim{}
	err := r.Get(ctx, types.NamespacedName{Name: pvcName, Namespace: sr.Namespace}, pvc)
	if errors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	logger.Info("Deleting PVC", "name", pvcName)
	if err := r.Delete(ctx, pvc); err != nil {
		logger.Error(err, "Failed to delete PVC", "name", pvcName)
		return err
	}
	return nil
}
