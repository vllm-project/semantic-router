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
	"encoding/json"
	"fmt"
	"reflect"
	"sync"

	appsv1 "k8s.io/api/apps/v1"
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
	"gopkg.in/yaml.v3"
)

const (
	typeAvailableSemanticRouter   = "Available"
	typeProgressingSemanticRouter = "Progressing"
	typeDegradedSemanticRouter    = "Degraded"
)

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
// +kubebuilder:rbac:groups=core,resources=services,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=serviceaccounts,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=configmaps,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=persistentvolumeclaims,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=autoscaling,resources=horizontalpodautoscalers,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=networking.k8s.io,resources=ingresses,verbs=get;list;watch;create;update;patch;delete

// Reconcile is part of the main kubernetes reconciliation loop which aims to
// move the current state of the cluster closer to the desired state.
func (r *SemanticRouterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	// Detect platform (cached after first call)
	r.isRunningOnOpenShift(ctx)

	// Fetch the SemanticRouter instance
	semanticrouter := &vllmv1alpha1.SemanticRouter{}
	err := r.Get(ctx, req.NamespacedName, semanticrouter)
	if err != nil {
		if errors.IsNotFound(err) {
			logger.Info("SemanticRouter resource not found. Ignoring since object must be deleted")
			return ctrl.Result{}, nil
		}
		logger.Error(err, "Failed to get SemanticRouter")
		return ctrl.Result{}, err
	}

	// Check if the SemanticRouter instance is marked for deletion
	if semanticrouter.DeletionTimestamp.IsZero() {
		// The object is not being deleted, so register finalizer if not present
		if !controllerutil.ContainsFinalizer(semanticrouter, SemanticRouterFinalizer) {
			controllerutil.AddFinalizer(semanticrouter, SemanticRouterFinalizer)
			if err := r.Update(ctx, semanticrouter); err != nil {
				return ctrl.Result{}, err
			}
		}
	} else {
		// The object is being deleted
		if controllerutil.ContainsFinalizer(semanticrouter, SemanticRouterFinalizer) {
			// Run finalization logic
			if err := r.finalizeSemanticRouter(ctx, semanticrouter); err != nil {
				return ctrl.Result{}, err
			}

			// Remove finalizer to allow deletion
			controllerutil.RemoveFinalizer(semanticrouter, SemanticRouterFinalizer)
			if err := r.Update(ctx, semanticrouter); err != nil {
				return ctrl.Result{}, err
			}
		}
		return ctrl.Result{}, nil
	}

	// Set status to Progressing (best-effort, don't fail reconcile on status update errors)
	if len(semanticrouter.Status.Conditions) == 0 {
		meta.SetStatusCondition(&semanticrouter.Status.Conditions, metav1.Condition{
			Type:    typeProgressingSemanticRouter,
			Status:  metav1.ConditionTrue,
			Reason:  "Reconciling",
			Message: "Starting reconciliation",
		})
		// Use retry logic for initial status update
		err := retry.RetryOnConflict(retry.DefaultRetry, func() error {
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
			// Log but don't fail - status updates are best-effort
			logger.Error(err, "Failed to update initial SemanticRouter status, will retry on next reconcile")
		}
		return ctrl.Result{Requeue: true}, nil
	}

	// Reconcile ServiceAccount
	if err := r.reconcileServiceAccount(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile ServiceAccount")
		return ctrl.Result{}, err
	}

	// Reconcile ConfigMap
	if err := r.reconcileConfigMap(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile ConfigMap")
		return ctrl.Result{}, err
	}

	// Reconcile PersistentVolumeClaim
	if err := r.reconcilePVC(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile PersistentVolumeClaim")
		return ctrl.Result{}, err
	}

	// Reconcile Deployment
	if err := r.reconcileDeployment(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile Deployment")
		return ctrl.Result{}, err
	}

	// Reconcile Service
	if err := r.reconcileService(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile Service")
		return ctrl.Result{}, err
	}

	// Reconcile HPA
	if err := r.reconcileHPA(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile HorizontalPodAutoscaler")
		return ctrl.Result{}, err
	}

	// Reconcile Ingress
	if err := r.reconcileIngress(ctx, semanticrouter); err != nil {
		logger.Error(err, "Failed to reconcile Ingress")
		return ctrl.Result{}, err
	}

	// Update status based on deployment status (best-effort, don't fail reconcile)
	if err := r.updateStatus(ctx, semanticrouter); err != nil {
		// Log but don't fail - status updates are best-effort and will retry on next reconcile
		logger.Error(err, "Failed to update SemanticRouter status, will retry on next reconcile")
	}

	// Reconciliation complete - rely on watch events for future updates
	return ctrl.Result{}, nil
}

// isRunningOnOpenShift detects if the operator is running on OpenShift
// by checking for OpenShift-specific API resources
func (r *SemanticRouterReconciler) isRunningOnOpenShift(ctx context.Context) bool {
	r.isOpenShiftOnce.Do(func() {
		logger := log.FromContext(ctx)

		// Check for route.openshift.io/v1 Route resource which is OpenShift-specific
		// This is a simpler check that works well for platform detection
		route := &metav1.PartialObjectMetadata{}
		route.SetGroupVersionKind(schema.GroupVersionKind{
			Group:   "route.openshift.io",
			Version: "v1",
			Kind:    "Route",
		})

		// Try to list routes - this will fail on standard Kubernetes
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

	// Update if annotations changed
	if !reflect.DeepEqual(found.Annotations, sa.Annotations) {
		found.Annotations = sa.Annotations
		return r.Update(ctx, found)
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcileConfigMap(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	configData, err := r.generateConfigYAML(sr)
	if err != nil {
		return err
	}

	toolsData, err := r.generateToolsJSON(sr)
	if err != nil {
		return err
	}

	cm := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name + "-config",
			Namespace: sr.Namespace,
		},
		Data: map[string]string{
			"config.yaml":   configData,
			"tools_db.json": toolsData,
		},
	}

	if err := controllerutil.SetControllerReference(sr, cm, r.Scheme); err != nil {
		return err
	}

	found := &corev1.ConfigMap{}
	err = r.Get(ctx, types.NamespacedName{Name: cm.Name, Namespace: cm.Namespace}, found)
	if err != nil && errors.IsNotFound(err) {
		return r.Create(ctx, cm)
	} else if err != nil {
		return err
	}

	// Update if data changed (with retry on conflict)
	if !reflect.DeepEqual(found.Data, cm.Data) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// Refetch to get latest resource version
			if err := r.Get(ctx, types.NamespacedName{Name: cm.Name, Namespace: cm.Namespace}, found); err != nil {
				return err
			}
			found.Data = cm.Data
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcilePVC(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	if sr.Spec.Persistence.Enabled == nil || !*sr.Spec.Persistence.Enabled {
		return nil
	}

	// Use existing claim if specified
	if sr.Spec.Persistence.ExistingClaim != "" {
		return nil
	}

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

func (r *SemanticRouterReconciler) reconcileDeployment(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	deployment := r.generateDeployment(sr)
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

	// Update if spec changed (with retry on conflict)
	if !reflect.DeepEqual(found.Spec, deployment.Spec) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// Refetch to get latest resource version
			if err := r.Get(ctx, types.NamespacedName{Name: deployment.Name, Namespace: deployment.Namespace}, found); err != nil {
				return err
			}
			found.Spec = deployment.Spec
			return r.Update(ctx, found)
		})
	}

	return nil
}

func (r *SemanticRouterReconciler) reconcileService(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	svc := r.generateService(sr)
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

	// Update if spec changed (preserving ClusterIP, with retry on conflict)
	if !reflect.DeepEqual(found.Spec.Ports, svc.Spec.Ports) ||
		found.Spec.Type != svc.Spec.Type ||
		!reflect.DeepEqual(found.Spec.Selector, svc.Spec.Selector) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// Refetch to get latest resource version
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
		// Delete HPA if it exists but autoscaling is disabled
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

	// Update if spec changed (with retry on conflict)
	if !reflect.DeepEqual(found.Spec, hpa.Spec) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// Refetch to get latest resource version
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
		// Delete Ingress if it exists but ingress is disabled
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

	// Update if spec changed (with retry on conflict)
	if !reflect.DeepEqual(found.Spec, ing.Spec) {
		return retry.RetryOnConflict(retry.DefaultRetry, func() error {
			// Refetch to get latest resource version
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
		// Refetch the SemanticRouter to get the latest resource version
		current := &vllmv1alpha1.SemanticRouter{}
		if err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, current); err != nil {
			return err
		}

		// Fetch deployment status
		deployment := &appsv1.Deployment{}
		err := r.Get(ctx, types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}, deployment)
		if err != nil {
			if errors.IsNotFound(err) {
				// Deployment not created yet - set status to Pending
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
			// Other errors should be returned
			return err
		}

		current.Status.Replicas = deployment.Status.Replicas
		current.Status.ReadyReplicas = deployment.Status.ReadyReplicas
		current.Status.ObservedGeneration = current.Generation

		// Determine phase
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

func (r *SemanticRouterReconciler) generateConfigYAML(sr *vllmv1alpha1.SemanticRouter) (string, error) {
	config := map[string]interface{}{}

	if sr.Spec.Config.BertModel != nil {
		config["bert_model"] = r.convertToConfigMap(sr.Spec.Config.BertModel)
	}
	if sr.Spec.Config.SemanticCache != nil {
		config["semantic_cache"] = r.convertToConfigMap(sr.Spec.Config.SemanticCache)
	}
	if sr.Spec.Config.Tools != nil {
		config["tools"] = r.convertToConfigMap(sr.Spec.Config.Tools)
	}
	if sr.Spec.Config.PromptGuard != nil {
		config["prompt_guard"] = r.convertToConfigMap(sr.Spec.Config.PromptGuard)
	}
	if sr.Spec.Config.Classifier != nil {
		config["classifier"] = r.convertToConfigMap(sr.Spec.Config.Classifier)
	}
	if sr.Spec.Config.ReasoningFamilies != nil {
		config["reasoning_families"] = r.convertToConfigMap(sr.Spec.Config.ReasoningFamilies)
	}
	if sr.Spec.Config.DefaultReasoningEffort != "" {
		config["default_reasoning_effort"] = sr.Spec.Config.DefaultReasoningEffort
	}
	if sr.Spec.Config.API != nil {
		config["api"] = r.convertToConfigMap(sr.Spec.Config.API)
	}
	if sr.Spec.Config.Observability != nil {
		config["observability"] = r.convertToConfigMap(sr.Spec.Config.Observability)
	}

	data, err := yaml.Marshal(config)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func (r *SemanticRouterReconciler) generateToolsJSON(sr *vllmv1alpha1.SemanticRouter) (string, error) {
	if sr.Spec.ToolsDb == nil {
		return "[]", nil
	}

	data, err := json.Marshal(sr.Spec.ToolsDb)
	if err != nil {
		return "", err
	}

	return string(data), nil
}

func (r *SemanticRouterReconciler) generatePVC(sr *vllmv1alpha1.SemanticRouter) (*corev1.PersistentVolumeClaim, error) {
	storageClass := sr.Spec.Persistence.StorageClassName
	size := sr.Spec.Persistence.Size
	if size == "" {
		size = DefaultPVCSize
	}

	quantity, err := r.parseQuantity(size)
	if err != nil {
		return nil, fmt.Errorf("invalid storage size %q: %w", size, err)
	}

	pvc := &corev1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:        sr.Name + "-models",
			Namespace:   sr.Namespace,
			Annotations: sr.Spec.Persistence.Annotations,
		},
		Spec: corev1.PersistentVolumeClaimSpec{
			AccessModes: []corev1.PersistentVolumeAccessMode{sr.Spec.Persistence.AccessMode},
			Resources: corev1.VolumeResourceRequirements{
				Requests: corev1.ResourceList{
					corev1.ResourceStorage: quantity,
				},
			},
		},
	}

	if storageClass != "" {
		pvc.Spec.StorageClassName = &storageClass
	}

	return pvc, nil
}

func (r *SemanticRouterReconciler) generateDeployment(sr *vllmv1alpha1.SemanticRouter) *appsv1.Deployment {
	replicas := DefaultReplicas
	if sr.Spec.Replicas != nil {
		replicas = *sr.Spec.Replicas
	}

	labels := map[string]string{
		"app.kubernetes.io/name":     "semantic-router",
		"app.kubernetes.io/instance": sr.Name,
	}

	saName := sr.Name
	if sr.Spec.ServiceAccount.Name != "" {
		saName = sr.Spec.ServiceAccount.Name
	}

	// Determine security context based on platform
	podSecurityContext := r.getPodSecurityContext(sr)

	deployment := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: sr.Spec.PodAnnotations,
				},
				Spec: corev1.PodSpec{
					ServiceAccountName: saName,
					SecurityContext:    podSecurityContext,
					ImagePullSecrets:   sr.Spec.ImagePullSecrets,
					Containers:         r.generateContainers(sr),
					Volumes:            r.generateVolumes(sr),
					NodeSelector:       sr.Spec.NodeSelector,
					Tolerations:        sr.Spec.Tolerations,
					Affinity:           sr.Spec.Affinity,
				},
			},
		},
	}

	return deployment
}

// getPodSecurityContext returns the appropriate pod security context based on platform
func (r *SemanticRouterReconciler) getPodSecurityContext(sr *vllmv1alpha1.SemanticRouter) *corev1.PodSecurityContext {
	// If user explicitly set a pod security context, use it
	if sr.Spec.PodSecurityContext != nil {
		return sr.Spec.PodSecurityContext
	}

	// On OpenShift, don't set runAsUser/runAsGroup/fsGroup - let SCC assign them
	if r.isOpenShift != nil && *r.isOpenShift {
		return &corev1.PodSecurityContext{
			// Only set capabilities and privilege escalation restrictions
			// OpenShift SCCs will handle UID/GID assignment
		}
	}

	// On standard Kubernetes, use secure defaults
	runAsNonRoot := DefaultRunAsNonRoot
	runAsUser := DefaultRunAsUser
	fsGroup := DefaultFSGroup

	return &corev1.PodSecurityContext{
		RunAsNonRoot: &runAsNonRoot,
		RunAsUser:    &runAsUser,
		FSGroup:      &fsGroup,
	}
}

// getContainerSecurityContext returns the appropriate container security context based on platform
func (r *SemanticRouterReconciler) getContainerSecurityContext(sr *vllmv1alpha1.SemanticRouter) *corev1.SecurityContext {
	// If user explicitly set a container security context, use it
	if sr.Spec.SecurityContext != nil {
		return sr.Spec.SecurityContext
	}

	// Common security settings for both platforms
	allowPrivilegeEscalation := DefaultAllowPrivEsc

	securityContext := &corev1.SecurityContext{
		AllowPrivilegeEscalation: &allowPrivilegeEscalation,
		Capabilities: &corev1.Capabilities{
			Drop: []corev1.Capability{"ALL"},
		},
	}

	// On OpenShift, don't set runAsUser - let SCC assign it
	if r.isOpenShift != nil && *r.isOpenShift {
		return securityContext
	}

	// On standard Kubernetes, set runAsUser and runAsNonRoot
	runAsNonRoot := DefaultRunAsNonRoot
	runAsUser := DefaultRunAsUser

	securityContext.RunAsNonRoot = &runAsNonRoot
	securityContext.RunAsUser = &runAsUser

	return securityContext
}

func (r *SemanticRouterReconciler) generateContainers(sr *vllmv1alpha1.SemanticRouter) []corev1.Container {
	image := DefaultImage
	if sr.Spec.Image.Repository != "" {
		image = sr.Spec.Image.Repository
		if sr.Spec.Image.Tag != "" {
			image = image + ":" + sr.Spec.Image.Tag
		}
	}
	if sr.Spec.Image.ImageRegistry != "" {
		image = sr.Spec.Image.ImageRegistry + "/" + image
	}

	pullPolicy := corev1.PullIfNotPresent
	if sr.Spec.Image.PullPolicy != "" {
		pullPolicy = sr.Spec.Image.PullPolicy
	}

	container := corev1.Container{
		Name:            "semantic-router",
		Image:           image,
		ImagePullPolicy: pullPolicy,
		Args:            sr.Spec.Args,
		SecurityContext: r.getContainerSecurityContext(sr),
		Ports: []corev1.ContainerPort{
			{
				Name:          "grpc",
				ContainerPort: DefaultGRPCPort,
				Protocol:      corev1.ProtocolTCP,
			},
			{
				Name:          "metrics",
				ContainerPort: DefaultMetricsPort,
				Protocol:      corev1.ProtocolTCP,
			},
			{
				Name:          "api",
				ContainerPort: DefaultAPIPort,
				Protocol:      corev1.ProtocolTCP,
			},
		},
		Env:          sr.Spec.Env,
		Resources:    sr.Spec.Resources,
		VolumeMounts: r.generateVolumeMounts(sr),
	}

	// Add probes
	if sr.Spec.StartupProbe != nil && (sr.Spec.StartupProbe.Enabled == nil || *sr.Spec.StartupProbe.Enabled) {
		container.StartupProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(int(DefaultGRPCPort)),
				},
			},
			PeriodSeconds:    r.getInt32OrDefault(sr.Spec.StartupProbe.PeriodSeconds, DefaultStartupProbePeriod),
			TimeoutSeconds:   r.getInt32OrDefault(sr.Spec.StartupProbe.TimeoutSeconds, DefaultStartupProbeTimeout),
			FailureThreshold: r.getInt32OrDefault(sr.Spec.StartupProbe.FailureThreshold, DefaultStartupProbeFailureThreshold),
		}
	}

	if sr.Spec.LivenessProbe != nil && (sr.Spec.LivenessProbe.Enabled == nil || *sr.Spec.LivenessProbe.Enabled) {
		container.LivenessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(int(DefaultGRPCPort)),
				},
			},
			InitialDelaySeconds: r.getInt32OrDefault(sr.Spec.LivenessProbe.InitialDelaySeconds, DefaultLivenessProbeInitialDelay),
			PeriodSeconds:       r.getInt32OrDefault(sr.Spec.LivenessProbe.PeriodSeconds, DefaultLivenessProbePeriod),
			TimeoutSeconds:      r.getInt32OrDefault(sr.Spec.LivenessProbe.TimeoutSeconds, DefaultLivenessProbeTimeout),
			FailureThreshold:    r.getInt32OrDefault(sr.Spec.LivenessProbe.FailureThreshold, DefaultLivenessProbeFailureThreshold),
		}
	}

	if sr.Spec.ReadinessProbe != nil && (sr.Spec.ReadinessProbe.Enabled == nil || *sr.Spec.ReadinessProbe.Enabled) {
		container.ReadinessProbe = &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(int(DefaultGRPCPort)),
				},
			},
			InitialDelaySeconds: r.getInt32OrDefault(sr.Spec.ReadinessProbe.InitialDelaySeconds, DefaultReadinessProbeInitialDelay),
			PeriodSeconds:       r.getInt32OrDefault(sr.Spec.ReadinessProbe.PeriodSeconds, DefaultReadinessProbePeriod),
			TimeoutSeconds:      r.getInt32OrDefault(sr.Spec.ReadinessProbe.TimeoutSeconds, DefaultReadinessProbeTimeout),
			FailureThreshold:    r.getInt32OrDefault(sr.Spec.ReadinessProbe.FailureThreshold, DefaultReadinessProbeFailureThreshold),
		}
	}

	return []corev1.Container{container}
}

func (r *SemanticRouterReconciler) generateVolumes(sr *vllmv1alpha1.SemanticRouter) []corev1.Volume {
	volumes := []corev1.Volume{
		{
			Name: "config-volume",
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: sr.Name + "-config",
					},
				},
			},
		},
		{
			Name: "cache-volume",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
	}

	// Always create a models volume - either PVC (if persistence enabled) or emptyDir (if disabled)
	if sr.Spec.Persistence.Enabled != nil && *sr.Spec.Persistence.Enabled {
		pvcName := sr.Name + "-models"
		if sr.Spec.Persistence.ExistingClaim != "" {
			pvcName = sr.Spec.Persistence.ExistingClaim
		}

		volumes = append(volumes, corev1.Volume{
			Name: "models-volume",
			VolumeSource: corev1.VolumeSource{
				PersistentVolumeClaim: &corev1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvcName,
				},
			},
		})
	} else {
		// Use emptyDir for models when persistence is disabled
		volumes = append(volumes, corev1.Volume{
			Name: "models-volume",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		})
	}

	return volumes
}

func (r *SemanticRouterReconciler) generateVolumeMounts(sr *vllmv1alpha1.SemanticRouter) []corev1.VolumeMount {
	mounts := []corev1.VolumeMount{
		{
			Name:      "config-volume",
			MountPath: "/app/config",
			ReadOnly:  true,
		},
		{
			Name:      "cache-volume",
			MountPath: "/.cache",
		},
		// Always mount models volume (backed by PVC or emptyDir depending on persistence setting)
		{
			Name:      "models-volume",
			MountPath: "/app/models",
		},
	}

	return mounts
}

func (r *SemanticRouterReconciler) generateService(sr *vllmv1alpha1.SemanticRouter) *corev1.Service {
	labels := map[string]string{
		"app.kubernetes.io/name":     "semantic-router",
		"app.kubernetes.io/instance": sr.Name,
	}

	serviceType := corev1.ServiceTypeClusterIP
	if sr.Spec.Service.Type != "" {
		serviceType = sr.Spec.Service.Type
	}

	ports := []corev1.ServicePort{
		{
			Name:       "grpc",
			Port:       r.getInt32OrDefault(&sr.Spec.Service.GRPC.Port, DefaultGRPCPort),
			TargetPort: intstr.FromInt(int(r.getInt32OrDefault(&sr.Spec.Service.GRPC.TargetPort, DefaultGRPCPort))),
			Protocol:   corev1.ProtocolTCP,
		},
		{
			Name:       "api",
			Port:       r.getInt32OrDefault(&sr.Spec.Service.API.Port, DefaultAPIPort),
			TargetPort: intstr.FromInt(int(r.getInt32OrDefault(&sr.Spec.Service.API.TargetPort, DefaultAPIPort))),
			Protocol:   corev1.ProtocolTCP,
		},
	}

	if sr.Spec.Service.Metrics.Enabled == nil || *sr.Spec.Service.Metrics.Enabled {
		metricsPort := sr.Spec.Service.Metrics.Port
		if metricsPort == 0 {
			metricsPort = DefaultMetricsPort
		}
		metricsTargetPort := sr.Spec.Service.Metrics.TargetPort
		if metricsTargetPort == 0 {
			metricsTargetPort = DefaultMetricsPort
		}
		ports = append(ports, corev1.ServicePort{
			Name:       "metrics",
			Port:       metricsPort,
			TargetPort: intstr.FromInt(int(metricsTargetPort)),
			Protocol:   corev1.ProtocolTCP,
		})
	}

	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
		},
		Spec: corev1.ServiceSpec{
			Type:     serviceType,
			Ports:    ports,
			Selector: labels,
		},
	}
}

func (r *SemanticRouterReconciler) generateHPA(sr *vllmv1alpha1.SemanticRouter) *autoscalingv2.HorizontalPodAutoscaler {
	minReplicas := DefaultHPAMinReplicas
	if sr.Spec.Autoscaling.MinReplicas != nil {
		minReplicas = *sr.Spec.Autoscaling.MinReplicas
	}

	maxReplicas := DefaultHPAMaxReplicas
	if sr.Spec.Autoscaling.MaxReplicas != nil {
		maxReplicas = *sr.Spec.Autoscaling.MaxReplicas
	}

	metrics := []autoscalingv2.MetricSpec{}
	if sr.Spec.Autoscaling.TargetCPUUtilizationPercentage != nil {
		metrics = append(metrics, autoscalingv2.MetricSpec{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name: corev1.ResourceCPU,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: sr.Spec.Autoscaling.TargetCPUUtilizationPercentage,
				},
			},
		})
	}

	if sr.Spec.Autoscaling.TargetMemoryUtilizationPercentage != nil {
		metrics = append(metrics, autoscalingv2.MetricSpec{
			Type: autoscalingv2.ResourceMetricSourceType,
			Resource: &autoscalingv2.ResourceMetricSource{
				Name: corev1.ResourceMemory,
				Target: autoscalingv2.MetricTarget{
					Type:               autoscalingv2.UtilizationMetricType,
					AverageUtilization: sr.Spec.Autoscaling.TargetMemoryUtilizationPercentage,
				},
			},
		})
	}

	return &autoscalingv2.HorizontalPodAutoscaler{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
		},
		Spec: autoscalingv2.HorizontalPodAutoscalerSpec{
			ScaleTargetRef: autoscalingv2.CrossVersionObjectReference{
				APIVersion: "apps/v1",
				Kind:       "Deployment",
				Name:       sr.Name,
			},
			MinReplicas: &minReplicas,
			MaxReplicas: maxReplicas,
			Metrics:     metrics,
		},
	}
}

func (r *SemanticRouterReconciler) generateIngress(sr *vllmv1alpha1.SemanticRouter) *networkingv1.Ingress {
	pathType := networkingv1.PathTypePrefix

	var rules []networkingv1.IngressRule
	for _, host := range sr.Spec.Ingress.Hosts {
		var paths []networkingv1.HTTPIngressPath
		for _, path := range host.Paths {
			pt := pathType
			if path.PathType != "" {
				switch path.PathType {
				case "Exact":
					pt = networkingv1.PathTypeExact
				case "Prefix":
					pt = networkingv1.PathTypePrefix
				}
			}

			paths = append(paths, networkingv1.HTTPIngressPath{
				Path:     path.Path,
				PathType: &pt,
				Backend: networkingv1.IngressBackend{
					Service: &networkingv1.IngressServiceBackend{
						Name: sr.Name,
						Port: networkingv1.ServiceBackendPort{
							Number: path.ServicePort,
						},
					},
				},
			})
		}

		rules = append(rules, networkingv1.IngressRule{
			Host: host.Host,
			IngressRuleValue: networkingv1.IngressRuleValue{
				HTTP: &networkingv1.HTTPIngressRuleValue{
					Paths: paths,
				},
			},
		})
	}

	var tls []networkingv1.IngressTLS
	for _, t := range sr.Spec.Ingress.TLS {
		tls = append(tls, networkingv1.IngressTLS{
			Hosts:      t.Hosts,
			SecretName: t.SecretName,
		})
	}

	ingress := &networkingv1.Ingress{
		ObjectMeta: metav1.ObjectMeta{
			Name:        sr.Name,
			Namespace:   sr.Namespace,
			Annotations: sr.Spec.Ingress.Annotations,
		},
		Spec: networkingv1.IngressSpec{
			Rules: rules,
			TLS:   tls,
		},
	}

	if sr.Spec.Ingress.ClassName != "" {
		ingress.Spec.IngressClassName = &sr.Spec.Ingress.ClassName
	}

	return ingress
}

// finalizeSemanticRouter performs cleanup before the SemanticRouter is deleted
func (r *SemanticRouterReconciler) finalizeSemanticRouter(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	logger := log.FromContext(ctx)
	logger.Info("Finalizing SemanticRouter", "name", sr.Name, "namespace", sr.Namespace)

	// Only cleanup PVCs if they were created by the operator (not existing claims)
	if sr.Spec.Persistence.Enabled != nil && *sr.Spec.Persistence.Enabled &&
		sr.Spec.Persistence.ExistingClaim == "" {
		pvcName := sr.Name + "-models"
		pvc := &corev1.PersistentVolumeClaim{}
		err := r.Get(ctx, types.NamespacedName{Name: pvcName, Namespace: sr.Namespace}, pvc)
		if err == nil {
			// PVC exists - delete it
			logger.Info("Deleting PVC", "name", pvcName)
			if err := r.Delete(ctx, pvc); err != nil {
				logger.Error(err, "Failed to delete PVC", "name", pvcName)
				return err
			}
		} else if !errors.IsNotFound(err) {
			// Error other than NotFound
			return err
		}
	}

	logger.Info("Successfully finalized SemanticRouter")
	return nil
}

// SetupWithManager sets up the controller with the Manager.
func (r *SemanticRouterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&vllmv1alpha1.SemanticRouter{}).
		Owns(&appsv1.Deployment{}).
		Owns(&corev1.Service{}).
		Owns(&corev1.ServiceAccount{}).
		Owns(&corev1.ConfigMap{}).
		Owns(&corev1.PersistentVolumeClaim{}).
		Owns(&autoscalingv2.HorizontalPodAutoscaler{}).
		Owns(&networkingv1.Ingress{}).
		Complete(r)
}
