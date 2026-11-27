package controllers

import (
	"context"
	"fmt"
	"reflect"
	"strconv"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

type standaloneMigration struct {
	deployment *appsv1.Deployment
	progress   metav1.Condition
	approved   bool
}

func (r *SemanticRouterReconciler) checkStandaloneMigration(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (*standaloneMigration, error) {
	deployment := &appsv1.Deployment{}
	err := r.Get(ctx, client.ObjectKeyFromObject(sr), deployment)
	if err != nil {
		return nil, client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(deployment, sr) {
		return nil, nil
	}

	var legacy *appsv1.ReplicaSet
	migrating := false
	if sr.Spec.Gateway == nil || sr.Spec.Gateway.ExistingRef == nil {
		migrating = hasEnvoySidecar(deployment.Spec.Template)
		if !migrating && (deployment.Generation != deployment.Status.ObservedGeneration || deployment.Status.UpdatedReplicas < deployment.Status.Replicas) {
			// An earlier Operator may already have replaced the template while
			// its legacy ReplicaSet is the only one still serving.
			legacy, err = r.activeEnvoyReplicaSet(ctx, sr, deployment)
			if err != nil {
				return nil, err
			}
			migrating = legacy != nil
		}
	}
	if !migrating {
		return nil, r.removeMigrationProbe(ctx, sr)
	}
	if err = r.keepLegacyInferencePort(ctx, sr); err != nil {
		return nil, err
	}

	migration := &standaloneMigration{deployment: deployment}
	migration.progress, migration.approved, err = r.checkMigrationImage(ctx, sr)
	if err != nil {
		return nil, err
	}
	if migration.progress.Reason == incompatibleRouterImage && legacy != nil && legacy.Status.ReadyReplicas > 0 {
		// Stop the failed rollout using the exact template of the serving
		// ReplicaSet. Deployment owns the pod-template-hash selector label.
		base := deployment.DeepCopy()
		deployment.Spec.Template = *legacy.Spec.Template.DeepCopy()
		delete(deployment.Spec.Template.Labels, appsv1.DefaultDeploymentUniqueLabelKey)
		if err = r.Patch(ctx, deployment, client.MergeFromWithOptions(base, client.MergeFromWithOptimisticLock{})); err != nil {
			return nil, err
		}
	}
	return migration, nil
}

func hasEnvoySidecar(template corev1.PodTemplateSpec) bool {
	for _, c := range template.Spec.Containers {
		if c.Name == "envoy-proxy" {
			return true
		}
	}
	return false
}

func (r *SemanticRouterReconciler) activeEnvoyReplicaSet(ctx context.Context, sr *vllmv1alpha1.SemanticRouter, dep *appsv1.Deployment) (*appsv1.ReplicaSet, error) {
	sets := &appsv1.ReplicaSetList{}
	if err := r.List(ctx, sets, client.InNamespace(sr.Namespace), client.MatchingLabels(semanticRouterLabels(sr))); err != nil {
		return nil, err
	}
	var serving *appsv1.ReplicaSet
	for i := range sets.Items {
		rs := &sets.Items[i]
		if !metav1.IsControlledBy(rs, dep) || !hasEnvoySidecar(rs.Spec.Template) || rs.Status.Replicas == 0 && ptr.Deref(rs.Spec.Replicas, 0) == 0 {
			continue
		}
		if serving == nil || rs.Status.ReadyReplicas > serving.Status.ReadyReplicas {
			serving = rs
		}
	}
	return serving, nil
}

func (r *SemanticRouterReconciler) keepLegacyInferencePort(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	svc := &corev1.Service{}
	err := r.Get(ctx, client.ObjectKeyFromObject(sr), svc)
	if errors.IsNotFound(err) {
		return r.reconcileService(ctx, sr, GatewayModeStandalone)
	}
	if err != nil {
		return err
	}
	if !metav1.IsControlledBy(svc, sr) {
		return fmt.Errorf("cannot protect inference on Service %s: not owned by SemanticRouter", svc.Name)
	}
	for i, p := range svc.Spec.Ports {
		if p.Port == DefaultListenerPort && p.TargetPort != intstr.FromInt32(DefaultListenerPort) {
			base := svc.DeepCopy()
			svc.Spec.Ports[i].TargetPort = intstr.FromInt32(DefaultListenerPort)
			return r.Patch(ctx, svc, client.MergeFromWithOptions(base, client.MergeFromWithOptimisticLock{}))
		}
	}
	return nil
}

const migrationReadinessAnnotation = "vllm.ai/standalone-migration-generation"

// Keep the migration revision's readiness check across reconciles, including
// the first reconcile after the last legacy Pod disappears. Removing it at
// that point would immediately launch another, unprobed rollout.
func protectMigrationReadiness(sr *vllmv1alpha1.SemanticRouter, desired, existing *appsv1.Deployment, migrating bool) {
	if migrating {
		// Let the standalone endpoint enter Service load balancing before
		// the Deployment can retire the serving sidecar replica.
		desired.Spec.MinReadySeconds = 10
		desired.Spec.Strategy = appsv1.DeploymentStrategy{
			Type: appsv1.RollingUpdateDeploymentStrategyType,
			RollingUpdate: &appsv1.RollingUpdateDeployment{
				MaxUnavailable: ptr.To(intstr.FromInt(0)), MaxSurge: ptr.To(intstr.FromInt(1)),
			},
		}
	}
	generation := strconv.FormatInt(sr.Generation, 10)
	if !migrating && existing.Spec.Template.Annotations[migrationReadinessAnnotation] != generation {
		return
	}
	router := &desired.Spec.Template.Spec.Containers[0]
	if router.ReadinessProbe == nil {
		router.ReadinessProbe = &corev1.Probe{ProbeHandler: routerProbeHandler(GatewayModeStandalone, "/ready", nil)}
		desired.Spec.Template.Annotations[migrationReadinessAnnotation] = generation
	}
}

func (r *SemanticRouterReconciler) reportMigration(ctx context.Context, sr, base *vllmv1alpha1.SemanticRouter, m *standaloneMigration) error {
	dep := m.deployment
	sr.Status.ObservedGeneration = sr.Generation
	sr.Status.Replicas = dep.Status.Replicas
	sr.Status.ReadyReplicas = dep.Status.ReadyReplicas
	available := metav1.Condition{
		Type: typeAvailableSemanticRouter, ObservedGeneration: sr.Generation,
		Status: metav1.ConditionFalse, Reason: "Pending", Message: "No replicas are ready",
	}
	if dep.Status.ReadyReplicas > 0 {
		available.Status, available.Reason, available.Message = metav1.ConditionTrue, "Serving", "Ready replicas are serving inference"
	}
	sr.Status.Phase = "Progressing"
	if m.progress.Status == metav1.ConditionFalse {
		sr.Status.Phase = "Blocked"
	}
	m.progress.Type, m.progress.ObservedGeneration = typeProgressingSemanticRouter, sr.Generation
	meta.SetStatusCondition(&sr.Status.Conditions, available)
	meta.SetStatusCondition(&sr.Status.Conditions, m.progress)
	if reflect.DeepEqual(sr.Status, base.Status) {
		return nil
	}
	return r.Status().Patch(ctx, sr, client.MergeFromWithOptions(base, client.MergeFromWithOptimisticLock{}))
}
