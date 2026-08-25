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
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"reflect"
	"strings"

	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	policyv1 "k8s.io/api/policy/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const (
	migrationStatePending   = "Pending"
	migrationStateRunning   = "Running"
	migrationStateSucceeded = "Succeeded"
	migrationStateFailed    = "Failed"
	migrationJobContract    = "v1"
)

func (r *SemanticRouterReconciler) reconcileMigrationJob(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) (*vllmv1alpha1.MigrationStatus, error) {
	job, err := r.generateMigrationJob(sr, bootstrap)
	if err != nil {
		return nil, err
	}
	if err := controllerutil.SetControllerReference(sr, job, r.Scheme); err != nil {
		return nil, err
	}

	found := &batchv1.Job{}
	err = r.Get(ctx, client.ObjectKeyFromObject(job), found)
	if err != nil && client.IgnoreNotFound(err) != nil {
		return nil, err
	}
	if err != nil {
		if err := r.Create(ctx, job); err != nil {
			return nil, err
		}
		return &vllmv1alpha1.MigrationStatus{JobName: job.Name, State: migrationStatePending}, nil
	}
	if err := ensureControlledBy(found, sr, "Management schema migration Job"); err != nil {
		return nil, err
	}

	for _, condition := range found.Status.Conditions {
		if condition.Type == batchv1.JobFailed && condition.Status == corev1.ConditionTrue {
			return &vllmv1alpha1.MigrationStatus{JobName: found.Name, State: migrationStateFailed},
				fmt.Errorf("Management schema migration Job %s failed", found.Name)
		}
		if condition.Type == batchv1.JobComplete && condition.Status == corev1.ConditionTrue {
			if err := r.deleteStaleMigrationJobs(ctx, sr, found.Name); err != nil {
				return &vllmv1alpha1.MigrationStatus{JobName: found.Name, State: migrationStateSucceeded}, err
			}
			return &vllmv1alpha1.MigrationStatus{JobName: found.Name, State: migrationStateSucceeded}, nil
		}
	}
	state := migrationStatePending
	if found.Status.Active > 0 {
		state = migrationStateRunning
	}
	return &vllmv1alpha1.MigrationStatus{JobName: found.Name, State: state}, nil
}

func (r *SemanticRouterReconciler) deleteStaleMigrationJobs(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
	currentName string,
) error {
	jobs := &batchv1.JobList{}
	if err := r.List(
		ctx,
		jobs,
		client.InNamespace(sr.Namespace),
		client.MatchingLabels{
			"app.kubernetes.io/instance":  sr.Name,
			"app.kubernetes.io/component": "management-migration",
		},
	); err != nil {
		return fmt.Errorf("list Management schema migration Jobs: %w", err)
	}
	for index := range jobs.Items {
		job := &jobs.Items[index]
		if job.Name == currentName || !metav1.IsControlledBy(job, sr) {
			continue
		}
		if err := r.Delete(ctx, job); err != nil && client.IgnoreNotFound(err) != nil {
			return fmt.Errorf("delete stale Management schema migration Job %s: %w", job.Name, err)
		}
	}
	return nil
}

func (r *SemanticRouterReconciler) generateMigrationJob(
	sr *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) (*batchv1.Job, error) {
	backoffLimit := int32(6)
	activeDeadlineSeconds := int64(600)
	args := []string{"--timeout", "5m"}
	if bootstrap.PostgresDSNEnv != "" {
		args = append([]string{"--dsn-env", bootstrap.PostgresDSNEnv}, args...)
	} else {
		args = append([]string{"--dsn-file", bootstrap.PostgresDSNFile}, args...)
	}

	labels := semanticRouterLabels(sr)
	labels["app.kubernetes.io/component"] = "management-migration"
	jobName, err := migrationJobName(sr, bootstrap)
	if err != nil {
		return nil, err
	}
	return &batchv1.Job{
		ObjectMeta: metav1.ObjectMeta{
			Name:      jobName,
			Namespace: sr.Namespace,
			Labels:    labels,
		},
		Spec: batchv1.JobSpec{
			BackoffLimit:          &backoffLimit,
			ActiveDeadlineSeconds: &activeDeadlineSeconds,
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: labels},
				Spec: corev1.PodSpec{
					RestartPolicy:      corev1.RestartPolicyNever,
					ServiceAccountName: serviceAccountName(sr),
					SecurityContext:    r.getPodSecurityContext(sr),
					ImagePullSecrets:   sr.Spec.ImagePullSecrets,
					NodeSelector:       sr.Spec.NodeSelector,
					Tolerations:        sr.Spec.Tolerations,
					Affinity:           sr.Spec.Affinity,
					Containers: []corev1.Container{{
						Name:            "management-migrate",
						Image:           semanticRouterImage(sr),
						ImagePullPolicy: imagePullPolicy(sr),
						Command:         []string{"/usr/local/bin/management-migrate"},
						Args:            args,
						Env:             sr.Spec.Env,
						EnvFrom:         sr.Spec.EnvFrom,
						SecurityContext: r.getContainerSecurityContext(sr),
						VolumeMounts:    sr.Spec.VolumeMounts,
					}},
					Volumes: sr.Spec.Volumes,
				},
			},
		},
	}, nil
}

func migrationJobName(
	sr *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) (string, error) {
	revisionInput := struct {
		Contract           string
		BootstrapRevision  string
		Image              string
		ImagePullPolicy    corev1.PullPolicy
		ImagePullSecrets   []corev1.LocalObjectReference
		ServiceAccountName string
		Env                []corev1.EnvVar
		EnvFrom            []corev1.EnvFromSource
		Volumes            []corev1.Volume
		VolumeMounts       []corev1.VolumeMount
		NodeSelector       map[string]string
		Tolerations        []corev1.Toleration
		Affinity           *corev1.Affinity
		PodSecurityContext *corev1.PodSecurityContext
		SecurityContext    *corev1.SecurityContext
	}{
		Contract:           migrationJobContract,
		BootstrapRevision:  bootstrap.Revision,
		Image:              semanticRouterImage(sr),
		ImagePullPolicy:    imagePullPolicy(sr),
		ImagePullSecrets:   sr.Spec.ImagePullSecrets,
		ServiceAccountName: serviceAccountName(sr),
		Env:                sr.Spec.Env,
		EnvFrom:            sr.Spec.EnvFrom,
		Volumes:            sr.Spec.Volumes,
		VolumeMounts:       sr.Spec.VolumeMounts,
		NodeSelector:       sr.Spec.NodeSelector,
		Tolerations:        sr.Spec.Tolerations,
		Affinity:           sr.Spec.Affinity,
		PodSecurityContext: sr.Spec.PodSecurityContext,
		SecurityContext:    sr.Spec.SecurityContext,
	}
	payload, err := json.Marshal(revisionInput)
	if err != nil {
		return "", fmt.Errorf("encode Management schema migration inputs: %w", err)
	}
	digest := sha256.Sum256(payload)
	suffix := fmt.Sprintf("%x", digest[:6])
	base := sr.Name
	const reserved = len("-management-migrate-") + 12
	if len(base) > 63-reserved {
		base = strings.TrimRight(base[:63-reserved], "-")
	}
	return base + "-management-migrate-" + suffix, nil
}

func serviceAccountName(sr *vllmv1alpha1.SemanticRouter) string {
	if sr.Spec.ServiceAccount.Name != "" {
		return sr.Spec.ServiceAccount.Name
	}
	if !serviceAccountShouldCreate(sr) {
		return ""
	}
	return sr.Name
}

func serviceAccountShouldCreate(sr *vllmv1alpha1.SemanticRouter) bool {
	return sr.Spec.ServiceAccount.Create == nil || *sr.Spec.ServiceAccount.Create
}

func imagePullPolicy(sr *vllmv1alpha1.SemanticRouter) corev1.PullPolicy {
	if sr.Spec.Image.PullPolicy != "" {
		return sr.Spec.Image.PullPolicy
	}
	return corev1.PullIfNotPresent
}

func (r *SemanticRouterReconciler) reconcilePodDisruptionBudget(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
	defaultEnabled bool,
) error {
	key := types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}
	if !configuredOrDefault(sr.Spec.PodDisruptionBudget.Enabled, defaultEnabled) {
		return r.deleteOwnedIfPresent(ctx, sr, key, &policyv1.PodDisruptionBudget{})
	}
	desired := generatePodDisruptionBudget(sr)
	if err := controllerutil.SetControllerReference(sr, desired, r.Scheme); err != nil {
		return err
	}
	found := &policyv1.PodDisruptionBudget{}
	if err := r.Get(ctx, key, found); err != nil {
		if client.IgnoreNotFound(err) != nil {
			return err
		}
		return r.Create(ctx, desired)
	}
	if err := ensureControlledBy(found, sr, "PodDisruptionBudget"); err != nil {
		return err
	}
	if reflect.DeepEqual(found.Spec, desired.Spec) && reflect.DeepEqual(found.Labels, desired.Labels) {
		return nil
	}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := r.Get(ctx, key, found); err != nil {
			return err
		}
		if err := ensureControlledBy(found, sr, "PodDisruptionBudget"); err != nil {
			return err
		}
		found.Spec = desired.Spec
		found.Labels = desired.Labels
		return r.Update(ctx, found)
	})
}

func generatePodDisruptionBudget(sr *vllmv1alpha1.SemanticRouter) *policyv1.PodDisruptionBudget {
	minimum := int32(1)
	if sr.Spec.PodDisruptionBudget.MinAvailable != nil {
		minimum = *sr.Spec.PodDisruptionBudget.MinAvailable
	}
	minAvailable := intstr.FromInt(int(minimum))
	return &policyv1.PodDisruptionBudget{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
			Labels:    semanticRouterLabels(sr),
		},
		Spec: policyv1.PodDisruptionBudgetSpec{
			MinAvailable: &minAvailable,
			Selector:     &metav1.LabelSelector{MatchLabels: semanticRouterLabels(sr)},
		},
	}
}

func (r *SemanticRouterReconciler) reconcileNetworkPolicy(
	ctx context.Context,
	sr *vllmv1alpha1.SemanticRouter,
	gatewayMode string,
	bootstrap bootstrapDeploymentContract,
) error {
	key := types.NamespacedName{Name: sr.Name, Namespace: sr.Namespace}
	if !configuredOrDefault(sr.Spec.NetworkPolicy.Enabled, bootstrap.enablesAvailabilityDefaults()) {
		return r.deleteOwnedIfPresent(ctx, sr, key, &networkingv1.NetworkPolicy{})
	}
	desired := generateNetworkPolicy(sr, gatewayMode, bootstrap)
	if err := controllerutil.SetControllerReference(sr, desired, r.Scheme); err != nil {
		return err
	}
	found := &networkingv1.NetworkPolicy{}
	if err := r.Get(ctx, key, found); err != nil {
		if client.IgnoreNotFound(err) != nil {
			return err
		}
		return r.Create(ctx, desired)
	}
	if err := ensureControlledBy(found, sr, "NetworkPolicy"); err != nil {
		return err
	}
	if reflect.DeepEqual(found.Spec, desired.Spec) && reflect.DeepEqual(found.Labels, desired.Labels) {
		return nil
	}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := r.Get(ctx, key, found); err != nil {
			return err
		}
		if err := ensureControlledBy(found, sr, "NetworkPolicy"); err != nil {
			return err
		}
		found.Spec = desired.Spec
		found.Labels = desired.Labels
		return r.Update(ctx, found)
	})
}

func generateNetworkPolicy(
	sr *vllmv1alpha1.SemanticRouter,
	gatewayMode string,
	bootstrap bootstrapDeploymentContract,
) *networkingv1.NetworkPolicy {
	rules := make([]networkingv1.NetworkPolicyIngressRule, 0, 4)
	if len(sr.Spec.NetworkPolicy.InferencePeers) > 0 {
		ports := []networkingv1.NetworkPolicyPort{networkPolicyPort(DefaultGRPCPort)}
		if gatewayMode == gatewayModeSidecar {
			ports = append(ports, networkPolicyPort(DefaultEnvoyPort))
		}
		if !bootstrap.usesDurableState() {
			ports = append(ports, networkPolicyPort(DefaultAPIPort))
		}
		rules = append(rules, networkingv1.NetworkPolicyIngressRule{
			From:  sr.Spec.NetworkPolicy.InferencePeers,
			Ports: ports,
		})
	}
	if bootstrap.exposesManagementAPI() {
		if len(sr.Spec.NetworkPolicy.ManagementPeers) > 0 {
			rules = append(rules, networkingv1.NetworkPolicyIngressRule{
				From:  sr.Spec.NetworkPolicy.ManagementPeers,
				Ports: []networkingv1.NetworkPolicyPort{networkPolicyPort(bootstrap.ManagementPort)},
			})
		}
	}
	if bootstrap.usesBackendDispatch() {
		rules = append(rules, networkingv1.NetworkPolicyIngressRule{
			From: []networkingv1.NetworkPolicyPeer{{
				PodSelector: &metav1.LabelSelector{MatchLabels: semanticRouterLabels(sr)},
			}},
			Ports: []networkingv1.NetworkPolicyPort{networkPolicyPort(bootstrap.BackendDispatchPort)},
		})
	}
	if len(sr.Spec.NetworkPolicy.MetricsPeers) > 0 && metricsEnabled(sr) {
		rules = append(rules, networkingv1.NetworkPolicyIngressRule{
			From:  sr.Spec.NetworkPolicy.MetricsPeers,
			Ports: []networkingv1.NetworkPolicyPort{networkPolicyPort(metricsTargetPort(sr))},
		})
	}

	return &networkingv1.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
			Labels:    semanticRouterLabels(sr),
		},
		Spec: networkingv1.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{MatchLabels: semanticRouterLabels(sr)},
			PolicyTypes: []networkingv1.PolicyType{networkingv1.PolicyTypeIngress},
			Ingress:     rules,
		},
	}
}

func networkPolicyPort(port int32) networkingv1.NetworkPolicyPort {
	protocol := corev1.ProtocolTCP
	value := intstr.FromInt(int(port))
	return networkingv1.NetworkPolicyPort{Protocol: &protocol, Port: &value}
}

func (r *SemanticRouterReconciler) deleteOwnedIfPresent(
	ctx context.Context,
	owner *vllmv1alpha1.SemanticRouter,
	key types.NamespacedName,
	object client.Object,
) error {
	if err := r.Get(ctx, key, object); err != nil {
		return client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(object, owner) {
		return nil
	}
	return r.Delete(ctx, object)
}

func ensureControlledBy(
	object client.Object,
	owner *vllmv1alpha1.SemanticRouter,
	resourceKind string,
) error {
	if metav1.IsControlledBy(object, owner) {
		return nil
	}
	return fmt.Errorf(
		"%s %s exists but is not controlled by SemanticRouter %s",
		resourceKind,
		object.GetName(),
		owner.Name,
	)
}
