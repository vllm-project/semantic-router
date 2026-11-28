package controllers

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const (
	incompatibleRouterImage = "IncompatibleRouterImage"
	probeSpecAnnotation     = "vllm.ai/standalone-check-spec"
)

func migrationProbeKey(sr *vllmv1alpha1.SemanticRouter) client.ObjectKey {
	return client.ObjectKey{Namespace: sr.Namespace, Name: "standalone-check-" + string(sr.UID)}
}

func (r *SemanticRouterReconciler) migrationProbe(sr *vllmv1alpha1.SemanticRouter) (*corev1.Pod, error) {
	// Use the same entrypoint, image resolution, credentials, scheduling and
	// security contexts as the Router. The flag check needs no models or GPU.
	template := r.generateDeployment(sr, GatewayModeStandalone).Spec.Template
	router := template.Spec.Containers[0]
	probe := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: migrationProbeKey(sr).Name, Namespace: sr.Namespace},
		Spec: corev1.PodSpec{
			ServiceAccountName:           template.Spec.ServiceAccountName,
			AutomountServiceAccountToken: ptr.To(false),
			SecurityContext:              template.Spec.SecurityContext,
			ImagePullSecrets:             template.Spec.ImagePullSecrets,
			NodeSelector:                 template.Spec.NodeSelector,
			Tolerations:                  template.Spec.Tolerations,
			Affinity:                     template.Spec.Affinity,
			RestartPolicy:                corev1.RestartPolicyNever,
			ActiveDeadlineSeconds:        ptr.To[int64](300),
			Containers: []corev1.Container{{
				Name:                     "check",
				Image:                    router.Image,
				ImagePullPolicy:          router.ImagePullPolicy,
				SecurityContext:          router.SecurityContext,
				Env:                      router.Env,
				TerminationMessagePolicy: corev1.TerminationMessageFallbackToLogsOnError,
				// The Go flag parser stops at -help. Test the required flags first.
				Args: append(routerGatewayArgs(GatewayModeStandalone), "-help"),
			}},
		},
	}
	spec, err := json.Marshal(probe.Spec)
	if err != nil {
		return nil, err
	}
	probe.Annotations = map[string]string{probeSpecAnnotation: fmt.Sprintf("%x", sha256.Sum256(spec))}
	return probe, controllerutil.SetControllerReference(sr, probe, r.Scheme)
}

func (r *SemanticRouterReconciler) removeMigrationProbe(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) error {
	pod := &corev1.Pod{}
	if err := r.Get(ctx, migrationProbeKey(sr), pod); err != nil {
		return client.IgnoreNotFound(err)
	}
	if !metav1.IsControlledBy(pod, sr) || !pod.DeletionTimestamp.IsZero() {
		return nil
	}
	return client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}))
}

func (r *SemanticRouterReconciler) checkMigrationImage(ctx context.Context, sr *vllmv1alpha1.SemanticRouter) (metav1.Condition, bool, error) {
	image := semanticRouterImage(sr)
	condition := metav1.Condition{Status: metav1.ConditionTrue, Reason: "CheckingRouterImage", Message: fmt.Sprintf("Checking standalone flags on image %s", image)}
	desired, err := r.migrationProbe(sr)
	if err != nil {
		return condition, false, err
	}
	pod := &corev1.Pod{}
	err = r.Get(ctx, migrationProbeKey(sr), pod)
	if apierrors.IsNotFound(err) {
		return condition, false, r.Create(ctx, desired)
	}
	if err != nil {
		return condition, false, err
	}
	if !metav1.IsControlledBy(pod, sr) {
		return condition, false, fmt.Errorf("migration check Pod %s has a different owner", pod.Name)
	}
	if !pod.DeletionTimestamp.IsZero() {
		return condition, false, nil
	}
	if pod.Annotations[probeSpecAnnotation] != desired.Annotations[probeSpecAnnotation] {
		return condition, false, client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}))
	}
	return r.migrationProbeResult(ctx, pod, image, condition)
}

func (r *SemanticRouterReconciler) migrationProbeResult(ctx context.Context, pod *corev1.Pod, image string, condition metav1.Condition) (metav1.Condition, bool, error) {
	for _, s := range pod.Status.ContainerStatuses {
		if s.Name != "check" {
			continue
		}
		if waiting := s.State.Waiting; waiting != nil && waiting.Reason != "" {
			condition.Reason = "RouterImageCheckPending"
			condition.Message = fmt.Sprintf("Pod %s: %s", pod.Name, waiting.Reason)
		}
		exit := s.State.Terminated
		if exit == nil {
			continue
		}
		if exit.ExitCode == 0 {
			condition.Reason = "RollingOutStandalone"
			condition.Message = "Router accepts standalone flags. Waiting for the standalone rollout."
			return condition, true, nil
		}
		condition.Reason = "RouterImageCheckFailed"
		condition.Message = fmt.Sprintf("Image %s check exited with code %d. Inspect Pod %s.", image, exit.ExitCode, pod.Name)
		if exit.ExitCode == 2 {
			flag, err := r.unsupportedMigrationFlag(ctx, pod, exit.Message)
			if err != nil {
				return condition, false, err
			}
			if flag != "" {
				condition.Status, condition.Reason = metav1.ConditionFalse, incompatibleRouterImage
				condition.Message = fmt.Sprintf("Image %s does not support %s. Update spec.image to a Router with standalone support.", image, flag)
				return condition, false, nil
			}
		}
	}
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodScheduled && c.Status == corev1.ConditionFalse {
			condition.Reason = "RouterImageCheckPending"
			condition.Message = fmt.Sprintf("Pod %s: %s", pod.Name, c.Reason)
		}
	}
	if pod.Status.Phase == corev1.PodFailed {
		if pod.Status.Reason != "" {
			condition.Reason = "RouterImageCheckFailed"
			condition.Message = fmt.Sprintf("Pod %s: %s", pod.Name, pod.Status.Reason)
		}
		if time.Since(pod.CreationTimestamp.Time) > 5*time.Minute {
			return condition, false, client.IgnoreNotFound(r.Delete(ctx, pod, client.Preconditions{UID: &pod.UID}))
		}
	}
	return condition, false, nil
}

func (r *SemanticRouterReconciler) unsupportedMigrationFlag(ctx context.Context, pod *corev1.Pod, message string) (string, error) {
	// On a verbose binary kubelet keeps only the tail of stderr in the
	// termination message. Go prints the offending flag before the usage text.
	find := func(output string) string {
		for _, arg := range routerGatewayArgs(GatewayModeStandalone) {
			flag, _, _ := strings.Cut(arg, "=")
			for _, line := range strings.Split(output, "\n") {
				if strings.TrimSpace(line) == "flag provided but not defined: "+flag {
					return flag
				}
			}
		}
		return ""
	}
	if flag := find(message); flag != "" {
		return flag, nil
	}
	data, err := r.probeLogs.Pods(pod.Namespace).GetLogs(pod.Name, &corev1.PodLogOptions{Container: "check", LimitBytes: ptr.To[int64](4096)}).DoRaw(ctx)
	if err != nil {
		return "", err
	}
	return find(string(data)), nil
}
