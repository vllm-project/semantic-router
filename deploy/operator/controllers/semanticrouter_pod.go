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
	"strconv"
	"strings"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

func (r *SemanticRouterReconciler) generateDeployment(sr *vllmv1alpha1.SemanticRouter, gatewayMode string) *appsv1.Deployment {
	replicas := DefaultReplicas
	if sr.Spec.Replicas != nil {
		replicas = *sr.Spec.Replicas
	}

	labels := semanticRouterLabels(sr)

	saName := sr.Name
	if sr.Spec.ServiceAccount.Name != "" {
		saName = sr.Spec.ServiceAccount.Name
	}

	return &appsv1.Deployment{
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
					SecurityContext:    r.getPodSecurityContext(sr),
					ImagePullSecrets:   sr.Spec.ImagePullSecrets,
					Containers:         r.generateContainers(sr, gatewayMode),
					Volumes:            r.generateVolumes(sr),
					NodeSelector:       sr.Spec.NodeSelector,
					Tolerations:        sr.Spec.Tolerations,
					Affinity:           sr.Spec.Affinity,
				},
			},
		},
	}
}

func semanticRouterLabels(sr *vllmv1alpha1.SemanticRouter) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":     "semantic-router",
		"app.kubernetes.io/instance": sr.Name,
	}
}

func (r *SemanticRouterReconciler) getPodSecurityContext(sr *vllmv1alpha1.SemanticRouter) *corev1.PodSecurityContext {
	if sr.Spec.PodSecurityContext != nil {
		return sr.Spec.PodSecurityContext
	}

	if r.isOpenShift != nil && *r.isOpenShift {
		return &corev1.PodSecurityContext{}
	}

	runAsNonRoot := DefaultRunAsNonRoot
	runAsUser := DefaultRunAsUser
	fsGroup := DefaultFSGroup

	return &corev1.PodSecurityContext{
		RunAsNonRoot: &runAsNonRoot,
		RunAsUser:    &runAsUser,
		FSGroup:      &fsGroup,
	}
}

func (r *SemanticRouterReconciler) getContainerSecurityContext(sr *vllmv1alpha1.SemanticRouter) *corev1.SecurityContext {
	if sr.Spec.SecurityContext != nil {
		return sr.Spec.SecurityContext
	}

	allowPrivilegeEscalation := DefaultAllowPrivEsc
	securityContext := &corev1.SecurityContext{
		AllowPrivilegeEscalation: &allowPrivilegeEscalation,
		Capabilities: &corev1.Capabilities{
			Drop: []corev1.Capability{"ALL"},
		},
	}

	if r.isOpenShift != nil && *r.isOpenShift {
		return securityContext
	}

	runAsNonRoot := DefaultRunAsNonRoot
	runAsUser := DefaultRunAsUser
	securityContext.RunAsNonRoot = &runAsNonRoot
	securityContext.RunAsUser = &runAsUser

	return securityContext
}

func (r *SemanticRouterReconciler) generateContainers(sr *vllmv1alpha1.SemanticRouter, gatewayMode string) []corev1.Container {
	container := r.buildSemanticRouterContainer(sr, gatewayMode)
	r.applySemanticRouterProbes(&container, sr, gatewayMode)
	return []corev1.Container{container}
}

func (r *SemanticRouterReconciler) buildSemanticRouterContainer(sr *vllmv1alpha1.SemanticRouter, gatewayMode string) corev1.Container {
	pullPolicy := corev1.PullIfNotPresent
	if sr.Spec.Image.PullPolicy != "" {
		pullPolicy = sr.Spec.Image.PullPolicy
	}

	traffic := corev1.ContainerPort{Name: "grpc", ContainerPort: DefaultGRPCPort, Protocol: corev1.ProtocolTCP}
	if gatewayMode == GatewayModeStandalone {
		traffic = corev1.ContainerPort{Name: DefaultListenerName, ContainerPort: DefaultListenerPort, Protocol: corev1.ProtocolTCP}
	}

	return corev1.Container{
		Name:            "semantic-router",
		Image:           semanticRouterImage(sr),
		ImagePullPolicy: pullPolicy,
		Args:            append(routerGatewayArgs(gatewayMode), sr.Spec.Args...),
		SecurityContext: r.getContainerSecurityContext(sr),
		Ports: []corev1.ContainerPort{
			traffic,
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
}

func semanticRouterImage(sr *vllmv1alpha1.SemanticRouter) string {
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
	return image
}

// routerProbeHandler is what a probe checks. A standalone Router answers
// /health once it serves and /ready once routing can take traffic, on its
// listener. Plaintext ext_proc startup/readiness requires a serving generation;
// liveness and TLS listeners retain TCP checks.
func routerProbeHandler(gatewayMode, path string, args []string) corev1.ProbeHandler {
	if gatewayMode == GatewayModeStandalone {
		return corev1.ProbeHandler{
			HTTPGet: &corev1.HTTPGetAction{
				Path:   path,
				Port:   intstr.FromString(DefaultListenerName),
				Scheme: corev1.URISchemeHTTP,
			},
		}
	}
	if path == "/ready" && !routerUsesGRPCTLS(args) {
		return corev1.ProbeHandler{GRPC: &corev1.GRPCAction{Port: DefaultGRPCPort}}
	}
	return corev1.ProbeHandler{
		TCPSocket: &corev1.TCPSocketAction{
			Port: intstr.FromInt(int(DefaultGRPCPort)),
		},
	}
}

// Native gRPC probes on supported clusters cannot probe TLS listeners.
// Match the Router's boolean flag syntax and last-value-wins behavior.
func routerUsesGRPCTLS(args []string) bool {
	secure := false
	for _, arg := range args {
		if arg == "--" {
			break
		}
		name, value, hasValue := strings.Cut(arg, "=")
		if name != "-secure" && name != "--secure" {
			continue
		}
		secure = true
		if hasValue {
			parsed, err := strconv.ParseBool(value)
			secure = err != nil || parsed
		}
	}
	return secure
}

func (r *SemanticRouterReconciler) applySemanticRouterProbes(container *corev1.Container, sr *vllmv1alpha1.SemanticRouter, gatewayMode string) {
	if sr.Spec.StartupProbe != nil && (sr.Spec.StartupProbe.Enabled == nil || *sr.Spec.StartupProbe.Enabled) {
		container.StartupProbe = &corev1.Probe{
			ProbeHandler:     routerProbeHandler(gatewayMode, "/ready", sr.Spec.Args),
			PeriodSeconds:    r.getInt32OrDefault(sr.Spec.StartupProbe.PeriodSeconds, DefaultStartupProbePeriod),
			TimeoutSeconds:   r.getInt32OrDefault(sr.Spec.StartupProbe.TimeoutSeconds, DefaultStartupProbeTimeout),
			FailureThreshold: r.getInt32OrDefault(sr.Spec.StartupProbe.FailureThreshold, DefaultStartupProbeFailureThreshold),
		}
	}

	if sr.Spec.LivenessProbe != nil && (sr.Spec.LivenessProbe.Enabled == nil || *sr.Spec.LivenessProbe.Enabled) {
		container.LivenessProbe = &corev1.Probe{
			ProbeHandler:        routerProbeHandler(gatewayMode, "/health", sr.Spec.Args),
			InitialDelaySeconds: r.getInt32OrDefault(sr.Spec.LivenessProbe.InitialDelaySeconds, DefaultLivenessProbeInitialDelay),
			PeriodSeconds:       r.getInt32OrDefault(sr.Spec.LivenessProbe.PeriodSeconds, DefaultLivenessProbePeriod),
			TimeoutSeconds:      r.getInt32OrDefault(sr.Spec.LivenessProbe.TimeoutSeconds, DefaultLivenessProbeTimeout),
			FailureThreshold:    r.getInt32OrDefault(sr.Spec.LivenessProbe.FailureThreshold, DefaultLivenessProbeFailureThreshold),
		}
	}

	if sr.Spec.ReadinessProbe != nil && (sr.Spec.ReadinessProbe.Enabled == nil || *sr.Spec.ReadinessProbe.Enabled) {
		container.ReadinessProbe = &corev1.Probe{
			ProbeHandler:        routerProbeHandler(gatewayMode, "/ready", sr.Spec.Args),
			InitialDelaySeconds: r.getInt32OrDefault(sr.Spec.ReadinessProbe.InitialDelaySeconds, DefaultReadinessProbeInitialDelay),
			PeriodSeconds:       r.getInt32OrDefault(sr.Spec.ReadinessProbe.PeriodSeconds, DefaultReadinessProbePeriod),
			TimeoutSeconds:      r.getInt32OrDefault(sr.Spec.ReadinessProbe.TimeoutSeconds, DefaultReadinessProbeTimeout),
			FailureThreshold:    r.getInt32OrDefault(sr.Spec.ReadinessProbe.FailureThreshold, DefaultReadinessProbeFailureThreshold),
		}
	}
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
		{
			Name: "router-workdir",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
		{
			Name: "var-run",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
		{
			Name: "var-log",
			VolumeSource: corev1.VolumeSource{
				EmptyDir: &corev1.EmptyDirVolumeSource{},
			},
		},
	}

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
	return []corev1.VolumeMount{
		{
			Name:      "config-volume",
			MountPath: "/app/config",
			ReadOnly:  true,
		},
		{
			Name:      "config-volume",
			MountPath: "/app/config.yaml",
			SubPath:   "config.yaml",
			ReadOnly:  true,
		},
		{
			Name:      "cache-volume",
			MountPath: "/.cache",
		},
		{
			Name:      "router-workdir",
			MountPath: "/app/.vllm-sr",
		},
		{
			Name:      "var-run",
			MountPath: "/var/run",
		},
		{
			Name:      "var-log",
			MountPath: "/var/log",
		},
		{
			Name:      "models-volume",
			MountPath: "/app/models",
		},
	}
}
