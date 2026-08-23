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
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

const managementInternalListenerEnv = "VLLM_SR_MANAGEMENT_INTERNAL_LISTENER"

func (r *SemanticRouterReconciler) generateDeployment(
	sr *vllmv1alpha1.SemanticRouter,
	gatewayMode string,
	bootstrap bootstrapDeploymentContract,
) *appsv1.Deployment {
	replicas := DefaultReplicas
	if sr.Spec.Replicas != nil {
		replicas = *sr.Spec.Replicas
	}

	labels := semanticRouterLabels(sr)

	return &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      sr.Name,
			Namespace: sr.Namespace,
			Labels:    semanticRouterLabels(sr),
		},
		Spec: appsv1.DeploymentSpec{
			Replicas: &replicas,
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels:      labels,
					Annotations: deploymentAnnotations(sr, bootstrap),
				},
				Spec: corev1.PodSpec{
					ServiceAccountName:        serviceAccountName(sr),
					SecurityContext:           r.getPodSecurityContext(sr),
					ImagePullSecrets:          sr.Spec.ImagePullSecrets,
					InitContainers:            r.generateInitContainers(),
					Containers:                r.generateContainers(sr, gatewayMode, bootstrap),
					Volumes:                   r.generateVolumes(sr, gatewayMode),
					NodeSelector:              sr.Spec.NodeSelector,
					Tolerations:               sr.Spec.Tolerations,
					Affinity:                  sr.Spec.Affinity,
					TopologySpreadConstraints: topologySpreadConstraints(sr, bootstrap.Mode),
				},
			},
		},
	}
}

func deploymentAnnotations(sr *vllmv1alpha1.SemanticRouter, bootstrap bootstrapDeploymentContract) map[string]string {
	annotations := make(map[string]string, len(sr.Spec.PodAnnotations)+1)
	for key, value := range sr.Spec.PodAnnotations {
		annotations[key] = value
	}
	annotations["vllm.ai/bootstrap-revision"] = bootstrap.Revision
	return annotations
}

func semanticRouterLabels(sr *vllmv1alpha1.SemanticRouter) map[string]string {
	return map[string]string{
		"app.kubernetes.io/name":      "semantic-router",
		"app.kubernetes.io/instance":  sr.Name,
		"app.kubernetes.io/component": "router",
	}
}

func topologySpreadConstraints(sr *vllmv1alpha1.SemanticRouter, mode string) []corev1.TopologySpreadConstraint {
	if !modeAwareEnabled(sr.Spec.TopologySpread.Enabled, mode == controlPlaneModeManaged) {
		return nil
	}
	maxSkew := sr.Spec.TopologySpread.MaxSkew
	if maxSkew == 0 {
		maxSkew = 1
	}
	topologyKey := sr.Spec.TopologySpread.TopologyKey
	if topologyKey == "" {
		topologyKey = "kubernetes.io/hostname"
	}
	whenUnsatisfiable := sr.Spec.TopologySpread.WhenUnsatisfiable
	if whenUnsatisfiable == "" {
		whenUnsatisfiable = corev1.ScheduleAnyway
	}
	return []corev1.TopologySpreadConstraint{{
		MaxSkew:           maxSkew,
		TopologyKey:       topologyKey,
		WhenUnsatisfiable: whenUnsatisfiable,
		LabelSelector:     &metav1.LabelSelector{MatchLabels: semanticRouterLabels(sr)},
	}}
}

func modeAwareEnabled(value *bool, modeDefault bool) bool {
	if value == nil {
		return modeDefault
	}
	return *value
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

func (r *SemanticRouterReconciler) generateContainers(
	sr *vllmv1alpha1.SemanticRouter,
	gatewayMode string,
	bootstrap bootstrapDeploymentContract,
) []corev1.Container {
	container := r.buildSemanticRouterContainer(sr, bootstrap)
	r.applySemanticRouterProbes(&container, sr, bootstrap)

	containers := []corev1.Container{container}
	if gatewayMode == gatewayModeSidecar {
		containers = append(containers, r.generateEnvoyContainer(sr))
	}
	return containers
}

func (r *SemanticRouterReconciler) buildSemanticRouterContainer(
	sr *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) corev1.Container {
	pullPolicy := corev1.PullIfNotPresent
	if sr.Spec.Image.PullPolicy != "" {
		pullPolicy = sr.Spec.Image.PullPolicy
	}

	ports := []corev1.ContainerPort{
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
	}
	if bootstrap.Mode == controlPlaneModeManaged {
		ports = append(ports,
			corev1.ContainerPort{Name: "management", ContainerPort: bootstrap.ManagementPort, Protocol: corev1.ProtocolTCP},
			corev1.ContainerPort{Name: backendDispatchPortName, ContainerPort: bootstrap.BackendDispatchPort, Protocol: corev1.ProtocolTCP},
		)
	} else {
		ports = append(ports, corev1.ContainerPort{Name: "api", ContainerPort: DefaultAPIPort, Protocol: corev1.ProtocolTCP})
	}

	environment := []corev1.EnvVar{{Name: "CONFIG_FILE", Value: "/app/config.yaml"}}
	if bootstrap.Mode == controlPlaneModeManaged {
		environment = append(environment, corev1.EnvVar{Name: managementInternalListenerEnv, Value: "true"})
	}

	return corev1.Container{
		Name:            "semantic-router",
		Image:           semanticRouterImage(sr),
		ImagePullPolicy: pullPolicy,
		Args:            sr.Spec.Args,
		SecurityContext: r.getContainerSecurityContext(sr),
		Ports:           ports,
		Env:             append(environment, sr.Spec.Env...),
		EnvFrom:         sr.Spec.EnvFrom,
		Resources:       sr.Spec.Resources,
		VolumeMounts:    r.generateVolumeMounts(sr),
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

func (r *SemanticRouterReconciler) applySemanticRouterProbes(
	container *corev1.Container,
	sr *vllmv1alpha1.SemanticRouter,
	bootstrap bootstrapDeploymentContract,
) {
	if probeEnabled(sr.Spec.StartupProbe) {
		container.StartupProbe = &corev1.Probe{
			ProbeHandler:     semanticRouterProbeHandler(bootstrap, "/ready"),
			PeriodSeconds:    probeValue(sr.Spec.StartupProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.PeriodSeconds }, DefaultStartupProbePeriod),
			TimeoutSeconds:   probeValue(sr.Spec.StartupProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.TimeoutSeconds }, DefaultStartupProbeTimeout),
			FailureThreshold: probeValue(sr.Spec.StartupProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.FailureThreshold }, DefaultStartupProbeFailureThreshold),
		}
	}

	if probeEnabled(sr.Spec.LivenessProbe) {
		container.LivenessProbe = &corev1.Probe{
			ProbeHandler:        semanticRouterProbeHandler(bootstrap, "/health"),
			InitialDelaySeconds: probeValue(sr.Spec.LivenessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.InitialDelaySeconds }, DefaultLivenessProbeInitialDelay),
			PeriodSeconds:       probeValue(sr.Spec.LivenessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.PeriodSeconds }, DefaultLivenessProbePeriod),
			TimeoutSeconds:      probeValue(sr.Spec.LivenessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.TimeoutSeconds }, DefaultLivenessProbeTimeout),
			FailureThreshold:    probeValue(sr.Spec.LivenessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.FailureThreshold }, DefaultLivenessProbeFailureThreshold),
		}
	}

	if probeEnabled(sr.Spec.ReadinessProbe) {
		container.ReadinessProbe = &corev1.Probe{
			ProbeHandler:        semanticRouterProbeHandler(bootstrap, "/ready"),
			InitialDelaySeconds: probeValue(sr.Spec.ReadinessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.InitialDelaySeconds }, DefaultReadinessProbeInitialDelay),
			PeriodSeconds:       probeValue(sr.Spec.ReadinessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.PeriodSeconds }, DefaultReadinessProbePeriod),
			TimeoutSeconds:      probeValue(sr.Spec.ReadinessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.TimeoutSeconds }, DefaultReadinessProbeTimeout),
			FailureThreshold:    probeValue(sr.Spec.ReadinessProbe, func(spec *vllmv1alpha1.ProbeSpec) *int32 { return spec.FailureThreshold }, DefaultReadinessProbeFailureThreshold),
		}
	}
}

func semanticRouterProbeHandler(
	bootstrap bootstrapDeploymentContract,
	path string,
) corev1.ProbeHandler {
	if bootstrap.Mode == controlPlaneModeManaged {
		return corev1.ProbeHandler{HTTPGet: &corev1.HTTPGetAction{
			Path:   path,
			Port:   intstr.FromString("management"),
			Scheme: corev1.URISchemeHTTPS,
		}}
	}
	return corev1.ProbeHandler{TCPSocket: &corev1.TCPSocketAction{
		Port: intstr.FromString("grpc"),
	}}
}

func probeEnabled(spec *vllmv1alpha1.ProbeSpec) bool {
	return spec == nil || spec.Enabled == nil || *spec.Enabled
}

func probeValue(
	spec *vllmv1alpha1.ProbeSpec,
	value func(*vllmv1alpha1.ProbeSpec) *int32,
	fallback int32,
) int32 {
	if spec == nil {
		return fallback
	}
	if configured := value(spec); configured != nil {
		return *configured
	}
	return fallback
}

func (r *SemanticRouterReconciler) generateEnvoyContainer(sr *vllmv1alpha1.SemanticRouter) corev1.Container {
	return corev1.Container{
		Name:            "envoy-proxy",
		Image:           "envoyproxy/envoy:v1.35.3",
		ImagePullPolicy: corev1.PullIfNotPresent,
		Command:         []string{"/usr/local/bin/envoy"},
		Args: []string{
			"-c",
			"/etc/envoy/envoy.yaml",
			"--component-log-level",
			"ext_proc:info,router:info,http:info",
		},
		SecurityContext: r.getContainerSecurityContext(sr),
		Ports: []corev1.ContainerPort{
			{
				Name:          "envoy-http",
				ContainerPort: 8801,
				Protocol:      corev1.ProtocolTCP,
			},
			{
				Name:          "envoy-admin",
				ContainerPort: 19000,
				Protocol:      corev1.ProtocolTCP,
			},
		},
		VolumeMounts: []corev1.VolumeMount{
			{
				Name:      "envoy-config-volume",
				MountPath: "/etc/envoy",
				ReadOnly:  true,
			},
		},
		Resources: corev1.ResourceRequirements{
			Requests: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("250m"),
				corev1.ResourceMemory: resource.MustParse("256Mi"),
			},
			Limits: corev1.ResourceList{
				corev1.ResourceCPU:    resource.MustParse("500m"),
				corev1.ResourceMemory: resource.MustParse("512Mi"),
			},
		},
		LivenessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(8801),
				},
			},
			InitialDelaySeconds: 30,
			PeriodSeconds:       30,
		},
		ReadinessProbe: &corev1.Probe{
			ProbeHandler: corev1.ProbeHandler{
				TCPSocket: &corev1.TCPSocketAction{
					Port: intstr.FromInt(8801),
				},
			},
			InitialDelaySeconds: 10,
			PeriodSeconds:       15,
		},
	}
}

func (r *SemanticRouterReconciler) generateVolumes(sr *vllmv1alpha1.SemanticRouter, gatewayMode string) []corev1.Volume {
	volumes := []corev1.Volume{
		{
			Name: "config-volume",
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: sr.Spec.Bootstrap.ConfigMapRef.Name,
					},
					Items: []corev1.KeyToPath{{
						Key:  sr.Spec.Bootstrap.ConfigMapRef.Key,
						Path: "config.yaml",
					}},
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

	if gatewayMode == gatewayModeSidecar {
		volumes = append(volumes, corev1.Volume{
			Name: "envoy-config-volume",
			VolumeSource: corev1.VolumeSource{
				ConfigMap: &corev1.ConfigMapVolumeSource{
					LocalObjectReference: corev1.LocalObjectReference{
						Name: sr.Name + "-envoy-config",
					},
				},
			},
		})
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

	volumes = append(volumes, sr.Spec.Volumes...)

	return volumes
}

func (r *SemanticRouterReconciler) generateVolumeMounts(sr *vllmv1alpha1.SemanticRouter) []corev1.VolumeMount {
	mounts := []corev1.VolumeMount{
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
	return append(mounts, sr.Spec.VolumeMounts...)
}

func (r *SemanticRouterReconciler) generateInitContainers() []corev1.Container {
	return []corev1.Container{
		{
			Name:  "setup-dirs",
			Image: "registry.access.redhat.com/ubi9/ubi-minimal:latest",
			Command: []string{
				"sh",
				"-c",
				"mkdir -p /var/log/supervisor",
			},
			VolumeMounts: []corev1.VolumeMount{
				{
					Name:      "var-log",
					MountPath: "/var/log",
				},
				{
					Name:      "var-run",
					MountPath: "/var/run",
				},
			},
			SecurityContext: &corev1.SecurityContext{
				AllowPrivilegeEscalation: func() *bool { b := false; return &b }(),
				Capabilities: &corev1.Capabilities{
					Drop: []corev1.Capability{"ALL"},
				},
			},
		},
	}
}
