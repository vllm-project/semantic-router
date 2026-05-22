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
	"fmt"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	corev1 "k8s.io/api/core/v1"
	networkingv1 "k8s.io/api/networking/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

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

func (r *SemanticRouterReconciler) generateService(sr *vllmv1alpha1.SemanticRouter, gatewayMode string) *corev1.Service {
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

	if gatewayMode == "standalone" {
		ports = append(ports, corev1.ServicePort{
			Name:       "envoy-http",
			Port:       8801,
			TargetPort: intstr.FromInt(8801),
			Protocol:   corev1.ProtocolTCP,
		})
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
			pt := resolveIngressPathType(path.PathType, pathType)
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

func resolveIngressPathType(pathType string, defaultType networkingv1.PathType) networkingv1.PathType {
	if pathType == "" {
		return defaultType
	}
	switch pathType {
	case "Exact":
		return networkingv1.PathTypeExact
	case "Prefix":
		return networkingv1.PathTypePrefix
	default:
		return defaultType
	}
}
