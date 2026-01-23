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

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	vllmv1alpha1 "github.com/vllm-project/semantic-router/operator/api/v1alpha1"
)

// BackendEndpoint represents a discovered backend endpoint
type BackendEndpoint struct {
	Name    string
	Address string
	Port    int32
	Weight  int
}

// ModelConfig represents model configuration for vllm
type ModelConfig struct {
	ReasoningFamily    string   `yaml:"reasoning_family,omitempty"`
	PreferredEndpoints []string `yaml:"preferred_endpoints,omitempty"`
}

// discoverKServeBackend discovers backend from KServe InferenceService
// Uses unstructured objects to avoid KServe version dependencies
func discoverKServeBackend(ctx context.Context, c client.Client, namespace string, inferenceServiceName string) (*BackendEndpoint, error) {
	logger := log.FromContext(ctx)

	// Define the GVR for InferenceService
	inferenceServiceGVR := schema.GroupVersionResource{
		Group:    "serving.kserve.io",
		Version:  "v1beta1",
		Resource: "inferenceservices",
	}

	// Get the InferenceService as unstructured
	inferenceService := &unstructured.Unstructured{}
	inferenceService.SetGroupVersionKind(schema.GroupVersionKind{
		Group:   inferenceServiceGVR.Group,
		Version: inferenceServiceGVR.Version,
		Kind:    "InferenceService",
	})

	err := c.Get(ctx, types.NamespacedName{
		Name:      inferenceServiceName,
		Namespace: namespace,
	}, inferenceService)

	if err != nil {
		logger.Error(err, "Failed to get InferenceService", "name", inferenceServiceName, "namespace", namespace)
		return nil, fmt.Errorf("failed to get InferenceService %s/%s: %w", namespace, inferenceServiceName, err)
	}

	// Extract predictor service information
	// KServe creates a predictor service with name: {inference-service-name}-predictor
	predictorServiceName := fmt.Sprintf("%s-predictor", inferenceServiceName)

	// KServe typically uses port 8443 for HTTPS or 8080 for HTTP
	port := int32(8443)

	// Try to extract URL from status if available
	statusURL, found, err := unstructured.NestedString(inferenceService.Object, "status", "url")
	if err == nil && found && statusURL != "" {
		logger.Info("InferenceService URL", "url", statusURL)
	}

	address := fmt.Sprintf("%s.%s.svc.cluster.local", predictorServiceName, namespace)

	endpoint := &BackendEndpoint{
		Address: address,
		Port:    port,
	}

	logger.Info("Discovered KServe backend", "address", address, "port", port)
	return endpoint, nil
}

// discoverLlamaStackBackend discovers backend from Llama Stack services using labels
func discoverLlamaStackBackend(ctx context.Context, c client.Client, namespace string, discoveryLabels map[string]string) (*BackendEndpoint, error) {
	logger := log.FromContext(ctx)

	// List services matching the labels
	serviceList := &corev1.ServiceList{}
	labelSelector := labels.SelectorFromSet(discoveryLabels)

	err := c.List(ctx, serviceList, &client.ListOptions{
		Namespace:     namespace,
		LabelSelector: labelSelector,
	})

	if err != nil {
		logger.Error(err, "Failed to list services", "namespace", namespace, "labels", discoveryLabels)
		return nil, fmt.Errorf("failed to list services in namespace %s: %w", namespace, err)
	}

	if len(serviceList.Items) == 0 {
		return nil, fmt.Errorf("no services found matching labels %v in namespace %s", discoveryLabels, namespace)
	}

	if len(serviceList.Items) > 1 {
		logger.Info("Multiple services found, using first one", "count", len(serviceList.Items))
	}

	service := serviceList.Items[0]

	// Extract port from service
	if len(service.Spec.Ports) == 0 {
		return nil, fmt.Errorf("service %s has no ports defined", service.Name)
	}

	port := service.Spec.Ports[0].Port
	address := fmt.Sprintf("%s.%s.svc.cluster.local", service.Name, namespace)

	endpoint := &BackendEndpoint{
		Address: address,
		Port:    port,
	}

	logger.Info("Discovered Llama Stack backend", "address", address, "port", port, "service", service.Name)
	return endpoint, nil
}

// discoverServiceBackend discovers backend from direct service reference
func discoverServiceBackend(ctx context.Context, serviceBackend *vllmv1alpha1.ServiceBackend, defaultNamespace string) (*BackendEndpoint, error) {
	logger := log.FromContext(ctx)

	namespace := serviceBackend.Namespace
	if namespace == "" {
		namespace = defaultNamespace
	}

	address := fmt.Sprintf("%s.%s.svc.cluster.local", serviceBackend.Name, namespace)

	endpoint := &BackendEndpoint{
		Address: address,
		Port:    serviceBackend.Port,
	}

	logger.Info("Configured service backend", "address", address, "port", serviceBackend.Port)
	return endpoint, nil
}

// discoverBackendEndpoint discovers a backend endpoint based on the VLLMEndpointSpec
func discoverBackendEndpoint(ctx context.Context, c client.Client, vllmEndpoint vllmv1alpha1.VLLMEndpointSpec, namespace string) (*BackendEndpoint, error) {
	logger := log.FromContext(ctx)

	var endpoint *BackendEndpoint
	var err error

	switch vllmEndpoint.Backend.Type {
	case "kserve":
		if vllmEndpoint.Backend.InferenceServiceName == "" {
			return nil, fmt.Errorf("inferenceServiceName is required for backend type kserve")
		}
		endpoint, err = discoverKServeBackend(ctx, c, namespace, vllmEndpoint.Backend.InferenceServiceName)

	case "llamastack":
		if len(vllmEndpoint.Backend.DiscoveryLabels) == 0 {
			return nil, fmt.Errorf("discoveryLabels are required for backend type llamastack")
		}
		endpoint, err = discoverLlamaStackBackend(ctx, c, namespace, vllmEndpoint.Backend.DiscoveryLabels)

	case "service":
		if vllmEndpoint.Backend.Service == nil {
			return nil, fmt.Errorf("service configuration is required for backend type service")
		}
		endpoint, err = discoverServiceBackend(ctx, vllmEndpoint.Backend.Service, namespace)

	default:
		return nil, fmt.Errorf("unknown backend type: %s", vllmEndpoint.Backend.Type)
	}

	if err != nil {
		return nil, err
	}

	// Set name and weight from VLLMEndpointSpec
	endpoint.Name = vllmEndpoint.Name
	endpoint.Weight = vllmEndpoint.Weight
	if endpoint.Weight == 0 {
		endpoint.Weight = 1
	}

	logger.Info("Discovered backend endpoint", "name", endpoint.Name, "address", endpoint.Address, "port", endpoint.Port, "weight", endpoint.Weight)
	return endpoint, nil
}

// generateVLLMEndpointsConfig generates YAML configuration for vllm_endpoints and model_config
func generateVLLMEndpointsConfig(ctx context.Context, c client.Client, vllmEndpoints []vllmv1alpha1.VLLMEndpointSpec, namespace string) (map[string]interface{}, map[string]ModelConfig, error) {
	logger := log.FromContext(ctx)

	if len(vllmEndpoints) == 0 {
		return nil, nil, nil
	}

	endpoints := make([]map[string]interface{}, 0)
	modelConfigs := make(map[string]ModelConfig)

	for _, vllmEndpoint := range vllmEndpoints {
		// Discover backend endpoint
		endpoint, err := discoverBackendEndpoint(ctx, c, vllmEndpoint, namespace)
		if err != nil {
			logger.Error(err, "Failed to discover backend endpoint", "name", vllmEndpoint.Name)
			// Continue with other endpoints instead of failing completely
			continue
		}

		// Add to endpoints list
		endpointConfig := map[string]interface{}{
			"name":    endpoint.Name,
			"address": endpoint.Address,
			"port":    endpoint.Port,
			"weight":  endpoint.Weight,
		}
		endpoints = append(endpoints, endpointConfig)

		// Add to model configs
		if vllmEndpoint.Model != "" {
			modelConfig := ModelConfig{
				ReasoningFamily:    vllmEndpoint.ReasoningFamily,
				PreferredEndpoints: []string{endpoint.Name},
			}
			modelConfigs[vllmEndpoint.Model] = modelConfig
		}
	}

	if len(endpoints) == 0 {
		logger.Info("No backend endpoints discovered")
		return nil, nil, nil
	}

	vllmEndpointsConfig := map[string]interface{}{
		"vllm_endpoints": endpoints,
	}

	logger.Info("Generated vLLM endpoints configuration", "count", len(endpoints))
	return vllmEndpointsConfig, modelConfigs, nil
}
