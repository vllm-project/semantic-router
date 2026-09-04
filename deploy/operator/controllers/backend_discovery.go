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
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// BackendEndpoint represents a discovered backend endpoint
type BackendEndpoint struct {
	Name     string
	Address  string
	Port     int32
	Protocol string
	Weight   int
}

// DiscoveredProviderModel represents one logical model plus its discovered
// backend bindings and routing metadata.
type DiscoveredProviderModel struct {
	Catalog     string
	Reasoning   *routerconfig.CanonicalReasoning
	BackendRefs []routerconfig.CanonicalBackendRef
	LoRAs       []vllmv1alpha1.LoRAAdapterSpec
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
		Address:  address,
		Port:     port,
		Protocol: "https",
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
		Address:  address,
		Port:     port,
		Protocol: "http",
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
		Address:  address,
		Port:     serviceBackend.Port,
		Protocol: "http",
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

// discoverVLLMBackends discovers Kubernetes backends and returns the backend
// refs plus model metadata used to render canonical providers.models[].backend_refs
// and routing.modelCards.
func discoverVLLMBackends(ctx context.Context, c client.Client, vllmEndpoints []vllmv1alpha1.VLLMEndpointSpec, namespace string) (map[string]DiscoveredProviderModel, error) {
	logger := log.FromContext(ctx)

	if len(vllmEndpoints) == 0 {
		return nil, nil
	}

	models := make(map[string]DiscoveredProviderModel)

	for _, vllmEndpoint := range vllmEndpoints {
		endpoint, err := discoverBackendEndpoint(ctx, c, vllmEndpoint, namespace)
		if err != nil {
			logger.Error(err, "Failed to discover backend endpoint", "name", vllmEndpoint.Name)
			// Continue with other endpoints instead of failing completely
			continue
		}
		if vllmEndpoint.Model == "" {
			continue
		}
		modelConfig, err := mergeDiscoveredProviderModel(models[vllmEndpoint.Model], vllmEndpoint, endpoint)
		if err != nil {
			return nil, err
		}
		models[vllmEndpoint.Model] = modelConfig
	}

	if len(models) == 0 {
		logger.Info("No backend endpoints discovered")
		return nil, nil
	}

	logger.Info("Generated discovered backend refs", "count", len(models))
	return models, nil
}

func mergeDiscoveredProviderModel(
	existing DiscoveredProviderModel,
	spec vllmv1alpha1.VLLMEndpointSpec,
	endpoint *BackendEndpoint,
) (DiscoveredProviderModel, error) {
	catalog, err := mergeDiscoveredCatalog(existing.Catalog, spec.Catalog, spec.Model)
	if err != nil {
		return existing, err
	}
	reasoning, err := mergeDiscoveredReasoning(existing.Reasoning, spec.Reasoning, spec.Model)
	if err != nil {
		return existing, err
	}
	existing.Catalog = catalog
	existing.Reasoning = reasoning
	existing.BackendRefs = append(existing.BackendRefs, routerconfig.CanonicalBackendRef{
		Name:     endpoint.Name,
		Endpoint: fmt.Sprintf("%s:%d", endpoint.Address, endpoint.Port),
		Protocol: endpoint.Protocol,
		Weight:   endpoint.Weight,
		Provider: "vllm",
	})
	existing.LoRAs = mergeDiscoveredLoRAs(existing.LoRAs, spec.LoRAs)
	return existing, nil
}

func mergeDiscoveredCatalog(existing, incoming, model string) (string, error) {
	if existing != "" && incoming != "" && existing != incoming {
		return "", fmt.Errorf("model %q declares conflicting catalog identities %q and %q", model, existing, incoming)
	}
	if existing != "" {
		return existing, nil
	}
	return incoming, nil
}

func mergeDiscoveredReasoning(
	existing *routerconfig.CanonicalReasoning,
	incoming *vllmv1alpha1.ModelReasoningSpec,
	model string,
) (*routerconfig.CanonicalReasoning, error) {
	candidate := canonicalReasoning(incoming)
	if existing != nil && candidate != nil && !sameCanonicalReasoning(existing, candidate) {
		return nil, fmt.Errorf("model %q declares conflicting reasoning behavior", model)
	}
	if existing != nil {
		return existing, nil
	}
	return candidate, nil
}

func canonicalReasoning(spec *vllmv1alpha1.ModelReasoningSpec) *routerconfig.CanonicalReasoning {
	if spec == nil {
		return nil
	}
	return &routerconfig.CanonicalReasoning{
		Family: spec.Family, Type: spec.Type, Parameter: spec.Parameter,
		Levels: append([]string(nil), spec.Levels...), Default: spec.Default,
	}
}

func sameCanonicalReasoning(left, right *routerconfig.CanonicalReasoning) bool {
	if left == nil || right == nil {
		return left == right
	}
	if left.Family != right.Family || left.Type != right.Type ||
		left.Parameter != right.Parameter || left.Default != right.Default ||
		len(left.Levels) != len(right.Levels) {
		return false
	}
	for index := range left.Levels {
		if left.Levels[index] != right.Levels[index] {
			return false
		}
	}
	return true
}

func mergeDiscoveredLoRAs(existing []vllmv1alpha1.LoRAAdapterSpec, incoming []vllmv1alpha1.LoRAAdapterSpec) []vllmv1alpha1.LoRAAdapterSpec {
	if len(incoming) == 0 {
		return existing
	}

	merged := append([]vllmv1alpha1.LoRAAdapterSpec(nil), existing...)
	seen := make(map[string]struct{}, len(merged))
	for _, adapter := range merged {
		seen[adapter.Name] = struct{}{}
	}
	for _, adapter := range incoming {
		if _, ok := seen[adapter.Name]; ok {
			continue
		}
		merged = append(merged, adapter)
		seen[adapter.Name] = struct{}{}
	}
	return merged
}
