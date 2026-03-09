package testcases

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/kubernetes"
)

var (
	httpRouteGVR = schema.GroupVersionResource{
		Group:    "gateway.networking.k8s.io",
		Version:  "v1",
		Resource: "httproutes",
	}
	inferencePoolGVR = schema.GroupVersionResource{
		Group:    "inference.networking.k8s.io",
		Version:  "v1",
		Resource: "inferencepools",
	}
)

type llmdResourceCheck struct {
	routeName             string
	poolName              string
	backendServiceName    string
	endpointPickerService string
	schedulerDeployment   string
	expectedSelectedModel string
}

func init() {
	pkgtestcases.Register("llm-d-inference-gateway-health", pkgtestcases.TestCase{
		Description: "Verify llm-d HTTPRoutes, InferencePools, schedulers, and gateway wiring are healthy",
		Tags:        []string{"llm-d", "gateway", "inference"},
		Fn:          testLLMDInferenceGatewayHealth,
	})
}

func testLLMDInferenceGatewayHealth(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.RestConfig == nil {
		return fmt.Errorf("llm-d health testcase requires Kubernetes REST config")
	}

	dynamicClient, err := newDynamicClient(opts)
	if err != nil {
		return err
	}

	if err := verifyInferenceGatewayService(ctx, client, opts); err != nil {
		return err
	}

	resourceDetails, err := collectLLMDResourceDetails(ctx, client, dynamicClient, opts)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"inference_gateway_service": "inference-gateway-istio",
			"resources":                 resourceDetails,
		})
	}

	return nil
}

func newDynamicClient(opts pkgtestcases.TestCaseOptions) (dynamic.Interface, error) {
	dynamicClient, err := dynamic.NewForConfig(opts.RestConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create dynamic client: %w", err)
	}
	return dynamicClient, nil
}

func verifyInferenceGatewayService(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if err := helpers.VerifyServicePodsRunning(ctx, client, "default", "inference-gateway-istio", opts.Verbose); err != nil {
		return fmt.Errorf("inference gateway is not healthy: %w", err)
	}
	return nil
}

func collectLLMDResourceDetails(
	ctx context.Context,
	client *kubernetes.Clientset,
	dynamicClient dynamic.Interface,
	opts pkgtestcases.TestCaseOptions,
) ([]map[string]interface{}, error) {
	checks := []llmdResourceCheck{
		{
			routeName:             "vsr-llama8b-svc",
			poolName:              "vllm-llama3-8b-instruct",
			backendServiceName:    "vllm-llama3-8b-instruct",
			endpointPickerService: "vllm-llama3-8b-instruct-epp",
			schedulerDeployment:   "llm-d-inference-scheduler-llama3-8b",
			expectedSelectedModel: "llama3-8b",
		},
		{
			routeName:             "vsr-phi4-mini-svc",
			poolName:              "vllm-phi4-mini",
			backendServiceName:    "phi4-mini",
			endpointPickerService: "vllm-phi4-mini-epp",
			schedulerDeployment:   "llm-d-inference-scheduler-phi4-mini",
			expectedSelectedModel: "phi4-mini",
		},
	}

	resourceDetails := make([]map[string]interface{}, 0, len(checks))
	for _, check := range checks {
		detail, err := verifyLLMDResourceCheck(ctx, client, dynamicClient, check, opts)
		if err != nil {
			return nil, err
		}
		resourceDetails = append(resourceDetails, detail)
	}
	return resourceDetails, nil
}

func verifyLLMDResourceCheck(
	ctx context.Context,
	client *kubernetes.Clientset,
	dynamicClient dynamic.Interface,
	check llmdResourceCheck,
	opts pkgtestcases.TestCaseOptions,
) (map[string]interface{}, error) {
	if err := verifyLLMDDeployments(ctx, client, check, opts.Verbose); err != nil {
		return nil, err
	}

	addressCount, err := verifyLLMDBackendService(ctx, client, check.backendServiceName)
	if err != nil {
		return nil, err
	}

	if err := verifyLLMDInferencePool(ctx, dynamicClient, check); err != nil {
		return nil, err
	}

	selectedModel, err := verifyLLMDHTTPRoute(ctx, dynamicClient, check)
	if err != nil {
		return nil, err
	}

	return map[string]interface{}{
		"route":                   check.routeName,
		"inference_pool":          check.poolName,
		"backend_service":         check.backendServiceName,
		"endpoint_picker_service": check.endpointPickerService,
		"scheduler_deployment":    check.schedulerDeployment,
		"selected_model":          selectedModel,
		"backend_endpoints":       addressCount,
	}, nil
}

func verifyLLMDDeployments(ctx context.Context, client *kubernetes.Clientset, check llmdResourceCheck, verbose bool) error {
	if err := helpers.CheckDeployment(ctx, client, "default", check.backendServiceName, verbose); err != nil {
		return fmt.Errorf("backend deployment %s is not healthy: %w", check.backendServiceName, err)
	}
	if err := helpers.CheckDeployment(ctx, client, "default", check.schedulerDeployment, verbose); err != nil {
		return fmt.Errorf("scheduler deployment %s is not healthy: %w", check.schedulerDeployment, err)
	}
	if err := helpers.VerifyServicePodsRunning(ctx, client, "default", check.endpointPickerService, verbose); err != nil {
		return fmt.Errorf("endpoint picker service %s is not healthy: %w", check.endpointPickerService, err)
	}
	return nil
}

func verifyLLMDBackendService(ctx context.Context, client *kubernetes.Clientset, serviceName string) (int, error) {
	addressCount, err := countReadyEndpointAddresses(ctx, client, "default", serviceName)
	if err != nil {
		return 0, fmt.Errorf("backend service %s endpoints check failed: %w", serviceName, err)
	}
	if addressCount == 0 {
		return 0, fmt.Errorf("backend service %s has no ready endpoints", serviceName)
	}
	return addressCount, nil
}

func verifyLLMDInferencePool(ctx context.Context, dynamicClient dynamic.Interface, check llmdResourceCheck) error {
	pool, err := dynamicClient.Resource(inferencePoolGVR).Namespace("default").Get(ctx, check.poolName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get InferencePool %s: %w", check.poolName, err)
	}

	endpointPickerName, found, err := unstructured.NestedString(pool.Object, "spec", "endpointPickerRef", "name")
	if err != nil {
		return fmt.Errorf("failed to read InferencePool %s endpointPickerRef: %w", check.poolName, err)
	}
	if !found || endpointPickerName != check.endpointPickerService {
		return fmt.Errorf(
			"InferencePool %s endpointPickerRef.name mismatch: expected %s, got %q",
			check.poolName,
			check.endpointPickerService,
			endpointPickerName,
		)
	}
	return nil
}

func verifyLLMDHTTPRoute(ctx context.Context, dynamicClient dynamic.Interface, check llmdResourceCheck) (string, error) {
	httpRoute, err := dynamicClient.Resource(httpRouteGVR).Namespace("default").Get(ctx, check.routeName, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("failed to get HTTPRoute %s: %w", check.routeName, err)
	}
	if err := expectHTTPRouteCondition(httpRoute, "Accepted", "True"); err != nil {
		return "", fmt.Errorf("HTTPRoute %s Accepted condition invalid: %w", check.routeName, err)
	}
	if err := expectHTTPRouteCondition(httpRoute, "ResolvedRefs", "True"); err != nil {
		return "", fmt.Errorf("HTTPRoute %s ResolvedRefs condition invalid: %w", check.routeName, err)
	}

	backendName, err := httpRouteBackendInferencePool(httpRoute)
	if err != nil {
		return "", fmt.Errorf("HTTPRoute %s backendRef validation failed: %w", check.routeName, err)
	}
	if backendName != check.poolName {
		return "", fmt.Errorf("HTTPRoute %s backendRef mismatch: expected %s, got %s", check.routeName, check.poolName, backendName)
	}

	selectedModel, err := httpRouteSelectedModelHeader(httpRoute)
	if err != nil {
		return "", fmt.Errorf("HTTPRoute %s selected-model header validation failed: %w", check.routeName, err)
	}
	if selectedModel != check.expectedSelectedModel {
		return "", fmt.Errorf("HTTPRoute %s selected-model mismatch: expected %s, got %s", check.routeName, check.expectedSelectedModel, selectedModel)
	}
	return selectedModel, nil
}

func countReadyEndpointAddresses(ctx context.Context, client *kubernetes.Clientset, namespace, serviceName string) (int, error) {
	endpoints, err := client.CoreV1().Endpoints(namespace).Get(ctx, serviceName, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}

	addressCount := 0
	for _, subset := range endpoints.Subsets {
		addressCount += len(subset.Addresses)
	}
	return addressCount, nil
}

func expectHTTPRouteCondition(route *unstructured.Unstructured, conditionType, expectedStatus string) error {
	parents, found, err := unstructured.NestedSlice(route.Object, "status", "parents")
	if err != nil {
		return err
	}
	if !found || len(parents) == 0 {
		return fmt.Errorf("status.parents is empty")
	}

	for _, parent := range parents {
		parentMap, ok := parent.(map[string]interface{})
		if !ok {
			continue
		}
		conditions, found, err := unstructured.NestedSlice(parentMap, "conditions")
		if err != nil || !found {
			continue
		}
		for _, rawCondition := range conditions {
			conditionMap, ok := rawCondition.(map[string]interface{})
			if !ok {
				continue
			}
			rawType, _, _ := unstructured.NestedString(conditionMap, "type")
			rawStatus, _, _ := unstructured.NestedString(conditionMap, "status")
			if rawType == conditionType && rawStatus == expectedStatus {
				return nil
			}
		}
	}

	return fmt.Errorf("condition %s=%s not found", conditionType, expectedStatus)
}

func httpRouteBackendInferencePool(route *unstructured.Unstructured) (string, error) {
	rules, found, err := unstructured.NestedSlice(route.Object, "spec", "rules")
	if err != nil {
		return "", err
	}
	if !found || len(rules) == 0 {
		return "", fmt.Errorf("spec.rules is empty")
	}

	for _, rule := range rules {
		ruleMap, ok := rule.(map[string]interface{})
		if !ok {
			continue
		}
		backendRefs, found, err := unstructured.NestedSlice(ruleMap, "backendRefs")
		if err != nil || !found {
			continue
		}
		for _, rawBackendRef := range backendRefs {
			backendRefMap, ok := rawBackendRef.(map[string]interface{})
			if !ok {
				continue
			}
			kind, _, _ := unstructured.NestedString(backendRefMap, "kind")
			name, _, _ := unstructured.NestedString(backendRefMap, "name")
			if kind == "InferencePool" && name != "" {
				return name, nil
			}
		}
	}

	return "", fmt.Errorf("no InferencePool backendRef found")
}

func httpRouteSelectedModelHeader(route *unstructured.Unstructured) (string, error) {
	rules, found, err := unstructured.NestedSlice(route.Object, "spec", "rules")
	if err != nil {
		return "", err
	}
	if !found || len(rules) == 0 {
		return "", fmt.Errorf("spec.rules is empty")
	}

	for _, rule := range rules {
		ruleMap, ok := rule.(map[string]interface{})
		if !ok {
			continue
		}
		value, found, err := findMatchHeaderValue(ruleMap, "x-selected-model")
		if err != nil {
			return "", err
		}
		if found {
			return value, nil
		}
	}

	return "", fmt.Errorf("x-selected-model header match not found")
}

func findMatchHeaderValue(rule map[string]interface{}, headerName string) (string, bool, error) {
	matches, found, err := unstructured.NestedSlice(rule, "matches")
	if err != nil {
		return "", false, err
	}
	if !found {
		return "", false, nil
	}

	for _, rawMatch := range matches {
		matchMap, ok := rawMatch.(map[string]interface{})
		if !ok {
			continue
		}
		value, found, err := findHeaderValue(matchMap, headerName)
		if err != nil {
			return "", false, err
		}
		if found {
			return value, true, nil
		}
	}

	return "", false, nil
}

func findHeaderValue(match map[string]interface{}, headerName string) (string, bool, error) {
	headers, found, err := unstructured.NestedSlice(match, "headers")
	if err != nil {
		return "", false, err
	}
	if !found {
		return "", false, nil
	}

	for _, rawHeader := range headers {
		headerMap, ok := rawHeader.(map[string]interface{})
		if !ok {
			continue
		}
		name, _, _ := unstructured.NestedString(headerMap, "name")
		value, _, _ := unstructured.NestedString(headerMap, "value")
		if name == headerName && value != "" {
			return value, true, nil
		}
	}

	return "", false, nil
}
