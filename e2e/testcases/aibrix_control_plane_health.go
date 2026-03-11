package testcases

import (
	"context"
	"fmt"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("aibrix-control-plane-health", pkgtestcases.TestCase{
		Description: "Verify the AIBrix control plane, model service wiring, and gateway health",
		Tags:        []string{"aibrix", "control-plane", "gateway"},
		Fn:          testAIBrixControlPlaneHealth,
	})
}

func testAIBrixControlPlaneHealth(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	deployments := []struct {
		namespace string
		name      string
	}{
		{namespace: "aibrix-system", name: "aibrix-gateway-plugins"},
		{namespace: "aibrix-system", name: "aibrix-metadata-service"},
		{namespace: "aibrix-system", name: "aibrix-controller-manager"},
	}

	deploymentDetails := make([]map[string]interface{}, 0, len(deployments))
	for _, deploymentRef := range deployments {
		deployment, err := client.AppsV1().Deployments(deploymentRef.namespace).Get(ctx, deploymentRef.name, metav1.GetOptions{})
		if err != nil {
			return fmt.Errorf("failed to get deployment %s/%s: %w", deploymentRef.namespace, deploymentRef.name, err)
		}
		if err := helpers.CheckDeployment(ctx, client, deploymentRef.namespace, deploymentRef.name, opts.Verbose); err != nil {
			return fmt.Errorf("deployment %s/%s is not healthy: %w", deploymentRef.namespace, deploymentRef.name, err)
		}

		deploymentDetails = append(deploymentDetails, map[string]interface{}{
			"namespace":      deploymentRef.namespace,
			"name":           deploymentRef.name,
			"ready_replicas": deployment.Status.ReadyReplicas,
			"replicas":       deployment.Status.Replicas,
		})
	}

	modelService, err := client.CoreV1().Services("default").Get(ctx, "vllm-llama3-8b-instruct", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get AIBrix model service: %w", err)
	}
	if labelValue := modelService.Labels["model.aibrix.ai/name"]; labelValue != "vllm-llama3-8b-instruct" {
		return fmt.Errorf("AIBrix model service label mismatch: expected model.aibrix.ai/name=vllm-llama3-8b-instruct, got %q", labelValue)
	}
	if err := helpers.VerifyServicePodsRunning(ctx, client, "default", modelService.Name, opts.Verbose); err != nil {
		return fmt.Errorf("AIBrix model service is not backed by ready pods: %w", err)
	}

	gatewayServiceName, err := helpers.GetServiceByLabelInNamespace(
		ctx,
		client,
		opts.ServiceConfig.Namespace,
		opts.ServiceConfig.LabelSelector,
		opts.Verbose,
	)
	if err != nil {
		return fmt.Errorf("failed to resolve AIBrix gateway service: %w", err)
	}
	if err := helpers.VerifyServicePodsRunning(ctx, client, opts.ServiceConfig.Namespace, gatewayServiceName, opts.Verbose); err != nil {
		return fmt.Errorf("AIBrix gateway service is not backed by ready pods: %w", err)
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"aibrix_deployments":   deploymentDetails,
			"gateway_service":      gatewayServiceName,
			"model_service":        modelService.Name,
			"model_service_labels": modelService.Labels,
		})
	}

	return nil
}
