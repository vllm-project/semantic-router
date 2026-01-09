package testcases

import (
	"context"
	"fmt"
	"strings"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("dynamo-health-check", pkgtestcases.TestCase{
		Description: "Verify Dynamo runtime components are healthy",
		Tags:        []string{"dynamo", "health", "functional"},
		Fn:          testDynamoHealthCheck,
	})
}

func testDynamoHealthCheck(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Checking Dynamo health")
	}

	namespace := "dynamo-system"

	// Check Dynamo Platform components (deployed via Helm)
	// The platform chart deploys: etcd, NATS, and operator
	if opts.Verbose {
		fmt.Println("[Test] Checking Dynamo Platform components...")
	}

	// Check Dynamo operator (controller manager)
	operatorDeployment, err := client.AppsV1().Deployments(namespace).Get(ctx, "dynamo-platform-dynamo-operator-controller-manager", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get dynamo-operator deployment: %w", err)
	}

	if operatorDeployment.Status.ReadyReplicas == 0 {
		return fmt.Errorf("dynamo-operator has 0 ready replicas")
	}

	if opts.Verbose {
		fmt.Printf("[Test] Dynamo Operator: %d/%d replicas ready\n",
			operatorDeployment.Status.ReadyReplicas, operatorDeployment.Status.Replicas)
	}

	// Check etcd StatefulSet
	etcdStatefulSet, err := client.AppsV1().StatefulSets(namespace).Get(ctx, "dynamo-platform-etcd", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get etcd statefulset: %w", err)
	}

	if etcdStatefulSet.Status.ReadyReplicas == 0 {
		return fmt.Errorf("etcd has 0 ready replicas")
	}

	if opts.Verbose {
		fmt.Printf("[Test] etcd: %d/%d replicas ready\n",
			etcdStatefulSet.Status.ReadyReplicas, etcdStatefulSet.Status.Replicas)
	}

	// Check NATS StatefulSet
	natsStatefulSet, err := client.AppsV1().StatefulSets(namespace).Get(ctx, "dynamo-platform-nats", metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get nats statefulset: %w", err)
	}

	if natsStatefulSet.Status.ReadyReplicas == 0 {
		return fmt.Errorf("NATS has 0 ready replicas")
	}

	if opts.Verbose {
		fmt.Printf("[Test] NATS: %d/%d replicas ready\n",
			natsStatefulSet.Status.ReadyReplicas, natsStatefulSet.Status.Replicas)
	}

	// Check operator pods
	operatorPods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "control-plane=controller-manager",
	})
	if err != nil {
		return fmt.Errorf("failed to list dynamo-operator pods: %w", err)
	}

	healthyOperatorPods := 0
	for _, pod := range operatorPods.Items {
		if pod.Status.Phase == corev1.PodRunning {
			allContainersReady := true
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if !containerStatus.Ready {
					allContainersReady = false
					break
				}
			}
			if allContainersReady {
				healthyOperatorPods++
			}
		}
	}

	if healthyOperatorPods == 0 {
		return fmt.Errorf("no healthy dynamo-operator pods found")
	}

	if opts.Verbose {
		fmt.Printf("[Test] Dynamo Operator: %d healthy pod(s)\n", healthyOperatorPods)
	}

	// Check for Dynamo Frontend (if operator created it from DynamoGraphDeployment)
	if opts.Verbose {
		fmt.Println("[Test] Checking for Dynamo Frontend (operator-created)...")
	}

	// Check Frontend deployment
	// The deployment name depends on how it was deployed:
	// - Helm chart: "dynamo-vllm-frontend" or "dynamo-vllm"
	// - Dynamo operator: "vllm-frontend"
	frontendNames := []string{
		"dynamo-vllm-frontend", // Helm chart generated
		"vllm-frontend",        // Operator generated
		"dynamo-vllm",          // Helm release name fallback
	}
	frontendFound := false
	for _, name := range frontendNames {
		frontendDeployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil && frontendDeployment.Status.ReadyReplicas > 0 {
			if opts.Verbose {
				fmt.Printf("[Test] ✅ Frontend %s: %d/%d replicas ready\n",
					name, frontendDeployment.Status.ReadyReplicas, frontendDeployment.Status.Replicas)
			}
			frontendFound = true
			break
		}
	}
	if !frontendFound {
		return fmt.Errorf("frontend deployment not found or not ready")
	}

	// Check for Prefill Worker (disaggregated deployment)
	// Names depend on deployment method:
	// - Helm chart: "dynamo-vllm-prefillworker0" (worker at index 0)
	// - Dynamo operator: "vllm-vllmprefillworker"
	prefillWorkerNames := []string{
		"dynamo-vllm-prefillworker0", // Helm chart generated (index 0)
		"dynamo-vllm-prefillworker1", // Legacy name
		"vllm-vllmprefillworker",     // Operator generated
		"vllm-prefillworker",
		"prefill-worker-0",
	}
	prefillFound := false
	for _, name := range prefillWorkerNames {
		deployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil && deployment.Status.ReadyReplicas > 0 {
			if opts.Verbose {
				fmt.Printf("[Test] ✅ Prefill Worker %s: %d/%d replicas ready\n",
					name, deployment.Status.ReadyReplicas, deployment.Status.Replicas)
			}
			prefillFound = true
			break
		}
	}
	if !prefillFound && opts.Verbose {
		fmt.Println("[Test] ⚠️  Prefill Worker not found (may be using aggregated deployment)")
	}

	// Check for Decode Worker
	// Names depend on deployment method:
	// - Helm chart: "dynamo-vllm-decodeworker1" (CamelCase from worker name)
	// - Dynamo operator: "vllm-vllmdecodeworker"
	decodeWorkerNames := []string{
		"dynamo-vllm-decodeworker1", // Helm chart generated (CamelCase)
		"vllm-vllmdecodeworker",     // Operator generated
		"vllm-decodeworker",
		"decode-worker-1",
	}
	decodeFound := false
	for _, name := range decodeWorkerNames {
		deployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err == nil && deployment.Status.ReadyReplicas > 0 {
			if opts.Verbose {
				fmt.Printf("[Test] ✅ Decode Worker %s: %d/%d replicas ready\n",
					name, deployment.Status.ReadyReplicas, deployment.Status.Replicas)
			}
			decodeFound = true
			break
		}
	}
	if !decodeFound {
		return fmt.Errorf("decode worker deployment not found or not ready")
	}

	// Check worker pods (these coordinate via Dynamo's etcd/NATS)
	// Dynamo operator creates pods with labels like nvidia.com/dynamo-component=VLLMDecodeWorker
	if opts.Verbose {
		fmt.Println("[Test] Checking worker pods...")
	}

	// Try multiple label selectors that Dynamo operator might use
	// Includes both Prefill and Decode workers for disaggregated deployment
	workerLabelSelectors := []string{
		"nvidia.com/dynamo-component=VLLMDecodeWorker",
		"nvidia.com/dynamo-component=VLLMPrefillWorker",
		"nvidia.com/dynamo-component-type=worker",
		"nvidia.com/dynamo-graph-deployment-name=vllm",
	}

	var workerPods *corev1.PodList
	for _, selector := range workerLabelSelectors {
		workerPods, err = client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
			LabelSelector: selector,
		})
		if err == nil && len(workerPods.Items) > 0 {
			if opts.Verbose {
				fmt.Printf("[Test] Found worker pods using selector: %s\n", selector)
			}
			break
		}
	}

	if workerPods == nil || len(workerPods.Items) == 0 {
		// If no worker pods found with specific labels, try listing all pods in namespace
		// and filter by name pattern
		allPods, listErr := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{})
		if listErr != nil {
			return fmt.Errorf("failed to list pods: %w", listErr)
		}

		workerPods = &corev1.PodList{}
		for _, pod := range allPods.Items {
			// Match pods with "vllmdecodeworker" in the name (Dynamo operator naming)
			if strings.Contains(strings.ToLower(pod.Name), "vllmdecodeworker") ||
				strings.Contains(strings.ToLower(pod.Name), "worker") {
				workerPods.Items = append(workerPods.Items, pod)
			}
		}

		if len(workerPods.Items) == 0 {
			return fmt.Errorf("no worker pods found in namespace %s", namespace)
		}

		if opts.Verbose {
			fmt.Printf("[Test] Found %d worker pods by name pattern\n", len(workerPods.Items))
		}
	}

	healthyWorkerPods := 0
	for _, pod := range workerPods.Items {
		if pod.Status.Phase == corev1.PodRunning {
			allContainersReady := true
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if !containerStatus.Ready {
					allContainersReady = false
					break
				}
			}
			if allContainersReady {
				healthyWorkerPods++
			}
		}
	}

	if healthyWorkerPods == 0 {
		return fmt.Errorf("no healthy worker pods found")
	}

	if opts.Verbose {
		fmt.Printf("[Test] Worker pods: %d/%d healthy\n", healthyWorkerPods, len(workerPods.Items))
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"operator_replicas_ready": operatorDeployment.Status.ReadyReplicas,
			"operator_replicas_total": operatorDeployment.Status.Replicas,
			"etcd_replicas_ready":     etcdStatefulSet.Status.ReadyReplicas,
			"etcd_replicas_total":     etcdStatefulSet.Status.Replicas,
			"nats_replicas_ready":     natsStatefulSet.Status.ReadyReplicas,
			"nats_replicas_total":     natsStatefulSet.Status.Replicas,
			"healthy_operator_pods":   healthyOperatorPods,
		})
	}

	if opts.Verbose {
		fmt.Println("[Test] ✅ Dynamo health check passed")
	}

	return nil
}
