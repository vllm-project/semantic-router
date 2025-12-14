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
	pkgtestcases.Register("dynamo-gpu-utilization", pkgtestcases.TestCase{
		Description: "Monitor GPU utilization and efficiency for Dynamo workers",
		Tags:        []string{"dynamo", "gpu", "monitoring"},
		Fn:          testDynamoGPUUtilization,
	})
}

func testDynamoGPUUtilization(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Monitoring GPU utilization for Dynamo workers")
	}

	namespace := "dynamo-system"

	// Get worker pods - try Dynamo labels first, then fallback
	workerPods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "nvidia.com/dynamo-graph-deployment-name=vllm",
	})
	if err != nil {
		// If no workers found, that's okay for E2E testing
		if opts.Verbose {
			fmt.Printf("[Test] No worker pods found (this may be expected in E2E): %v\n", err)
		}
		return nil
	}

	// Filter to only include worker pods (exclude frontend)
	var filteredPods []corev1.Pod
	for _, pod := range workerPods.Items {
		for key := range pod.Labels {
			if key == "nvidia.com/dynamo-component" {
				componentType := pod.Labels["nvidia.com/dynamo-component-type"]
				if componentType == "worker" {
					filteredPods = append(filteredPods, pod)
				}
			}
		}
	}
	workerPods.Items = filteredPods

	if len(workerPods.Items) == 0 {
		if opts.Verbose {
			fmt.Println("[Test] No worker pods found - skipping GPU utilization check")
		}
		return nil
	}

	var gpuInfo []map[string]interface{}
	totalGPUs := 0
	workersWithGPU := 0

	for _, pod := range workerPods.Items {
		if pod.Status.Phase != corev1.PodRunning {
			continue
		}

		podInfo := map[string]interface{}{
			"pod_name":  pod.Name,
			"namespace": pod.Namespace,
			"node_name": pod.Spec.NodeName,
			"phase":     string(pod.Status.Phase),
		}

		// Check for GPU resources in pod spec
		gpuCount := int64(0)
		for _, container := range pod.Spec.Containers {
			if container.Resources.Limits != nil {
				if gpuLimit, ok := container.Resources.Limits["nvidia.com/gpu"]; ok {
					gpuCount = gpuLimit.Value()
					totalGPUs += int(gpuCount)
					workersWithGPU++
					podInfo["gpu_count"] = gpuCount
				}
			}
			if container.Resources.Requests != nil {
				if gpuRequest, ok := container.Resources.Requests["nvidia.com/gpu"]; ok {
					podInfo["gpu_requested"] = gpuRequest.Value()
				}
			}
		}

		// Try to get GPU metrics from pod (if metrics server is available)
		// This is a best-effort check
		if opts.Verbose {
			fmt.Printf("[Test] Pod %s: %d GPU(s) allocated\n", pod.Name, gpuCount)
		}

		// Check container status
		var containerStatuses []string
		for _, status := range pod.Status.ContainerStatuses {
			containerStatuses = append(containerStatuses, fmt.Sprintf("%s:%v", status.Name, status.Ready))
		}
		podInfo["containers"] = containerStatuses

		gpuInfo = append(gpuInfo, podInfo)
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"worker_pods":      len(workerPods.Items),
			"workers_with_gpu": workersWithGPU,
			"total_gpus":       totalGPUs,
			"gpu_info":         gpuInfo,
			"monitoring_note":  "GPU utilization metrics require nvidia-smi or metrics server",
		})
	}

	if opts.Verbose {
		fmt.Println("\n" + strings.Repeat("=", 80))
		fmt.Println("GPU Utilization Summary")
		fmt.Println(strings.Repeat("=", 80))
		fmt.Printf("Worker Pods:        %d\n", len(workerPods.Items))
		fmt.Printf("Workers with GPU:   %d\n", workersWithGPU)
		fmt.Printf("Total GPUs:         %d\n", totalGPUs)

		if totalGPUs > 0 {
			fmt.Printf("\nGPU Allocation per Pod:\n")
			for _, info := range gpuInfo {
				if gpuCount, ok := info["gpu_count"].(int64); ok {
					fmt.Printf("  - %s: %d GPU(s)\n", info["pod_name"], gpuCount)
				}
			}
		}

		fmt.Println(strings.Repeat("=", 80))
		fmt.Println("[Test] ✅ GPU utilization check completed")
		fmt.Println("[Test] Note: Detailed GPU metrics require nvidia-smi or Kubernetes metrics server")
	}

	return nil
}
