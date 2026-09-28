package testcases

import (
	"context"
	"fmt"
	"slices"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("istio-sidecar-health-check", pkgtestcases.TestCase{
		Description: "Verify Envoy sidecar is injected and healthy in Semantic Router pods",
		Tags:        []string{"istio", "sidecar", "health"},
		Fn:          testIstioSidecarHealthCheck,
	})
}

func testIstioSidecarHealthCheck(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Verifying Istio sidecar injection and health")
	}

	// Istio-specific test: always use vllm-semantic-router-system namespace
	namespace := "vllm-semantic-router-system"

	// Get all pods in the semantic-router namespace
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: "app.kubernetes.io/name=semantic-router",
	})
	if err != nil {
		return fmt.Errorf("failed to list pods: %w", err)
	}

	if len(pods.Items) == 0 {
		return fmt.Errorf("no semantic-router pods found in namespace %s", namespace)
	}

	var (
		totalPods       int
		podsWithSidecar int
		healthySidecars int
		sidecarDetails  []map[string]interface{}
	)

	for _, pod := range pods.Items {
		totalPods++
		details, err := verifyIstioProxy(pod, opts.Verbose)
		if err != nil {
			return err
		}
		podsWithSidecar++
		healthySidecars++
		sidecarDetails = append(sidecarDetails, details)
	}

	// Verify istio-injection label on namespace
	ns, err := client.CoreV1().Namespaces().Get(ctx, namespace, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get namespace %s: %w", namespace, err)
	}

	injectionEnabled := ns.Labels["istio-injection"] == "enabled"
	if !injectionEnabled {
		return fmt.Errorf("namespace %s does not have istio-injection=enabled label", namespace)
	}

	// Set details for reporting
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_pods":                totalPods,
			"pods_with_sidecar":         podsWithSidecar,
			"healthy_sidecars":          healthySidecars,
			"namespace_injection_label": injectionEnabled,
			"sidecar_details":           sidecarDetails,
		})
	}

	if opts.Verbose {
		fmt.Printf("[Test] ✅ Istio sidecar health check passed: %d/%d pods have healthy sidecars\n",
			healthySidecars, totalPods)
	}

	return nil
}

func verifyIstioProxy(pod corev1.Pod, verbose bool) (map[string]interface{}, error) {
	injected, ready := istioProxyStatus(pod)
	if verbose {
		fmt.Printf("[Test] Pod %s: sidecar_injected=%v, sidecar_healthy=%v\n", pod.Name, injected, ready)
	}

	details := map[string]interface{}{
		"pod_name":         pod.Name,
		"sidecar_injected": injected,
		"sidecar_healthy":  ready,
		"pod_phase":        string(pod.Status.Phase),
	}
	if !injected {
		return details, fmt.Errorf("pod %s does not have istio-proxy sidecar injected", pod.Name)
	}
	if !ready {
		return details, fmt.Errorf("pod %s has istio-proxy sidecar but it is not healthy", pod.Name)
	}
	return details, nil
}

func istioProxyStatus(pod corev1.Pod) (injected, ready bool) {
	for _, container := range slices.Concat(pod.Spec.Containers, pod.Spec.InitContainers) {
		if container.Name != "istio-proxy" {
			continue
		}
		for _, status := range slices.Concat(pod.Status.ContainerStatuses, pod.Status.InitContainerStatuses) {
			if status.Name == "istio-proxy" {
				return true, status.Ready
			}
		}
		return true, false
	}
	return false, false
}
