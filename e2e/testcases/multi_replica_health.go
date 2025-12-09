package testcases

import (
	"context"
	"fmt"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("multi-replica-health", pkgtestcases.TestCase{
		Description: "Verify multi-replica Deployments (router and vLLM) have all replicas Ready",
		Tags:        []string{"ha", "lb", "readiness"},
		Fn:          testMultiReplicaHealth,
	})
}

func testMultiReplicaHealth(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	// Verify semantic-router Deployment has >= 2 ready replicas
	if err := waitDeploymentReadyReplicas(ctx, client, "vllm-semantic-router-system", "semantic-router", 2, 5*time.Minute, opts.Verbose); err != nil {
		return fmt.Errorf("semantic-router not multi-ready: %w", err)
	}
	// Verify vLLM demo Deployment has >= 2 ready replicas
	if err := waitDeploymentReadyReplicas(ctx, client, "default", "vllm-llama3-8b-instruct", 2, 5*time.Minute, opts.Verbose); err != nil {
		return fmt.Errorf("vllm demo not multi-ready: %w", err)
	}
	return nil
}

func waitDeploymentReadyReplicas(ctx context.Context, client *kubernetes.Clientset, namespace, name string, minReady int32, timeout time.Duration, verbose bool) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		dep, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if verbose {
			var desiredReplicas int32
			if dep.Spec.Replicas != nil {
				desiredReplicas = *dep.Spec.Replicas
			}
			fmt.Printf("[Test] %s/%s ready=%d desired=%d\n", namespace, name, dep.Status.ReadyReplicas, desiredReplicas)
		}
		if dep.Status.ReadyReplicas >= minReady {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(5 * time.Second):
		}
	}
	return fmt.Errorf("timeout waiting for %s/%s to have at least %d ready replicas", namespace, name, minReady)
}
