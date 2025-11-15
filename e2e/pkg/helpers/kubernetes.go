package helpers

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
)

// CheckDeployment checks if a deployment is healthy (ready replicas > 0)
func CheckDeployment(ctx context.Context, client *kubernetes.Clientset, namespace, name string, verbose bool) error {
	deployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deployment: %w", err)
	}

	if deployment.Status.ReadyReplicas == 0 {
		return fmt.Errorf("deployment has 0 ready replicas")
	}

	if verbose {
		fmt.Printf("[Helper] Deployment %s/%s is healthy (%d/%d replicas ready)\n",
			namespace, name, deployment.Status.ReadyReplicas, deployment.Status.Replicas)
	}

	return nil
}

// GetEnvoyServiceName finds the Envoy service name in the envoy-gateway-system namespace
// using label selectors to match the Gateway-owned service
func GetEnvoyServiceName(ctx context.Context, client *kubernetes.Clientset, labelSelector string, verbose bool) (string, error) {
	services, err := client.CoreV1().Services("envoy-gateway-system").List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return "", fmt.Errorf("failed to list services with selector %s: %w", labelSelector, err)
	}

	if len(services.Items) == 0 {
		return "", fmt.Errorf("no service found with selector %s in envoy-gateway-system namespace", labelSelector)
	}

	// Return the first matching service (should only be one)
	serviceName := services.Items[0].Name
	if verbose {
		fmt.Printf("[Helper] Found Envoy service: %s (matched by labels: %s)\n", serviceName, labelSelector)
	}

	return serviceName, nil
}

// VerifyServicePodsRunning verifies that exactly 1 pod exists for a service and it's running with all containers ready
func VerifyServicePodsRunning(ctx context.Context, client *kubernetes.Clientset, namespace, serviceName string, verbose bool) error {
	// Get the service
	svc, err := client.CoreV1().Services(namespace).Get(ctx, serviceName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get service: %w", err)
	}

	// Build label selector from service selector
	var selectorParts []string
	for key, value := range svc.Spec.Selector {
		selectorParts = append(selectorParts, fmt.Sprintf("%s=%s", key, value))
	}
	labelSelector := strings.Join(selectorParts, ",")

	// List pods matching the selector
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return fmt.Errorf("failed to list pods: %w", err)
	}

	// Verify exactly 1 pod exists
	if len(pods.Items) != 1 {
		return fmt.Errorf("expected exactly 1 pod for service %s/%s, but found %d pods", namespace, serviceName, len(pods.Items))
	}

	// Check if all pods are running and ready
	runningCount := 0
	for _, pod := range pods.Items {
		if pod.Status.Phase == "Running" {
			// Also check if all containers are ready
			allContainersReady := true
			for _, containerStatus := range pod.Status.ContainerStatuses {
				if !containerStatus.Ready {
					allContainersReady = false
					break
				}
			}
			if allContainersReady {
				runningCount++
			}
		}
	}

	// All pods must be running and ready (and we already verified count is 1)
	if runningCount != len(pods.Items) {
		return fmt.Errorf("not all pods are running for service %s/%s: %d/%d pods ready", namespace, serviceName, runningCount, len(pods.Items))
	}

	if verbose {
		fmt.Printf("[Helper] Service %s/%s has all %d pod(s) running and ready\n",
			namespace, serviceName, len(pods.Items))
	}

	return nil
}

// StartPortForward starts port forwarding to a service by finding a pod behind it
// The ports parameter should be in format "localPort:servicePort" (e.g., "8080:80")
// Note: Kubernetes API doesn't support port-forward directly to services, only to pods.
// This function mimics kubectl's behavior by finding a pod behind the service.
// Returns a stop function that should be called to clean up the port forwarding.
func StartPortForward(ctx context.Context, client *kubernetes.Clientset, restConfig *rest.Config, namespace, service, ports string, verbose bool) (func(), error) {
	// Parse ports (e.g., "8080:80" -> local=8080, service=80)
	portParts := strings.Split(ports, ":")
	if len(portParts) != 2 {
		return nil, fmt.Errorf("invalid port format: %s (expected format: localPort:servicePort)", ports)
	}
	localPort := portParts[0]
	servicePort := portParts[1]

	if verbose {
		fmt.Printf("[Helper] Starting port-forward to service %s/%s (%s:%s)\n", namespace, service, localPort, servicePort)
	}

	// Get the service to find its selector
	svc, err := client.CoreV1().Services(namespace).Get(ctx, service, metav1.GetOptions{})
	if err != nil {
		return nil, fmt.Errorf("failed to get service: %w", err)
	}

	// Build label selector from service selector
	var selectorParts []string
	for key, value := range svc.Spec.Selector {
		selectorParts = append(selectorParts, fmt.Sprintf("%s=%s", key, value))
	}
	labelSelector := strings.Join(selectorParts, ",")

	// Find pods matching the service selector
	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return nil, fmt.Errorf("failed to list pods for service: %w", err)
	}

	if len(pods.Items) == 0 {
		return nil, fmt.Errorf("no pods found for service %s/%s", namespace, service)
	}

	// Use the first running pod
	var targetPod *corev1.Pod
	for i := range pods.Items {
		pod := &pods.Items[i]
		if pod.Status.Phase == corev1.PodRunning {
			targetPod = pod
			break
		}
	}

	if targetPod == nil {
		return nil, fmt.Errorf("no running pods found for service %s/%s", namespace, service)
	}

	if verbose {
		fmt.Printf("[Helper] Found running pod: %s\n", targetPod.Name)
	}

	// Map service port to container port
	var containerPort string
	for _, port := range svc.Spec.Ports {
		if fmt.Sprintf("%d", port.Port) == servicePort {
			containerPort = fmt.Sprintf("%d", port.TargetPort.IntVal)
			if port.TargetPort.IntVal == 0 {
				// TargetPort is a named port, use the port number directly
				containerPort = servicePort
			}
			break
		}
	}
	if containerPort == "" {
		containerPort = servicePort // fallback to service port
	}

	// Create SPDY transport
	transport, upgrader, err := spdy.RoundTripperFor(restConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create SPDY transport: %w", err)
	}

	// Build the URL for port forwarding to pod
	url := client.CoreV1().RESTClient().Post().
		Resource("pods").
		Namespace(namespace).
		Name(targetPod.Name).
		SubResource("portforward").
		URL()

	// Create dialer
	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", url)

	// Setup channels
	stopChan := make(chan struct{}, 1)
	readyChan := make(chan struct{})
	out := io.Discard
	errOut := io.Discard
	if verbose {
		out = os.Stdout
		errOut = os.Stderr
	}

	// Create port forwarder (forward localPort to containerPort on the pod)
	forwarder, err := portforward.New(dialer, []string{fmt.Sprintf("%s:%s", localPort, containerPort)}, stopChan, readyChan, out, errOut)
	if err != nil {
		return nil, fmt.Errorf("failed to create port forwarder: %w", err)
	}

	// Start port forwarding in background
	go func() {
		if err := forwarder.ForwardPorts(); err != nil {
			if verbose {
				fmt.Printf("[Helper] Port forwarding error: %v\n", err)
			}
		}
	}()

	// Wait for ready or timeout
	select {
	case <-readyChan:
		if verbose {
			fmt.Printf("[Helper] Port forwarding is ready\n")
		}
		// Return stop function
		stopFunc := func() {
			if verbose {
				fmt.Printf("[Helper] Stopping port forwarding to %s/%s\n", namespace, service)
			}
			close(stopChan)
		}
		return stopFunc, nil
	case <-time.After(30 * time.Second):
		close(stopChan)
		return nil, fmt.Errorf("timeout waiting for port forward to be ready")
	case <-ctx.Done():
		close(stopChan)
		return nil, ctx.Err()
	}
}
