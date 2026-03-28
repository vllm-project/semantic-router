package helpers

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"
	"slices"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"k8s.io/client-go/tools/portforward"
	"k8s.io/client-go/transport/spdy"
)

// DeploymentRef identifies a deployment that must be healthy.
type DeploymentRef struct {
	Namespace string
	Name      string
}

var deploymentFailFastReasons = []string{
	"CrashLoopBackOff",
	"CreateContainerConfigError",
	"CreateContainerError",
	"ErrImagePull",
	"ImagePullBackOff",
	"InvalidImageName",
	"RunContainerError",
}

// NewKubeClient builds a clientset from a kubeconfig path.
func NewKubeClient(kubeConfig string) (*kubernetes.Clientset, error) {
	config, err := clientcmd.BuildConfigFromFlags("", kubeConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to build kubeconfig: %w", err)
	}
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return nil, fmt.Errorf("failed to create kube client: %w", err)
	}
	return client, nil
}

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

// WaitForDeploymentReady polls until a deployment becomes healthy.
func WaitForDeploymentReady(ctx context.Context, client *kubernetes.Clientset, namespace, name string, timeout, retryInterval time.Duration, verbose bool) error {
	deadline := time.Now().Add(timeout)
	var lastErr error
	for time.Now().Before(deadline) {
		if failErr := DetectDeploymentFailFastError(ctx, client, namespace, name); failErr != nil {
			return failErr
		}
		lastErr = CheckDeployment(ctx, client, namespace, name, verbose)
		if lastErr == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(retryInterval):
		}
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("timed out waiting for deployment")
	}
	return fmt.Errorf("deployment %s/%s not healthy after %v: %w", namespace, name, timeout, lastErr)
}

// DetectDeploymentFailFastError returns an error when a deployment's pods hit a
// non-recoverable startup state that should fail setup immediately.
func DetectDeploymentFailFastError(ctx context.Context, client *kubernetes.Clientset, namespace, name string) error {
	deployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get deployment for fail-fast check: %w", err)
	}

	selector := metav1.FormatLabelSelector(deployment.Spec.Selector)
	if selector == "" {
		return nil
	}

	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return fmt.Errorf("failed to list deployment pods for fail-fast check: %w", err)
	}

	for _, pod := range pods.Items {
		if failErr := failFastErrorForPod(pod); failErr != nil {
			return fmt.Errorf("deployment %s/%s failed early: %w", namespace, name, failErr)
		}
	}

	return nil
}

func failFastErrorForPod(pod corev1.Pod) error {
	for _, status := range pod.Status.InitContainerStatuses {
		if err := failFastErrorForContainerStatus(pod.Name, status); err != nil {
			return err
		}
	}

	for _, status := range pod.Status.ContainerStatuses {
		if err := failFastErrorForContainerStatus(pod.Name, status); err != nil {
			return err
		}
	}

	return nil
}

func failFastErrorForContainerStatus(podName string, status corev1.ContainerStatus) error {
	if waiting := status.State.Waiting; waiting != nil && slices.Contains(deploymentFailFastReasons, waiting.Reason) {
		return fmt.Errorf(
			"pod %s container %s waiting: %s (%s)",
			podName,
			status.Name,
			waiting.Reason,
			strings.TrimSpace(waiting.Message),
		)
	}

	terminated := status.State.Terminated
	if terminated == nil {
		return nil
	}

	if terminated.ExitCode == 0 {
		return nil
	}

	return fmt.Errorf(
		"pod %s container %s restarted after termination: reason=%s exit_code=%d message=%s",
		podName,
		status.Name,
		terminated.Reason,
		terminated.ExitCode,
		strings.TrimSpace(terminated.Message),
	)
}

// VerifyDeployments checks a set of deployments for readiness.
func VerifyDeployments(ctx context.Context, client *kubernetes.Clientset, deployments []DeploymentRef, verbose bool) error {
	for _, deployment := range deployments {
		if err := CheckDeployment(ctx, client, deployment.Namespace, deployment.Name, verbose); err != nil {
			return fmt.Errorf("%s/%s: %w", deployment.Namespace, deployment.Name, err)
		}
	}
	return nil
}

// GetEnvoyServiceName finds the Envoy service name in the envoy-gateway-system namespace
// using label selectors to match the Gateway-owned service
// Deprecated: Use GetServiceByLabelInNamespace for more flexibility
func GetEnvoyServiceName(ctx context.Context, client *kubernetes.Clientset, labelSelector string, verbose bool) (string, error) {
	return GetServiceByLabelInNamespace(ctx, client, "envoy-gateway-system", labelSelector, verbose)
}

// GetServiceByLabelInNamespace finds a service by label selector in a specific namespace
func GetServiceByLabelInNamespace(ctx context.Context, client *kubernetes.Clientset, namespace string, labelSelector string, verbose bool) (string, error) {
	services, err := client.CoreV1().Services(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: labelSelector,
	})
	if err != nil {
		return "", fmt.Errorf("failed to list services with selector %s: %w", labelSelector, err)
	}

	if len(services.Items) == 0 {
		return "", fmt.Errorf("no service found with selector %s in %s namespace", labelSelector, namespace)
	}

	// Return the first matching service (should only be one)
	serviceName := services.Items[0].Name
	if verbose {
		fmt.Printf("[Helper] Found service: %s (matched by labels: %s in namespace: %s)\n", serviceName, labelSelector, namespace)
	}

	return serviceName, nil
}

// WaitForServiceByLabelWithReadyPods resolves a service by label selector and waits until its backing pods are ready.
func WaitForServiceByLabelWithReadyPods(
	ctx context.Context,
	client *kubernetes.Clientset,
	namespace string,
	labelSelector string,
	timeout time.Duration,
	retryInterval time.Duration,
	verbose bool,
	logf func(format string, args ...interface{}),
) (string, error) {
	deadline := time.Now().Add(timeout)
	var lastErr error

	for time.Now().Before(deadline) {
		serviceName, err := GetServiceByLabelInNamespace(ctx, client, namespace, labelSelector, verbose)
		if err == nil {
			if podErr := VerifyServicePodsRunning(ctx, client, namespace, serviceName, verbose); podErr == nil {
				return serviceName, nil
			} else {
				lastErr = fmt.Errorf("service pods not ready: %w", podErr)
			}
		} else {
			lastErr = err
		}

		if verbose && logf != nil {
			logf(
				"Service with selector %s in %s not ready, retrying in %v... (elapsed: %v)",
				labelSelector,
				namespace,
				retryInterval,
				timeout-time.Until(deadline).Round(time.Second),
			)
		}

		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(retryInterval):
		}
	}

	return "", fmt.Errorf(
		"failed to get service with selector %s in %s after %v: %w",
		labelSelector,
		namespace,
		timeout,
		lastErr,
	)
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
	localPort, servicePort, err := ParsePortMapping(ports)
	if err != nil {
		return nil, err
	}
	if verbose {
		fmt.Printf("[Helper] Starting port-forward to service %s/%s (%s:%s)\n", namespace, service, localPort, servicePort)
	}

	svc, targetPod, err := resolvePortForwardTarget(ctx, client, namespace, service, verbose)
	if err != nil {
		return nil, err
	}

	containerPort := resolveContainerPort(svc, servicePort)
	forwarder, stopChan, readyChan, err := newPortForwarder(
		client,
		restConfig,
		namespace,
		targetPod.Name,
		localPort,
		containerPort,
		verbose,
	)
	if err != nil {
		return nil, err
	}

	go func() {
		if err := forwarder.ForwardPorts(); err != nil {
			if verbose {
				fmt.Printf("[Helper] Port forwarding error: %v\n", err)
			}
		}
	}()

	return waitForPortForwardReady(ctx, namespace, service, stopChan, readyChan, verbose)
}

// ParsePortMapping parses a legacy local:service port-forward mapping.
func ParsePortMapping(ports string) (string, string, error) {
	portParts := strings.Split(ports, ":")
	if len(portParts) != 2 {
		return "", "", fmt.Errorf("invalid port format: %s (expected format: localPort:servicePort)", ports)
	}
	return portParts[0], portParts[1], nil
}

func resolvePortForwardTarget(
	ctx context.Context,
	client *kubernetes.Clientset,
	namespace string,
	service string,
	verbose bool,
) (*corev1.Service, *corev1.Pod, error) {
	svc, err := client.CoreV1().Services(namespace).Get(ctx, service, metav1.GetOptions{})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get service: %w", err)
	}

	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{
		LabelSelector: serviceSelector(svc),
	})
	if err != nil {
		return nil, nil, fmt.Errorf("failed to list pods for service: %w", err)
	}
	if len(pods.Items) == 0 {
		return nil, nil, fmt.Errorf("no pods found for service %s/%s", namespace, service)
	}

	targetPod, err := firstRunningPod(pods.Items, namespace, service)
	if err != nil {
		return nil, nil, err
	}
	if verbose {
		fmt.Printf("[Helper] Found running pod: %s\n", targetPod.Name)
	}
	return svc, targetPod, nil
}

func serviceSelector(svc *corev1.Service) string {
	var selectorParts []string
	for key, value := range svc.Spec.Selector {
		selectorParts = append(selectorParts, fmt.Sprintf("%s=%s", key, value))
	}
	return strings.Join(selectorParts, ",")
}

func firstRunningPod(pods []corev1.Pod, namespace, service string) (*corev1.Pod, error) {
	for i := range pods {
		pod := &pods[i]
		if pod.Status.Phase == corev1.PodRunning {
			return pod, nil
		}
	}
	return nil, fmt.Errorf("no running pods found for service %s/%s", namespace, service)
}

func resolveContainerPort(svc *corev1.Service, servicePort string) string {
	for _, port := range svc.Spec.Ports {
		if fmt.Sprintf("%d", port.Port) != servicePort {
			continue
		}
		if port.TargetPort.IntVal == 0 {
			return servicePort
		}
		return fmt.Sprintf("%d", port.TargetPort.IntVal)
	}
	return servicePort
}

func newPortForwarder(
	client *kubernetes.Clientset,
	restConfig *rest.Config,
	namespace string,
	podName string,
	localPort string,
	containerPort string,
	verbose bool,
) (*portforward.PortForwarder, chan struct{}, chan struct{}, error) {
	transport, upgrader, err := spdy.RoundTripperFor(restConfig)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create SPDY transport: %w", err)
	}

	url := client.CoreV1().RESTClient().Post().
		Resource("pods").
		Namespace(namespace).
		Name(podName).
		SubResource("portforward").
		URL()
	dialer := spdy.NewDialer(upgrader, &http.Client{Transport: transport}, "POST", url)

	stopChan := make(chan struct{}, 1)
	readyChan := make(chan struct{})
	out, errOut := portForwardStreams(verbose)

	forwarder, err := portforward.New(
		dialer,
		[]string{fmt.Sprintf("%s:%s", localPort, containerPort)},
		stopChan,
		readyChan,
		out,
		errOut,
	)
	if err != nil {
		return nil, nil, nil, fmt.Errorf("failed to create port forwarder: %w", err)
	}
	return forwarder, stopChan, readyChan, nil
}

func portForwardStreams(verbose bool) (io.Writer, io.Writer) {
	if verbose {
		return os.Stdout, os.Stderr
	}
	return io.Discard, io.Discard
}

func waitForPortForwardReady(
	ctx context.Context,
	namespace string,
	service string,
	stopChan chan struct{},
	readyChan chan struct{},
	verbose bool,
) (func(), error) {
	select {
	case <-readyChan:
		if verbose {
			fmt.Printf("[Helper] Port forwarding is ready\n")
		}
		return func() {
			if verbose {
				fmt.Printf("[Helper] Stopping port forwarding to %s/%s\n", namespace, service)
			}
			close(stopChan)
		}, nil
	case <-time.After(30 * time.Second):
		close(stopChan)
		return nil, fmt.Errorf("timeout waiting for port forward to be ready")
	case <-ctx.Done():
		close(stopChan)
		return nil, ctx.Err()
	}
}
