package productionstack

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

const (
	// Profile constants
	profileName = "production-stack"

	// Namespace constants
	namespaceSemanticRouter = "vllm-semantic-router-system"
	namespaceEnvoyGateway   = "envoy-gateway-system"
	namespaceAIGateway      = "envoy-ai-gateway-system"
	namespaceDefault        = "default"

	// Deployment name constants
	deploymentSemanticRouter = "semantic-router"
	deploymentDemoLLM        = "vllm-llama3-8b-instruct"

	// File path constants
	valuesFile           = "e2e/profiles/production-stack/values.yaml"
	baseModelManifest    = "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"
	gatewayAPIManifest   = "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"
	prometheusConfigFile = "e2e/profiles/production-stack/prometheus-config.yaml"

	// Timeout constants
	timeoutDeploymentWait = 30 * time.Minute
	timeoutStableRollout  = 5 * time.Minute
	timeoutTrafficProbe   = 3 * time.Minute
	stableRolloutWindow   = 30 * time.Second
	stabilityPollInterval = 5 * time.Second
	trafficProbeRequests  = 3
)

var resourceManifests = []string{
	baseModelManifest,
	gatewayAPIManifest,
}

var waitDeployments = []helpers.DeploymentRef{
	{Namespace: namespaceDefault, Name: deploymentDemoLLM},
}

// Profile implements the production-stack test profile.
type Profile struct {
	verbose bool
	stack   *gatewaystack.Stack
}

// NewProfile creates a new production-stack profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     profileName,
			SemanticRouterValuesFile: valuesFile,
			SemanticRouterSet: map[string]string{
				"replicaCount": "1",
			},
			ResourceManifests: resourceManifests,
			WaitDeployments:   waitDeployments,
		}),
	}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return profileName
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Semantic Router with Envoy AI Gateway integration (production-stack)"
}

// Setup deploys the shared gateway stack, then adds production-oriented extras.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Production Stack test environment")

	if err := p.stack.Setup(ctx, opts); err != nil {
		return err
	}

	p.log("Scaling deployments for high availability")
	if err := p.scaleDeployments(ctx, opts); err != nil {
		return fmt.Errorf("failed to scale deployments: %w", err)
	}

	p.log("Deploying Prometheus for monitoring")
	if err := p.deployPrometheus(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy prometheus: %w", err)
	}

	p.log("Waiting for production stack to stabilize before HA tests")
	if err := p.waitForStackStability(ctx, opts); err != nil {
		return fmt.Errorf("failed to stabilize production stack: %w", err)
	}

	p.log("Production Stack test environment setup complete")
	return nil
}

// Teardown cleans up production-only resources, then the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Production Stack test environment")

	p.log("Cleaning up Prometheus")
	if err := p.cleanupPrometheus(ctx, opts); err != nil {
		p.log("Warning: failed to cleanup Prometheus resources: %v", err)
	}

	if err := p.stack.Teardown(ctx, opts); err != nil {
		return err
	}

	p.log("Production Stack test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(
		testmatrix.RouterSmoke,
		[]string{
			"multi-replica-health",
			"load-balancing-verification",
			"failover-during-traffic",
			"performance-throughput",
			"resource-utilization-monitoring",
		},
	)
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return p.stack.ServiceConfig()
}

func (p *Profile) scaleDeployments(ctx context.Context, opts *framework.SetupOptions) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	p.log("Scaling semantic-router deployment to 2 replicas")
	if err := p.kubectl(ctx, opts.KubeConfig, "scale", "deployment", deploymentSemanticRouter, "-n", namespaceSemanticRouter, "--replicas=2"); err != nil {
		return fmt.Errorf("failed to scale semantic-router deployment: %w", err)
	}

	if err := deployer.WaitForDeployment(ctx, namespaceSemanticRouter, deploymentSemanticRouter, timeoutDeploymentWait); err != nil {
		return fmt.Errorf("semantic-router deployment not ready after scaling: %w", err)
	}

	p.log("Scaling %s deployment to 2 replicas", deploymentDemoLLM)
	if err := p.kubectl(ctx, opts.KubeConfig, "scale", "deployment", deploymentDemoLLM, "-n", namespaceDefault, "--replicas=2"); err != nil {
		return fmt.Errorf("failed to scale vllm demo deployment: %w", err)
	}

	if err := deployer.WaitForDeployment(ctx, namespaceDefault, deploymentDemoLLM, timeoutDeploymentWait); err != nil {
		return fmt.Errorf("vllm demo deployment not ready after scaling: %w", err)
	}

	return nil
}

func (p *Profile) deployPrometheus(ctx context.Context, opts *framework.SetupOptions) error {
	prometheusDir := "deploy/kubernetes/observability/prometheus"

	if err := p.kubectl(ctx, opts.KubeConfig, "create", "serviceaccount", "prometheus", "-n", namespaceDefault); err != nil {
		p.log("ServiceAccount prometheus may already exist, continuing...")
	}

	if err := p.kubectl(ctx, opts.KubeConfig, "apply", "-f", prometheusDir+"/rbac.yaml", "--server-side"); err != nil {
		return fmt.Errorf("failed to apply prometheus RBAC: %w", err)
	}

	if err := p.kubectl(ctx, opts.KubeConfig, "patch", "clusterrolebinding", "prometheus", "--type", "json", "-p", `[{"op": "replace", "path": "/subjects/0/namespace", "value": "default"}]`); err != nil {
		p.log("Patching ClusterRoleBinding, if it fails we'll continue...")
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceDefault, prometheusDir+"/configmap.yaml"); err != nil {
		return fmt.Errorf("failed to apply prometheus configmap: %w", err)
	}

	updatedConfig, err := os.ReadFile(prometheusConfigFile)
	if err != nil {
		return fmt.Errorf("failed to read prometheus config file: %w", err)
	}

	if err := p.kubectl(ctx, opts.KubeConfig, "patch", "configmap", "prometheus-config", "-n", namespaceDefault, "--type", "merge", "-p", fmt.Sprintf(`{"data":{"prometheus.yml":%q}}`, string(updatedConfig))); err != nil {
		p.log("Warning: Could not update prometheus configmap, using default: %v", err)
	} else {
		p.log("Reloading Prometheus configuration...")
		time.Sleep(2 * time.Second)
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceDefault, prometheusDir+"/pvc.yaml"); err != nil {
		return fmt.Errorf("failed to apply prometheus PVC: %w", err)
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceDefault, prometheusDir+"/deployment.yaml"); err != nil {
		return fmt.Errorf("failed to apply prometheus deployment: %w", err)
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceDefault, prometheusDir+"/service.yaml"); err != nil {
		return fmt.Errorf("failed to apply prometheus service: %w", err)
	}

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	if err := deployer.WaitForDeployment(ctx, namespaceDefault, "prometheus", timeoutDeploymentWait); err != nil {
		return fmt.Errorf("prometheus deployment not ready: %w", err)
	}

	p.log("Waiting for Prometheus to start scraping metrics...")
	time.Sleep(30 * time.Second)

	return nil
}

func (p *Profile) cleanupPrometheus(ctx context.Context, opts *framework.TeardownOptions) error {
	prometheusDir := "deploy/kubernetes/observability/prometheus"
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/service.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/deployment.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/pvc.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/configmap.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/rbac.yaml", "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "serviceaccount", "prometheus", "-n", namespaceDefault, "--ignore-not-found=true")
	return nil
}

func (p *Profile) waitForStackStability(ctx context.Context, opts *framework.SetupOptions) error {
	if opts.KubeClient == nil {
		return fmt.Errorf("kube client is required for stack stabilization")
	}

	if err := p.waitForDeploymentPodsStable(
		ctx,
		opts.KubeClient,
		namespaceSemanticRouter,
		deploymentSemanticRouter,
		2,
		timeoutStableRollout,
		stableRolloutWindow,
	); err != nil {
		return err
	}

	if err := p.waitForDeploymentPodsStable(
		ctx,
		opts.KubeClient,
		namespaceDefault,
		deploymentDemoLLM,
		2,
		timeoutStableRollout,
		stableRolloutWindow,
	); err != nil {
		return err
	}

	return p.waitForGatewayTrafficReady(ctx, opts, timeoutTrafficProbe)
}

func (p *Profile) waitForDeploymentPodsStable(
	ctx context.Context,
	client *kubernetes.Clientset,
	namespace, name string,
	minReady int32,
	timeout, stableWindow time.Duration,
) error {
	deadline := time.Now().Add(timeout)
	var (
		lastErr       error
		lastSignature string
		stableSince   time.Time
	)

	for time.Now().Before(deadline) {
		signature, err := deploymentPodSignature(ctx, client, namespace, name, minReady)
		if err != nil {
			lastErr = err
			lastSignature = ""
			stableSince = time.Time{}
			p.log("Deployment %s/%s not stable yet: %v", namespace, name, err)
		} else {
			lastErr = nil
			if signature != lastSignature {
				lastSignature = signature
				stableSince = time.Now()
				p.log("Observed new stable candidate for %s/%s", namespace, name)
			} else if time.Since(stableSince) >= stableWindow {
				p.log("Deployment %s/%s stayed stable for %s", namespace, name, stableWindow)
				return nil
			}
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(stabilityPollInterval):
		}
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("timed out waiting for stable pod state")
	}
	return fmt.Errorf("deployment %s/%s did not stabilize within %s: %w", namespace, name, timeout, lastErr)
}

func deploymentPodSignature(
	ctx context.Context,
	client *kubernetes.Clientset,
	namespace, name string,
	minReady int32,
) (string, error) {
	deployment, err := client.AppsV1().Deployments(namespace).Get(ctx, name, metav1.GetOptions{})
	if err != nil {
		return "", fmt.Errorf("get deployment: %w", err)
	}
	if deployment.Status.ObservedGeneration < deployment.Generation {
		return "", fmt.Errorf(
			"observed generation %d is behind desired generation %d",
			deployment.Status.ObservedGeneration,
			deployment.Generation,
		)
	}
	if deployment.Status.UpdatedReplicas < minReady {
		return "", fmt.Errorf("updated replicas %d below %d", deployment.Status.UpdatedReplicas, minReady)
	}
	if deployment.Status.ReadyReplicas < minReady {
		return "", fmt.Errorf("ready replicas %d below %d", deployment.Status.ReadyReplicas, minReady)
	}
	if deployment.Status.AvailableReplicas < minReady {
		return "", fmt.Errorf("available replicas %d below %d", deployment.Status.AvailableReplicas, minReady)
	}

	selector := metav1.FormatLabelSelector(deployment.Spec.Selector)
	if selector == "" {
		return "", fmt.Errorf("deployment %s/%s has empty selector", namespace, name)
	}

	pods, err := client.CoreV1().Pods(namespace).List(ctx, metav1.ListOptions{LabelSelector: selector})
	if err != nil {
		return "", fmt.Errorf("list pods: %w", err)
	}

	readyPods := 0
	entries := make([]string, 0, len(pods.Items))
	for _, pod := range pods.Items {
		ready := isRunningAndReady(pod)
		if ready {
			readyPods++
		}
		entries = append(entries, fmt.Sprintf("%s:%t:%d", pod.Name, ready, totalRestarts(pod.Status.ContainerStatuses)))
	}
	sort.Strings(entries)

	if readyPods < int(minReady) {
		return "", fmt.Errorf("only %d ready pod(s) found for selector %s", readyPods, selector)
	}

	return fmt.Sprintf(
		"observed=%d updated=%d ready=%d available=%d pods=%s",
		deployment.Status.ObservedGeneration,
		deployment.Status.UpdatedReplicas,
		deployment.Status.ReadyReplicas,
		deployment.Status.AvailableReplicas,
		strings.Join(entries, ","),
	), nil
}

func isRunningAndReady(pod corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	if len(pod.Status.ContainerStatuses) == 0 {
		return false
	}
	for _, status := range pod.Status.ContainerStatuses {
		if !status.Ready {
			return false
		}
	}
	return true
}

func totalRestarts(statuses []corev1.ContainerStatus) int32 {
	var total int32
	for _, status := range statuses {
		total += status.RestartCount
	}
	return total
}

func (p *Profile) waitForGatewayTrafficReady(
	ctx context.Context,
	opts *framework.SetupOptions,
	timeout time.Duration,
) error {
	deadline := time.Now().Add(timeout)
	var lastErr error

	for attempt := 1; time.Now().Before(deadline); attempt++ {
		lastErr = p.probeGatewayTraffic(ctx, opts)
		if lastErr == nil {
			p.log("Gateway traffic probe succeeded on attempt %d", attempt)
			return nil
		}

		p.log("Gateway traffic probe attempt %d failed: %v", attempt, lastErr)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(stabilityPollInterval):
		}
	}

	if lastErr == nil {
		lastErr = fmt.Errorf("timed out waiting for gateway traffic readiness")
	}
	return fmt.Errorf("gateway traffic did not stabilize within %s: %w", timeout, lastErr)
}

func (p *Profile) probeGatewayTraffic(ctx context.Context, opts *framework.SetupOptions) error {
	if opts.KubeClient == nil {
		return fmt.Errorf("kube client is required for gateway probing")
	}

	svcConfig := p.stack.ServiceConfig()
	serviceName, err := p.gatewayServiceName(ctx, opts, svcConfig)
	if err != nil {
		return err
	}

	restConfig, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("build kube rest config: %w", err)
	}

	localPort, err := availablePort()
	if err != nil {
		return err
	}

	stop, err := helpers.StartPortForward(
		ctx,
		opts.KubeClient,
		restConfig,
		svcConfig.Namespace,
		serviceName,
		fmt.Sprintf("%s:%s", localPort, svcConfig.ServicePort),
		p.verbose,
	)
	if err != nil {
		return fmt.Errorf("start port-forward for %s/%s: %w", svcConfig.Namespace, serviceName, err)
	}
	defer stop()

	time.Sleep(2 * time.Second)

	baseURL := fmt.Sprintf("http://localhost:%s", localPort)
	httpClient := &http.Client{Timeout: 30 * time.Second}
	for i := 1; i <= trafficProbeRequests; i++ {
		if err := sendGatewayProbeRequest(ctx, httpClient, baseURL, i); err != nil {
			return err
		}
	}
	return nil
}

func (p *Profile) gatewayServiceName(
	ctx context.Context,
	opts *framework.SetupOptions,
	svcConfig framework.ServiceConfig,
) (string, error) {
	if svcConfig.Name != "" {
		return svcConfig.Name, nil
	}
	return helpers.GetServiceByLabelInNamespace(
		ctx,
		opts.KubeClient,
		svcConfig.Namespace,
		svcConfig.LabelSelector,
		p.verbose,
	)
}

func availablePort() (string, error) {
	listener, err := net.Listen("tcp", ":0")
	if err != nil {
		return "", fmt.Errorf("allocate local port: %w", err)
	}
	defer func() {
		_ = listener.Close()
	}()

	return fmt.Sprintf("%d", listener.Addr().(*net.TCPAddr).Port), nil
}

func sendGatewayProbeRequest(ctx context.Context, httpClient *http.Client, baseURL string, requestID int) error {
	requestBody := map[string]interface{}{
		"model": "MoM",
		"messages": []map[string]string{
			{"role": "user", "content": fmt.Sprintf("Warm up the production-stack gateway path. Request %d.", requestID)},
		},
		"max_tokens": 16,
	}

	jsonData, err := json.Marshal(requestBody)
	if err != nil {
		return fmt.Errorf("marshal gateway probe request: %w", err)
	}

	req, err := http.NewRequestWithContext(
		ctx,
		http.MethodPost,
		baseURL+"/v1/chat/completions",
		bytes.NewReader(jsonData),
	)
	if err != nil {
		return fmt.Errorf("create gateway probe request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := httpClient.Do(req)
	if err != nil {
		return fmt.Errorf("send gateway probe request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("read gateway probe response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("gateway probe returned %d: %s", resp.StatusCode, truncateProbeBody(string(body), 200))
	}
	return nil
}

func truncateProbeBody(body string, maxLen int) string {
	if len(body) <= maxLen {
		return body
	}
	return body[:maxLen] + "..."
}

func (p *Profile) kubectl(ctx context.Context, kubeConfig string, args ...string) error {
	return p.runKubectl(ctx, kubeConfig, args...)
}

func (p *Profile) kubectlApplyWithNamespace(ctx context.Context, kubeConfig, namespace, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest, "-n", namespace)
}

func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[Production-Stack] "+format+"\n", args...)
	}
}
