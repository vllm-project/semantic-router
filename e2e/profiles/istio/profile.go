package istio

import (
	"context"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	// Istio Configuration
	istioVersionDefault = "1.28.0"       // Default Istio version to install
	istioNamespace      = "istio-system" // Istio control plane namespace
	istioIngressGateway = "istio-ingressgateway"

	// Semantic Router Configuration
	semanticRouterNamespace  = "vllm-semantic-router-system" // Namespace for semantic router
	semanticRouterDeployment = "semantic-router"
	semanticRouterService    = "semantic-router"

	// Demo LLM Configuration
	demoLLMDeployment = "vllm-llama3-8b-instruct" // Demo LLM deployment name
	demoLLMService    = "vllm-llama3-8b-instruct" // Demo LLM service name

	// Timeouts
	timeoutIstioInstall         = 5 * time.Minute
	timeoutSemanticRouterDeploy = 20 * time.Minute // Model downloads can take 15-20 minutes
	timeoutDemoLLMDeploy        = 10 * time.Minute
	timeoutSidecarInjection     = 2 * time.Minute
	timeoutGatewayReady         = 5 * time.Minute
	timeoutStabilization        = 60 * time.Second
	retryInterval               = 5 * time.Second
)

// Profile implements the Istio test profile
type Profile struct {
	verbose      bool
	istioVersion string
}

// NewProfile creates a new Istio profile
func NewProfile() *Profile {
	istioVersion := os.Getenv("ISTIO_VERSION")
	if istioVersion == "" {
		istioVersion = istioVersionDefault
	}

	return &Profile{
		istioVersion: istioVersion,
	}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "istio"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return fmt.Sprintf("Tests Semantic Router with Istio service mesh (version: %s)", p.istioVersion)
}

// Setup deploys all required components for Istio testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Istio test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Track what we've deployed for cleanup on error
	var (
		istioInstalled          bool
		namespaceConfigured     bool
		semanticRouterDeployed  bool
		envoyGatewayDeployed    bool
		envoyAIGatewayDeployed  bool
		demoLLMDeployed         bool
		gatewayResourcesCreated bool
	)

	// Ensure cleanup on error
	defer func() {
		if r := recover(); r != nil {
			p.log("Panic during setup, cleaning up...")
			p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, semanticRouterDeployed, envoyGatewayDeployed, envoyAIGatewayDeployed, demoLLMDeployed, gatewayResourcesCreated)
			panic(r) // Re-panic after cleanup
		}
	}()

	// Get Istio version from env var or use default
	istioVersion := os.Getenv("ISTIO_VERSION")
	if istioVersion == "" {
		istioVersion = istioVersionDefault
	}

	// Step 1: Install Istio control plane
	p.log("Step 1/9: Installing Istio control plane (version: %s)", istioVersion)
	if err := p.installIstio(ctx, opts); err != nil {
		return fmt.Errorf("failed to install Istio: %w", err)
	}
	istioInstalled = true

	// Step 2: Configure namespace with sidecar injection
	p.log("Step 2/9: Configuring namespace for sidecar injection")
	if err := p.configureNamespace(ctx, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, false, false, false, false, false, false)
		return fmt.Errorf("failed to configure namespace: %w", err)
	}
	namespaceConfigured = true

	// Step 3: Deploy Semantic Router
	p.log("Step 3/9: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, false, false, false, false, false)
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}
	semanticRouterDeployed = true

	// Step 4: Deploy Envoy Gateway
	p.log("Step 4/9: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, semanticRouterDeployed, false, false, false, false)
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}
	envoyGatewayDeployed = true

	// Step 5: Deploy Envoy AI Gateway
	p.log("Step 5/9: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, semanticRouterDeployed, envoyGatewayDeployed, false, false, false)
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}
	envoyAIGatewayDeployed = true

	// Step 6: Deploy Demo LLM backend
	p.log("Step 6/9: Deploying Demo LLM backend")
	if err := p.deployDemoLLM(ctx, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, semanticRouterDeployed, envoyGatewayDeployed, envoyAIGatewayDeployed, false, false)
		return fmt.Errorf("failed to deploy demo LLM: %w", err)
	}
	demoLLMDeployed = true

	// Step 7: Deploy Gateway API Resources
	p.log("Step 7/9: Deploying Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, semanticRouterDeployed, envoyGatewayDeployed, envoyAIGatewayDeployed, demoLLMDeployed, false)
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}
	gatewayResourcesCreated = true

	// Step 8: Create Istio resources (optional - for Istio-specific tests)
	p.log("Step 8/9: Creating Istio resources for service mesh testing")
	if err := p.createIstioResources(ctx, opts); err != nil {
		p.log("Warning: Failed to create Istio resources (non-critical): %v", err)
		// Don't fail - Istio resources are for mesh tests only
	}

	// Step 9: Verify environment is ready
	p.log("Step 9/9: Verifying environment")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		p.log("ERROR: Environment verification failed: %v", err)
		p.cleanupPartialDeployment(ctx, opts, istioInstalled, namespaceConfigured, semanticRouterDeployed, envoyGatewayDeployed, envoyAIGatewayDeployed, demoLLMDeployed, gatewayResourcesCreated)
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Istio test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Istio test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order
	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Cleaning up Demo LLM")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml")

	p.log("Uninstalling Envoy AI Gateway")
	deployer.Uninstall(ctx, "aieg", "envoy-ai-gateway-system")
	deployer.Uninstall(ctx, "aieg-crd", "envoy-ai-gateway-system")

	p.log("Uninstalling Envoy Gateway")
	deployer.Uninstall(ctx, "eg", "envoy-gateway-system")

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, semanticRouterDeployment, semanticRouterNamespace)

	p.log("Removing sidecar injection label from namespace")
	p.removeSidecarInjection(ctx, opts)

	p.log("Uninstalling Istio")
	p.uninstallIstio(ctx, opts)

	p.log("Istio test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// Istio-specific functionality tests
		// These validate Istio integration: sidecar injection, traffic routing,
		// mTLS, and observability features
		"istio-sidecar-health-check",
		"istio-traffic-routing",
		"istio-mtls-verification",
		"istio-tracing-observability",

		// Common functionality tests (through Istio Gateway)
		// These validate that Semantic Router features work correctly when
		// deployed with Istio service mesh and routed through Istio Gateway
		"chat-completions-request",
		"chat-completions-stress-request",

		// Classification and routing tests
		"domain-classify",

		// Feature tests
		"semantic-cache",
		"pii-detection",
		"jailbreak-detection",

		// Signal-Decision engine tests
		"decision-priority-selection", // Priority-based routing
		"plugin-chain-execution",      // Plugin ordering and blocking
		"rule-condition-logic",        // AND/OR operators
		"decision-fallback-behavior",  // Fallback to default
		"keyword-routing",             // Keyword-based decisions
		"plugin-config-variations",    // Plugin configuration testing

		// Load tests
		"chat-completions-progressive-stress",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		Namespace:     "envoy-gateway-system",
		PortMapping:   "8080:80",
	}
}

// installIstio installs Istio control plane using Helm charts
func (p *Profile) installIstio(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Installing Istio with Helm (version: %s)", p.istioVersion)

	deployer := helm.NewDeployer(opts.KubeConfig, p.verbose)

	// Step 1: Install Istio base (CRDs)
	p.log("Installing Istio base (CRDs)...")
	baseOpts := helm.InstallOptions{
		ReleaseName: "istio-base",
		Chart:       fmt.Sprintf("https://istio-release.storage.googleapis.com/charts/base-%s.tgz", p.istioVersion),
		Namespace:   istioNamespace,
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, baseOpts); err != nil {
		return fmt.Errorf("failed to install Istio base: %w", err)
	}

	// Step 2: Install Istiod (control plane)
	p.log("Installing Istiod (control plane)...")
	istiodOpts := helm.InstallOptions{
		ReleaseName: "istiod",
		Chart:       fmt.Sprintf("https://istio-release.storage.googleapis.com/charts/istiod-%s.tgz", p.istioVersion),
		Namespace:   istioNamespace,
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, istiodOpts); err != nil {
		return fmt.Errorf("failed to install Istiod: %w", err)
	}

	// Wait for istiod to be ready
	p.log("Waiting for istiod to be ready...")
	if err := p.waitForDeployment(ctx, opts, istioNamespace, "istiod", timeoutIstioInstall); err != nil {
		return err
	}

	// Step 3: Install Istio Ingress Gateway
	p.log("Installing Istio Ingress Gateway...")
	gatewayOpts := helm.InstallOptions{
		ReleaseName: "istio-ingressgateway",
		Chart:       fmt.Sprintf("https://istio-release.storage.googleapis.com/charts/gateway-%s.tgz", p.istioVersion),
		Namespace:   istioNamespace,
		Wait:        false, // Don't wait for LoadBalancer (never gets EXTERNAL-IP in Kind)
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, gatewayOpts); err != nil {
		return fmt.Errorf("failed to install Istio Ingress Gateway: %w", err)
	}

	// Wait for ingress gateway deployment to be ready (verifies pod is Running)
	p.log("Waiting for Istio Ingress Gateway to be ready...")
	return p.waitForDeployment(ctx, opts, istioNamespace, istioIngressGateway, timeoutGatewayReady)
}

// configureNamespace configures the namespace for automatic sidecar injection
func (p *Profile) configureNamespace(ctx context.Context, opts *framework.SetupOptions) error {
	// Create namespace if it doesn't exist
	p.log("Creating namespace: %s", semanticRouterNamespace)
	createCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"create", "namespace", semanticRouterNamespace)

	if p.verbose {
		createCmd.Stdout = os.Stdout
		createCmd.Stderr = os.Stderr
	}

	if err := createCmd.Run(); err != nil {
		p.log("Warning: Namespace creation failed (may already exist): %v", err)
	}

	// Label namespace for sidecar injection
	p.log("Enabling automatic sidecar injection for namespace: %s", semanticRouterNamespace)
	labelCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"label", "namespace", semanticRouterNamespace,
		"istio-injection=enabled",
		"--overwrite")

	if p.verbose {
		labelCmd.Stdout = os.Stdout
		labelCmd.Stderr = os.Stderr
	}

	if err := labelCmd.Run(); err != nil {
		return fmt.Errorf("failed to label namespace for sidecar injection: %w", err)
	}

	return nil
}

// deploySemanticRouter deploys Semantic Router via Helm
func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Use AI-Gateway values file (no explicit vllm_endpoints needed - uses AIServiceBackend CRDs)
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "e2e/profiles/ai-gateway/values.yaml"

	// Deploy Semantic Router with AI-Gateway config + Istio sidecar injection
	installOpts := helm.InstallOptions{
		ReleaseName: semanticRouterDeployment,
		Chart:       chartPath,
		Namespace:   "vllm-semantic-router-system", // Use standard namespace
		ValuesFiles: []string{valuesFile},
		Set: map[string]string{
			"image.repository": "ghcr.io/vllm-project/semantic-router/extproc",
			"image.tag":        opts.ImageTag,
			"image.pullPolicy": "Never", // Use local image
			// Sidecar injection is automatic via namespace label (istio-injection=enabled)
		},
		Wait:    true,
		Timeout: "30m", // Model downloads can take 15-20 minutes
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	// Wait for deployment to be ready with sidecar injected
	p.log("Waiting for Semantic Router deployment to be ready...")
	if err := deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", semanticRouterDeployment, timeoutSemanticRouterDeploy); err != nil {
		return err
	}

	// Verify sidecar injection
	p.log("Verifying Istio sidecar injection...")
	return p.verifySidecarInjection(ctx, opts)
}

// deployEnvoyGateway deploys Envoy Gateway for ExtProc protocol support
func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Version:     "v1.6.0",
		ValuesFiles: []string{"https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml"},
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-gateway-system", "envoy-gateway", 10*time.Minute)
}

// deployEnvoyAIGateway deploys Envoy AI Gateway for AI-specific routing
func (p *Profile) deployEnvoyAIGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	// Install AI Gateway CRDs
	crdOpts := helm.InstallOptions{
		ReleaseName: "aieg-crd",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-crds-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, crdOpts); err != nil {
		return err
	}

	// Install AI Gateway
	installOpts := helm.InstallOptions{
		ReleaseName: "aieg",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     "10m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-ai-gateway-system", "ai-gateway-controller", 10*time.Minute)
}

// deployGatewayResources deploys Gateway API resources (Gateway, HTTPRoute, etc.)
func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Applying Gateway API resources...")

	// Apply base-model (vLLM backend with AIServiceBackend)
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	// Apply Gateway API resources (Gateway, HTTPRoute, etc.)
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	// Wait for Envoy Gateway pods to be running (don't wait for LoadBalancer IP in Kind)
	p.log("Waiting for Envoy Gateway service pods to be ready...")

	// Create Kubernetes client
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kube client: %w", err)
	}

	// Wait for Envoy Gateway service pods to be ready (same approach as AI-Gateway profile)
	labelSelector := "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"

	retryTimeout := 5 * time.Minute
	retryInterval := 5 * time.Second
	startTime := time.Now()

	for {
		envoyService, err := helpers.GetEnvoyServiceName(ctx, client, labelSelector, p.verbose)
		if err == nil {
			// Verify pods are running
			if podErr := helpers.VerifyServicePodsRunning(ctx, client, "envoy-gateway-system", envoyService, p.verbose); podErr == nil {
				p.log("✓ Envoy Gateway service is ready: %s", envoyService)
				return nil
			}
		}

		if time.Since(startTime) >= retryTimeout {
			return fmt.Errorf("timeout waiting for Envoy Gateway service pods to be ready")
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(retryInterval):
			// Continue retrying
		}
	}
}

// getServiceClusterIP retrieves the ClusterIP of a Kubernetes service
func (p *Profile) getServiceClusterIP(ctx context.Context, opts *framework.SetupOptions, serviceName, namespace string) (string, error) {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "svc", serviceName, "-n", namespace,
		"-o", "jsonpath={.spec.clusterIP}")

	output, err := cmd.Output()
	if err != nil {
		return "", fmt.Errorf("failed to get service ClusterIP: %w", err)
	}

	clusterIP := strings.TrimSpace(string(output))
	if clusterIP == "" {
		return "", fmt.Errorf("ClusterIP is empty for service %s/%s", namespace, serviceName)
	}

	// Validate it's a valid IP address
	if net.ParseIP(clusterIP) == nil {
		return "", fmt.Errorf("invalid ClusterIP format: %s", clusterIP)
	}

	return clusterIP, nil
}

// cleanupGatewayResources cleans up Gateway API resources
func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) {
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/httproute.yaml")
	p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/gateway.yaml")
	// Note: base-model.yaml is cleaned up separately in Teardown
}

// createTempValuesFile creates a temporary values file with the ClusterIP injected
func (p *Profile) createTempValuesFile(clusterIP string) (string, error) {
	// Read the base values file
	valuesPath := "e2e/profiles/istio/values.yaml"
	content, err := os.ReadFile(valuesPath)
	if err != nil {
		return "", fmt.Errorf("failed to read values file: %w", err)
	}

	// Replace the placeholder with actual ClusterIP
	modifiedContent := strings.ReplaceAll(string(content), "PLACEHOLDER_CLUSTERIP", clusterIP)

	// Create a temporary file
	tempFile, err := os.CreateTemp("", "istio-values-*.yaml")
	if err != nil {
		return "", fmt.Errorf("failed to create temp file: %w", err)
	}
	defer tempFile.Close()

	// Write the modified content
	if _, err := tempFile.WriteString(modifiedContent); err != nil {
		os.Remove(tempFile.Name())
		return "", fmt.Errorf("failed to write temp file: %w", err)
	}

	if p.verbose {
		p.log("Created temporary values file: %s", tempFile.Name())
	}

	return tempFile.Name(), nil
}

// createIstioResources creates Istio Gateway and VirtualService
func (p *Profile) createIstioResources(ctx context.Context, opts *framework.SetupOptions) error {
	// Create Gateway, VirtualService, and DestinationRule as a single YAML
	gatewayYAML := `apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: semantic-router-gateway
  namespace: ` + semanticRouterNamespace + `
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: semantic-router
  namespace: ` + semanticRouterNamespace + `
spec:
  hosts:
  - "*"
  gateways:
  - semantic-router-gateway
  http:
  - match:
    - uri:
        prefix: /v1
    route:
    - destination:
        host: ` + semanticRouterService + `
        port:
          number: 8080
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: semantic-router
  namespace: ` + semanticRouterNamespace + `
spec:
  host: ` + semanticRouterService + `
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
`

	// Apply resources using kubectl apply with stdin
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"apply", "-f", "-")
	cmd.Stdin = strings.NewReader(gatewayYAML)

	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create Istio resources: %w", err)
	}

	// Wait for Istio ingress gateway to be ready
	p.log("Waiting for Istio ingress gateway to be ready...")
	return p.waitForDeployment(ctx, opts, istioNamespace, istioIngressGateway, timeoutGatewayReady)
}

// verifyEnvironment verifies all components are ready
func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	// Verify istiod is running
	p.log("Verifying istiod is running...")
	if err := p.verifyDeployment(ctx, opts, istioNamespace, "istiod"); err != nil {
		return fmt.Errorf("istiod verification failed: %w", err)
	}

	// Verify ingress gateway is running
	p.log("Verifying Istio ingress gateway is running...")
	if err := p.verifyDeployment(ctx, opts, istioNamespace, istioIngressGateway); err != nil {
		return fmt.Errorf("ingress gateway verification failed: %w", err)
	}

	// Verify semantic router is running
	p.log("Verifying Semantic Router is running...")
	if err := p.verifyDeployment(ctx, opts, semanticRouterNamespace, semanticRouterDeployment); err != nil {
		return fmt.Errorf("semantic router verification failed: %w", err)
	}

	// Verify sidecar injection
	p.log("Verifying sidecar injection...")
	if err := p.verifySidecarInjection(ctx, opts); err != nil {
		return fmt.Errorf("sidecar injection verification failed: %w", err)
	}

	// Verify semantic router service is actually responding
	p.log("Verifying Semantic Router service health...")
	if err := p.verifyServiceHealth(ctx, opts); err != nil {
		p.log("Warning: Service health check failed: %v", err)
		p.log("This may cause traffic routing tests to fail")
	}

	// Allow time for everything to stabilize
	p.log("Allowing %v for environment stabilization...", timeoutStabilization)
	time.Sleep(timeoutStabilization)

	p.log("Environment verification complete")
	return nil
}

// verifyServiceHealth checks if the semantic router service is actually responding
func (p *Profile) verifyServiceHealth(ctx context.Context, opts *framework.SetupOptions) error {
	// Check if all containers in the semantic-router pod are ready
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "pods",
		"-n", semanticRouterNamespace,
		"-l", "app.kubernetes.io/name=semantic-router",
		"-o", "jsonpath={.items[*].status.containerStatuses[*].ready}")

	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to check pod readiness: %w (output: %s)", err, string(output))
	}

	readyStatus := strings.TrimSpace(string(output))
	if !strings.Contains(readyStatus, "true") {
		return fmt.Errorf("semantic-router pod containers not all ready: %s", readyStatus)
	}

	// Check that all containers report ready (expecting "true true" for main + sidecar)
	readyCount := strings.Count(readyStatus, "true")
	if readyCount < 2 {
		return fmt.Errorf("expected 2 ready containers (main + sidecar), got %d", readyCount)
	}

	p.log("Semantic Router service health check passed: %d/2 containers ready", readyCount)

	// Give a bit more time for the service to be fully ready after containers report ready
	p.log("Waiting additional 10s for service to be fully ready...")
	time.Sleep(10 * time.Second)

	return nil
}

// verifySidecarInjection verifies that Istio sidecar is injected
func (p *Profile) verifySidecarInjection(ctx context.Context, opts *framework.SetupOptions) error {
	// Get pod and check for istio-proxy container
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "pods",
		"-n", semanticRouterNamespace,
		"-l", "app.kubernetes.io/name=semantic-router",
		"-o", "jsonpath={.items[0].spec.containers[*].name}")

	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get pod containers: %w", err)
	}

	containers := string(output)
	if !contains(containers, "istio-proxy") {
		return fmt.Errorf("istio-proxy sidecar not found in pod. Containers: %s", containers)
	}

	p.log("✓ Istio sidecar successfully injected")
	return nil
}

// Helper functions

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[istio] "+format+"\n", args...)
	}
}

func (p *Profile) waitForDeployment(ctx context.Context, opts *framework.SetupOptions, namespace, name string, timeout time.Duration) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	return deployer.WaitForDeployment(ctx, namespace, name, timeout)
}

func (p *Profile) verifyDeployment(ctx context.Context, opts *framework.SetupOptions, namespace, name string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "deployment", name,
		"-n", namespace,
		"-o", "jsonpath={.status.readyReplicas}")

	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get deployment status: %w", err)
	}

	if string(output) == "0" || string(output) == "" {
		return fmt.Errorf("deployment %s/%s has no ready replicas", namespace, name)
	}

	p.log("✓ Deployment %s/%s is ready", namespace, name)
	return nil
}

func (p *Profile) cleanupPartialDeployment(ctx context.Context, opts *framework.SetupOptions, istioInstalled, namespaceConfigured, semanticRouterDeployed, envoyGatewayDeployed, envoyAIGatewayDeployed, demoLLMDeployed, gatewayResourcesCreated bool) {
	p.log("Cleaning up partial deployment...")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	teardownOpts := &framework.TeardownOptions{
		KubeClient:  opts.KubeClient,
		KubeConfig:  opts.KubeConfig,
		ClusterName: opts.ClusterName,
		Verbose:     opts.Verbose,
	}

	// Cleanup in reverse order
	if gatewayResourcesCreated {
		p.log("Cleaning up Gateway API resources")
		p.cleanupGatewayResources(ctx, teardownOpts)
	}

	if demoLLMDeployed {
		p.log("Cleaning up Demo LLM")
		p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml")
	}

	if envoyAIGatewayDeployed {
		deployer.Uninstall(ctx, "aieg", "envoy-ai-gateway-system")
		deployer.Uninstall(ctx, "aieg-crd", "envoy-ai-gateway-system")
	}

	if envoyGatewayDeployed {
		deployer.Uninstall(ctx, "eg", "envoy-gateway-system")
	}

	if semanticRouterDeployed {
		deployer.Uninstall(ctx, semanticRouterDeployment, semanticRouterNamespace)
	}

	if namespaceConfigured {
		p.removeSidecarInjection(ctx, teardownOpts)
	}

	if istioInstalled {
		p.uninstallIstio(ctx, teardownOpts)
	}
}

func (p *Profile) cleanupIstioResources(ctx context.Context, opts *framework.TeardownOptions) {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"delete", "gateway,virtualservice,destinationrule",
		"-n", semanticRouterNamespace,
		"--all",
		"--ignore-not-found")

	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		p.log("Warning: Failed to delete Istio resources: %v", err)
	}
}

func (p *Profile) removeSidecarInjection(ctx context.Context, opts *framework.TeardownOptions) {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"label", "namespace", semanticRouterNamespace,
		"istio-injection-")

	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		p.log("Warning: Failed to remove sidecar injection label: %v", err)
	}
}

func (p *Profile) uninstallIstio(ctx context.Context, opts *framework.TeardownOptions) {
	deployer := helm.NewDeployer(opts.KubeConfig, p.verbose)

	// Uninstall in reverse order
	p.log("Uninstalling Istio Ingress Gateway...")
	deployer.Uninstall(ctx, "istio-ingressgateway", istioNamespace)

	p.log("Uninstalling Istiod...")
	deployer.Uninstall(ctx, "istiod", istioNamespace)

	p.log("Uninstalling Istio base...")
	deployer.Uninstall(ctx, "istio-base", istioNamespace)

	// Delete istio-system namespace
	p.log("Deleting istio-system namespace...")
	deleteNsCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"delete", "namespace", istioNamespace,
		"--ignore-not-found",
		"--timeout=60s")

	if p.verbose {
		deleteNsCmd.Stdout = os.Stdout
		deleteNsCmd.Stderr = os.Stderr
	}

	if err := deleteNsCmd.Run(); err != nil {
		p.log("Warning: Failed to delete istio-system namespace: %v", err)
	}
}

func contains(s, substr string) bool {
	return len(s) > 0 && len(substr) > 0 && (s == substr || len(s) > len(substr) && (s[:len(substr)] == substr || s[len(s)-len(substr):] == substr || containsMiddle(s, substr)))
}

func containsMiddle(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}

// deployDemoLLM deploys the demo LLM (vLLM backend) for testing
func (p *Profile) deployDemoLLM(ctx context.Context, opts *framework.SetupOptions) error {
	// Note: Demo LLM is now deployed as part of Gateway API resources in deployGatewayResources
	// This function is kept for backward compatibility but does minimal work
	p.log("Demo LLM will be deployed with Gateway API resources")
	return nil
}

// kubectlApply applies a Kubernetes manifest
func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "--server-side", "-f", manifest)
}

// kubectlDelete deletes a Kubernetes manifest
func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "-f", manifest)
}

// runKubectl runs a kubectl command
func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append([]string{"--kubeconfig", kubeConfig}, args...)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}
