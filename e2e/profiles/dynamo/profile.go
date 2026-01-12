package dynamo

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

// Profile implements the Dynamo test profile
type Profile struct {
	verbose bool
}

// NewProfile creates a new Dynamo profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "dynamo"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Semantic Router with Nvidia Dynamo integration (GPU-enabled disaggregated vLLM deployment)"
}

// Setup deploys all required components for Dynamo testing
// Note: GPU setup is handled by the cluster creation in e2e/pkg/cluster/kind.go
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Dynamo test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Nvidia Dynamo components (includes workers)
	p.log("Step 1/5: Deploying Nvidia Dynamo components")
	if err := p.deployDynamo(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy dynamo: %w", err)
	}

	// Step 2: Deploy Envoy Gateway (must be before Semantic Router to install Gateway API CRDs)
	p.log("Step 2/5: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 3: Deploy Semantic Router with Dynamo integration
	p.log("Step 3/5: Deploying Semantic Router with Dynamo integration")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 4: Configure Gateway API routing
	p.log("Step 4/5: Configuring Gateway API routing")
	if err := p.configureDynamoSettings(ctx, opts); err != nil {
		return fmt.Errorf("failed to configure dynamo settings: %w", err)
	}

	// Step 5: Verify all components are ready
	p.log("Step 5/5: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Dynamo test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	p.log("Cleaning up worker resources")
	p.cleanupWorkerResources(ctx, opts)

	p.log("Uninstalling Envoy Gateway")
	deployer.Uninstall(ctx, "eg", "envoy-gateway-system")

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, "semantic-router", "vllm-semantic-router-system")

	p.log("Uninstalling Dynamo components")
	p.cleanupDynamo(ctx, deployer, opts)

	p.log("Dynamo test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// Dynamo-specific test cases
		"dynamo-health-check",
		"dynamo-category-classification",
		"dynamo-optimized-inference",
		"dynamo-performance-comparison",
		"dynamo-dynamic-batching",
		"dynamo-gpu-utilization",
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

func (p *Profile) deployDynamo(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Deploy Dynamo using official Helm charts
	// This installs CRDs, platform (etcd, NATS, operator), then we deploy custom frontend
	// Reference: https://github.com/ai-dynamo/dynamo/blob/main/docs/kubernetes/README.md

	namespace := "dynamo-system"
	releaseVersion := "0.6.1" // Using v0.6.1 with minimal spec (no dynamoComponent needed)

	// Step 1: Install Dynamo CRDs (cluster-scoped, install to default namespace)
	p.log("  Step 1/5: Installing Dynamo CRDs (version: %s)", releaseVersion)
	crdsChartURL := fmt.Sprintf("https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-crds-%s.tgz", releaseVersion)

	crdsInstallOpts := helm.InstallOptions{
		ReleaseName: "dynamo-crds",
		Chart:       crdsChartURL,
		Namespace:   "default", // CRDs are cluster-scoped
		Wait:        true,
		Timeout:     "5m",
	}
	if err := deployer.Install(ctx, crdsInstallOpts); err != nil {
		return fmt.Errorf("failed to install Dynamo CRDs: %w", err)
	}
	p.log("Dynamo CRDs installed successfully")

	// Wait for CRDs to be registered in the API server
	p.log("Waiting for DynamoGraphDeployment CRD to be registered...")
	if err := p.waitForCRD(ctx, opts.KubeConfig, "dynamographdeployments.nvidia.com", 2*time.Minute); err != nil {
		return fmt.Errorf("DynamoGraphDeployment CRD not registered: %w", err)
	}
	p.log("✅ DynamoGraphDeployment CRD is ready")

	// Step 2: Install Dynamo Platform (includes etcd, NATS, operator)
	p.log("  Step 2/5: Installing Dynamo Platform (includes etcd, NATS, operator)")
	platformChartURL := fmt.Sprintf("https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-%s.tgz", releaseVersion)

	platformInstallOpts := helm.InstallOptions{
		ReleaseName: "dynamo-platform",
		Chart:       platformChartURL,
		Namespace:   namespace,
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, platformInstallOpts); err != nil {
		return fmt.Errorf("failed to install Dynamo Platform: %w", err)
	}
	p.log("Dynamo Platform installed successfully")

	// Step 3: List what was actually deployed (for debugging)
	if p.verbose {
		p.log("Listing resources in %s namespace...", namespace)
		p.runKubectl(ctx, opts.KubeConfig, "get", "all", "-n", namespace)
		p.log("Listing StatefulSets in %s namespace...", namespace)
		p.runKubectl(ctx, opts.KubeConfig, "get", "statefulsets", "-n", namespace)
	}

	// Step 3: Wait for platform components to be ready
	p.log("  Step 3/5: Waiting for Dynamo platform components to be ready...")

	// The platform chart may deploy etcd/NATS with different names or as StatefulSets
	// Check for common resource names and types
	etcdReady := false
	natsReady := false

	// Try to find etcd (could be Deployment, StatefulSet, or different name)
	etcdNames := []string{"dynamo-platform-etcd", "etcd", "dynamo-etcd", "etcd-server"}
	for _, name := range etcdNames {
		if err := deployer.WaitForDeployment(ctx, namespace, name, 2*time.Minute); err == nil {
			p.log("etcd is ready (found as deployment: %s)", name)
			etcdReady = true
			break
		}
		// Try StatefulSet
		if err := p.waitForStatefulSet(ctx, opts.KubeConfig, namespace, name, 2*time.Minute); err == nil {
			p.log("etcd is ready (found as statefulset: %s)", name)
			etcdReady = true
			break
		}
	}
	if !etcdReady {
		p.log("Warning: etcd not found with common names, platform chart may use different naming")
		p.log("  Continuing anyway - etcd may be deployed differently or not required")
	}

	// Try to find NATS (could be Deployment, StatefulSet, or different name)
	natsNames := []string{"dynamo-platform-nats", "nats", "dynamo-nats", "nats-server"}
	for _, name := range natsNames {
		if err := deployer.WaitForDeployment(ctx, namespace, name, 2*time.Minute); err == nil {
			p.log("NATS is ready (found as deployment: %s)", name)
			natsReady = true
			break
		}
		// Try StatefulSet
		if err := p.waitForStatefulSet(ctx, opts.KubeConfig, namespace, name, 2*time.Minute); err == nil {
			p.log("NATS is ready (found as statefulset: %s)", name)
			natsReady = true
			break
		}
	}
	if !natsReady {
		p.log("Warning: NATS not found with common names, platform chart may use different naming")
		p.log("  Continuing anyway - NATS may be deployed differently or not required")
	}

	// Wait for Dynamo operator (if deployed)
	operatorNames := []string{"dynamo-platform-dynamo-operator-controller-manager", "dynamo-operator"}
	operatorFound := false
	for _, name := range operatorNames {
		if err := deployer.WaitForDeployment(ctx, namespace, name, 2*time.Minute); err == nil {
			p.log("Dynamo operator is ready (found as deployment: %s)", name)
			operatorFound = true
			break
		}
	}
	if !operatorFound {
		p.log("Warning: dynamo-operator not found or not ready (may not be included in platform chart)")
	}

	// Additional wait for NATS JetStream to be fully initialized (if NATS was found)
	if natsReady {
		p.log("Waiting for NATS JetStream to be fully initialized...")
		time.Sleep(10 * time.Second)
	}

	// Step 4: Deploy DynamoGraphDeployment via Helm chart
	// This tests proper Dynamo integration (Frontend coordinates with workers via etcd/NATS)
	// The Helm chart allows dynamic model configuration via values
	p.log("  Step 4/5: Deploying DynamoGraphDeployment via Helm chart")

	// Use the local Helm chart for Dynamo vLLM deployment
	dynamoVllmChartPath := "deploy/kubernetes/dynamo/helm-chart"

	// Build Helm install options with model configuration
	// Default uses TinyLlama for E2E testing (lightweight model)
	dynamoVllmOpts := helm.InstallOptions{
		ReleaseName: "dynamo-vllm",
		Chart:       dynamoVllmChartPath,
		Namespace:   namespace,
		Set: map[string]string{
			// Use default TinyLlama model from values.yaml
			// Can be overridden via opts.DynamoModel if needed
			"global.namespace": namespace,
			"global.logLevel":  "info",
		},
		Wait:    true,
		Timeout: "15m", // Model loading can take time
	}

	// Allow custom model configuration via environment or options
	if modelPath := os.Getenv("DYNAMO_MODEL_PATH"); modelPath != "" {
		p.log("Using custom model: %s", modelPath)
		dynamoVllmOpts.Set["workers[0].model.path"] = modelPath
		dynamoVllmOpts.Set["workers[0].workerType"] = "prefill"
		dynamoVllmOpts.Set["workers[1].model.path"] = modelPath
		dynamoVllmOpts.Set["workers[1].workerType"] = "decode"
	}

	p.log("Deploying DynamoGraphDeployment (Frontend + Prefill Worker + Decode Worker with GPU)...")
	if err := deployer.Install(ctx, dynamoVllmOpts); err != nil {
		return fmt.Errorf("failed to deploy DynamoGraphDeployment via Helm: %w", err)
	}

	// Step 5: Wait for Dynamo operator to create resources
	p.log("  Step 5/5: Waiting for Dynamo operator to create Frontend and Workers...")

	// Wait for operator to create deployments (may take time)
	time.Sleep(15 * time.Second)

	// Wait for Frontend to be ready
	// Helm chart creates deployment with name pattern: <release>-frontend or vllm-frontend
	p.log("Waiting for Frontend deployment...")
	frontendNames := []string{"dynamo-vllm-frontend", "vllm-frontend", "dynamo-vllm"}
	frontendFound := false
	for _, name := range frontendNames {
		if err := deployer.WaitForDeployment(ctx, namespace, name, 5*time.Minute); err == nil {
			p.log("✅ Frontend is ready (found as deployment: %s)", name)
			frontendFound = true
			break
		}
	}
	if !frontendFound {
		return fmt.Errorf("frontend deployment not ready: no frontend deployment found")
	}

	// Wait for Prefill Worker (disaggregated deployment)
	// Helm chart creates workers with names from values: prefill-worker-0, decode-worker-1
	p.log("Waiting for Prefill Worker deployment...")
	prefillNames := []string{
		"dynamo-vllm-prefillworker0", // Helm chart generated name (index 0)
		"dynamo-vllm-prefillworker1", // Legacy name (if values.yaml not updated)
		"vllm-vllmprefillworker",     // Operator generated name
		"vllm-prefillworker",
		"prefill-worker-0",
	}
	prefillFound := false
	for _, name := range prefillNames {
		if err := deployer.WaitForDeployment(ctx, namespace, name, 10*time.Minute); err == nil {
			p.log("✅ Prefill Worker is ready (found as deployment: %s)", name)
			prefillFound = true
			break
		}
	}
	if !prefillFound {
		p.log("⚠️  Prefill Worker not found (may be using aggregated deployment)")
	}

	// Wait for Decode Worker
	p.log("Waiting for Decode Worker deployment...")
	decodeNames := []string{
		"dynamo-vllm-decodeworker1", // Helm chart generated name (CamelCase)
		"vllm-vllmdecodeworker",     // Operator generated name
		"vllm-decodeworker",
		"decode-worker-1",
	}
	decodeFound := false
	for _, name := range decodeNames {
		if err := deployer.WaitForDeployment(ctx, namespace, name, 10*time.Minute); err == nil {
			p.log("✅ Decode Worker is ready (found as deployment: %s)", name)
			decodeFound = true
			break
		}
	}
	if !decodeFound {
		return fmt.Errorf("decode worker deployment not ready: no decode worker deployment found")
	}

	p.log("✅ DynamoGraphDeployment created successfully via Helm!")
	p.log("   Disaggregated Deployment: Frontend + Prefill Worker + Decode Worker")
	p.log("   Model configuration can be customized via Helm values")
	p.log("   All components register with ETCD/NATS for KV-aware routing")

	return nil
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	// Use local Helm chart instead of remote OCI registry (path relative to project root)
	chartPath := "deploy/helm/semantic-router"
	valuesFile := "deploy/kubernetes/dynamo/semantic-router-values/values.yaml"

	// Check if Dynamo-specific values file exists, otherwise use default
	if _, err := os.Stat(valuesFile); err != nil {
		p.log("Dynamo-specific values file not found, using default values")
		valuesFile = "deploy/kubernetes/ai-gateway/semantic-router-values/values.yaml"
	}

	// Override image to use locally built image
	imageRepo := "ghcr.io/vllm-project/semantic-router/extproc"
	imageTag := opts.ImageTag

	installOpts := helm.InstallOptions{
		ReleaseName: "semantic-router",
		Chart:       chartPath,
		Namespace:   "vllm-semantic-router-system",
		ValuesFiles: []string{valuesFile},
		Set: map[string]string{
			"image.repository": imageRepo,
			"image.tag":        imageTag,
			"image.pullPolicy": "Never", // Use local image, don't pull from registry
		},
		Wait:    true,
		Timeout: "30m", // Increased timeout for Semantic Router (model downloads can take time)
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		// If Helm install fails but pod is running, continue anyway
		// This can happen if Helm times out but the deployment is actually ready
		p.log("Warning: Helm install reported error, but checking if deployment is ready anyway: %v", err)
		// Don't return error immediately - check if deployment is actually ready
	}

	// Wait for deployment separately with longer timeout
	// This ensures we wait even if Helm's wait timed out
	return deployer.WaitForDeployment(ctx, "vllm-semantic-router-system", "semantic-router", 30*time.Minute)
}

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	// Use Dynamo-specific values file that enables EnvoyPatchPolicy
	// This is required for Semantic Router ExtProc integration
	valuesFile := "deploy/kubernetes/dynamo/dynamo-resources/envoy-gateway-values.yaml"

	installOpts := helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Version:     "v0.0.0-latest",
		ValuesFiles: []string{valuesFile}, // Enable extensionAPIs.enableEnvoyPatchPolicy
		Wait:        true,
		Timeout:     "5m",
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, "envoy-gateway-system", "envoy-gateway", 5*time.Minute)
}

func (p *Profile) configureDynamoSettings(ctx context.Context, opts *framework.SetupOptions) error {
	dynamoManifestsPath := "deploy/kubernetes/dynamo/dynamo-resources"

	// Deploy RBAC for Semantic Router to access Dynamo CRDs
	// Must be done AFTER Semantic Router deployment (which creates vllm-semantic-router-system namespace)
	rbacPath := fmt.Sprintf("%s/rbac.yaml", dynamoManifestsPath)
	p.log("Deploying RBAC for Semantic Router to access Dynamo CRDs...")
	if err := p.kubectlApply(ctx, opts.KubeConfig, rbacPath); err != nil {
		return fmt.Errorf("failed to deploy RBAC: %w", err)
	}
	p.log("RBAC configured successfully")

	// Configure Dynamo optimization settings via ConfigMap
	// This includes:
	// - KV cache management settings
	// - Worker pool configuration
	// - Routing optimization parameters (KV-aware routing, load balancing)
	// - NATS configuration for message queuing

	// Apply Dynamo configuration ConfigMap
	configPath := fmt.Sprintf("%s/dynamo-config.yaml", dynamoManifestsPath)
	if _, err := os.Stat(configPath); err == nil {
		p.log("Applying Dynamo optimization settings (KV cache, routing, worker pool config)")
		if err := p.kubectlApply(ctx, opts.KubeConfig, configPath); err != nil {
			return fmt.Errorf("failed to apply dynamo config: %w", err)
		}
		p.log("Dynamo optimization settings configured")
	} else {
		p.log("Warning: Dynamo config not found at %s, using platform defaults", configPath)
	}

	// Configure Envoy to route to Dynamo backend
	// This must be done AFTER Envoy Gateway is deployed (which installs Gateway API CRDs)
	gatewayResourcesPath := fmt.Sprintf("%s/gwapi-resources.yaml", dynamoManifestsPath)
	if _, err := os.Stat(gatewayResourcesPath); err == nil {
		p.log("Applying Gateway API resources for Dynamo routing (CRDs should be installed by Envoy Gateway)")
		// Wait a bit to ensure CRDs are fully registered
		time.Sleep(5 * time.Second)
		if err := p.kubectlApply(ctx, opts.KubeConfig, gatewayResourcesPath); err != nil {
			return fmt.Errorf("failed to apply gateway resources: %w", err)
		}
		p.log("Gateway API resources configured for Dynamo routing")
	} else {
		p.log("Warning: Gateway API resources not found at %s", gatewayResourcesPath)
	}

	return nil
}

func (p *Profile) deployWorkerResources(ctx context.Context, opts *framework.SetupOptions) error {
	// Workers are now deployed via the Dynamo vLLM Helm chart in deployDynamo()
	// This function is kept for backward compatibility or additional worker pools
	//
	// To deploy additional workers, you can:
	// 1. Upgrade the Helm release with additional workers in values
	// 2. Or apply additional DynamoGraphDeployment resources
	//
	// Example: helm upgrade dynamo-vllm ./deploy/kubernetes/dynamo/helm-chart \
	//          -f custom-workers.yaml -n dynamo-system

	p.log("Workers are managed via Dynamo vLLM Helm chart")
	p.log("To add more workers, upgrade the Helm release with custom values")

	return nil
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	// Create Kubernetes client
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kube client: %w", err)
	}

	// Wait for Envoy Gateway service to be ready with retry
	retryTimeout := 10 * time.Minute
	retryInterval := 5 * time.Second
	startTime := time.Now()

	p.log("Waiting for Envoy Gateway service to be ready...")

	// Label selector for the semantic-router gateway service
	labelSelector := "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"

	var envoyService string
	for {
		// Try to get Envoy service name
		envoyService, err = helpers.GetEnvoyServiceName(ctx, client, labelSelector, p.verbose)
		if err == nil {
			// Verify that the service has exactly 1 pod running with all containers ready
			podErr := helpers.VerifyServicePodsRunning(ctx, client, "envoy-gateway-system", envoyService, p.verbose)
			if podErr == nil {
				p.log("Envoy Gateway service is ready: %s", envoyService)
				break
			}
			if p.verbose {
				p.log("Envoy service found but pods not ready: %v", podErr)
			}
			err = fmt.Errorf("service pods not ready: %w", podErr)
		}

		if time.Since(startTime) >= retryTimeout {
			return fmt.Errorf("failed to get Envoy service with running pods after %v: %w", retryTimeout, err)
		}

		if p.verbose {
			p.log("Envoy service not ready, retrying in %v... (elapsed: %v)",
				retryInterval, time.Since(startTime).Round(time.Second))
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(retryInterval):
			// Continue retry
		}
	}

	// Check all deployments are healthy
	p.log("Verifying all deployments are healthy...")

	// Check semantic-router deployment
	if err := helpers.CheckDeployment(ctx, client, "vllm-semantic-router-system", "semantic-router", p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	// Check envoy-gateway deployment
	if err := helpers.CheckDeployment(ctx, client, "envoy-gateway-system", "envoy-gateway", p.verbose); err != nil {
		return fmt.Errorf("envoy-gateway deployment not healthy: %w", err)
	}

	// Check Dynamo operator (managed by platform chart)
	if err := helpers.CheckDeployment(ctx, client, "dynamo-system", "dynamo-platform-dynamo-operator-controller-manager", p.verbose); err != nil {
		if p.verbose {
			p.log("Dynamo operator deployment not found or not healthy: %v", err)
		}
	}

	p.log("All deployments are healthy")

	return nil
}

func (p *Profile) cleanupDynamo(ctx context.Context, deployer *helm.Deployer, opts *framework.TeardownOptions) error {
	// Clean up Dynamo resources
	namespace := "dynamo-system"

	// Step 1: Uninstall Dynamo vLLM Helm chart (DynamoGraphDeployment)
	p.log("Uninstalling Dynamo vLLM Helm release (DynamoGraphDeployment)")
	if err := deployer.Uninstall(ctx, "dynamo-vllm", namespace); err != nil {
		p.log("Warning: Failed to uninstall dynamo-vllm: %v", err)
	}

	// Step 2: Clean up any remaining static resources (RBAC, etc.)
	dynamoManifestsPath := "deploy/kubernetes/dynamo/dynamo-resources"
	cleanupFiles := []string{
		"rbac.yaml", // RBAC for Dynamo CRDs (may still be applied separately)
	}

	for _, file := range cleanupFiles {
		filePath := fmt.Sprintf("%s/%s", dynamoManifestsPath, file)
		if _, err := os.Stat(filePath); err == nil {
			p.log("Cleaning up %s", file)
			p.kubectlDelete(ctx, opts.KubeConfig, filePath)
		}
	}

	// Step 3: Uninstall Dynamo Platform Helm release (includes etcd, NATS, operator)
	p.log("Uninstalling Dynamo Platform Helm release")
	if err := deployer.Uninstall(ctx, "dynamo-platform", namespace); err != nil {
		p.log("Warning: Failed to uninstall dynamo-platform: %v", err)
	}

	// Step 4: Uninstall Dynamo CRDs Helm release (cluster-scoped)
	p.log("Uninstalling Dynamo CRDs Helm release")
	if err := deployer.Uninstall(ctx, "dynamo-crds", "default"); err != nil {
		p.log("Warning: Failed to uninstall dynamo-crds: %v", err)
	}

	return nil
}

func (p *Profile) cleanupWorkerResources(ctx context.Context, opts *framework.TeardownOptions) error {
	// Clean up RBAC and Gateway API resources
	// Note: DynamoGraphDeployment is now cleaned up via Helm uninstall in cleanupDynamo
	dynamoManifestsPath := "deploy/kubernetes/dynamo/dynamo-resources"

	cleanupFiles := []string{
		"rbac.yaml", // RBAC for Dynamo CRDs
	}

	for _, file := range cleanupFiles {
		filePath := fmt.Sprintf("%s/%s", dynamoManifestsPath, file)
		if _, err := os.Stat(filePath); err == nil {
			p.log("Cleaning up %s", file)
			p.kubectlDelete(ctx, opts.KubeConfig, filePath)
		} else {
			p.log("%s not found, skipping cleanup", file)
		}
	}

	// Clean up Gateway API resources
	gatewayResourcesPath := "deploy/kubernetes/dynamo/dynamo-resources/gwapi-resources.yaml"
	if _, err := os.Stat(gatewayResourcesPath); err == nil {
		p.log("Cleaning up Gateway API resources")
		p.kubectlDelete(ctx, opts.KubeConfig, gatewayResourcesPath)
	}

	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

// waitForStatefulSet waits for a StatefulSet to be ready
func (p *Profile) waitForStatefulSet(ctx context.Context, kubeConfig, namespace, name string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	timeoutSeconds := int(timeout.Seconds())
	// Use rollout status for StatefulSets instead of wait --for=condition=Ready
	// (StatefulSets don't have a Ready condition like Deployments)
	cmd := exec.CommandContext(ctx, "kubectl", "rollout", "status",
		fmt.Sprintf("statefulset/%s", name),
		"-n", namespace,
		fmt.Sprintf("--timeout=%ds", timeoutSeconds),
		"--kubeconfig", kubeConfig)

	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}

	if err := cmd.Run(); err != nil {
		return fmt.Errorf("statefulset failed to become ready: %w", err)
	}

	return nil
}

// waitForCRD waits for a CRD to be registered in the API server
func (p *Profile) waitForCRD(ctx context.Context, kubeConfig, crdName string, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return fmt.Errorf("timeout waiting for CRD %s to be registered", crdName)
		case <-ticker.C:
			// Try to get the CRD
			cmd := exec.CommandContext(ctx, "kubectl", "get", "crd", crdName, "--kubeconfig", kubeConfig)
			if err := cmd.Run(); err == nil {
				// CRD exists and is registered
				return nil
			}
			// CRD not ready yet, continue waiting
			if p.verbose {
				p.log("Waiting for CRD %s to be registered...", crdName)
			}
		}
	}
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "-f", manifest)
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
		fmt.Printf("[Dynamo] "+format+"\n", args...)
	}
}
