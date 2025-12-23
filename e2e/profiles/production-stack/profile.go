package productionstack

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

const (
	// Profile constants
	profileName = "production-stack"

	// Namespace constants
	namespaceSemanticRouter = "vllm-semantic-router-system"
	namespaceEnvoyGateway   = "envoy-gateway-system"
	namespaceAIGateway      = "envoy-ai-gateway-system"
	namespaceDefault        = "default"

	// Release name constants
	releaseSemanticRouter = "semantic-router"
	releaseEnvoyGateway   = "eg"
	releaseAIGatewayCRD   = "aieg-crd"
	releaseAIGateway      = "aieg"

	// Deployment name constants
	deploymentSemanticRouter = "semantic-router"
	deploymentEnvoyGateway   = "envoy-gateway"
	deploymentAIGateway      = "ai-gateway-controller"

	// Chart and URL constants
	chartPathSemanticRouter = "deploy/helm/semantic-router"
	chartEnvoyGateway       = "oci://docker.io/envoyproxy/gateway-helm"
	chartAIGatewayCRD       = "oci://docker.io/envoyproxy/ai-gateway-crds-helm"
	chartAIGateway          = "oci://docker.io/envoyproxy/ai-gateway-helm"
	envoyGatewayValuesURL   = "https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml"

	// File path constants
	valuesFile           = "e2e/profiles/production-stack/values.yaml"
	baseModelManifest    = "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"
	gatewayAPIManifest   = "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"
	prometheusConfigFile = "e2e/profiles/production-stack/prometheus-config.yaml"

	// Timeout constants
	timeoutSemanticRouterInstall = "30m"
	timeoutHelmInstall           = "10m"
	timeoutDeploymentWait        = 30 * time.Minute
	timeoutServiceRetry          = 10 * time.Minute
	intervalServiceRetry         = 5 * time.Second

	// Image constants
	imageRepository = "ghcr.io/vllm-project/semantic-router/extproc"
	imagePullPolicy = "Never"

	// Label selector constants
	labelSelectorGateway = "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router"

	// Port mapping constants
	portMapping = "8080:80"
)

// Profile implements the production-stack test profile
type Profile struct {
	verbose bool
}

// NewProfile creates a new production-stack profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return profileName
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests Semantic Router with Envoy AI Gateway integration (production-stack)"
}

// Setup deploys all required components for production-stack testing
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Production Stack test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Semantic Router
	p.log("Step 1/7: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 2: Deploy Envoy Gateway
	p.log("Step 2/7: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}

	// Step 3: Deploy Envoy AI Gateway
	p.log("Step 3/7: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}

	// Step 4: Deploy Demo LLM and Gateway API Resources
	p.log("Step 4/7: Deploying Demo LLM and Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}

	// Step 5: Scale deployments for HA/LB
	p.log("Step 5/7: Scaling deployments for high availability")
	if err := p.scaleDeployments(ctx, opts); err != nil {
		return fmt.Errorf("failed to scale deployments: %w", err)
	}

	// Step 6: Deploy Prometheus for monitoring
	p.log("Step 6/7: Deploying Prometheus for monitoring")
	if err := p.deployPrometheus(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy prometheus: %w", err)
	}

	// Step 7: Verify all components are ready
	p.log("Step 7/7: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("Production Stack test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Production Stack test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	p.log("Cleaning up Gateway API resources")
	p.cleanupGatewayResources(ctx, opts)

	p.log("Cleaning up Prometheus")
	p.cleanupPrometheus(ctx, opts)

	p.log("Uninstalling Envoy AI Gateway")
	deployer.Uninstall(ctx, releaseAIGatewayCRD, namespaceAIGateway)
	deployer.Uninstall(ctx, releaseAIGateway, namespaceAIGateway)

	p.log("Uninstalling Envoy Gateway")
	deployer.Uninstall(ctx, releaseEnvoyGateway, namespaceEnvoyGateway)

	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, releaseSemanticRouter, namespaceSemanticRouter)

	p.log("Production Stack test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		// Standard functional tests
		"chat-completions-request",
		"chat-completions-stress-request",
		"domain-classify",
		"semantic-cache",
		"pii-detection",
		"jailbreak-detection",
		"chat-completions-progressive-stress",
		// Production stack specific tests (HA/LB/Observability)
		"multi-replica-health",
		"load-balancing-verification",
		"failover-during-traffic",
		"performance-throughput",
		"resource-utilization-monitoring",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: labelSelectorGateway,
		Namespace:     namespaceEnvoyGateway,
		PortMapping:   portMapping,
	}
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: releaseSemanticRouter,
		Chart:       chartPathSemanticRouter,
		Namespace:   namespaceSemanticRouter,
		ValuesFiles: []string{valuesFile},
		Set: map[string]string{
			"image.repository": imageRepository,
			"image.tag":        opts.ImageTag,
			"image.pullPolicy": imagePullPolicy,
			"replicaCount":     "1", // Start with 1 replica, scale to 2 later
		},
		Wait:    true,
		Timeout: timeoutSemanticRouterInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	if err := deployer.WaitForDeployment(ctx, namespaceSemanticRouter, deploymentSemanticRouter, timeoutDeploymentWait); err != nil {
		return err
	}

	return nil
}

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: releaseEnvoyGateway,
		Chart:       chartEnvoyGateway,
		Namespace:   namespaceEnvoyGateway,
		Version:     "v1.6.0",
		ValuesFiles: []string{envoyGatewayValuesURL},
		Wait:        true,
		Timeout:     timeoutHelmInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, namespaceEnvoyGateway, deploymentEnvoyGateway, timeoutDeploymentWait)
}

func (p *Profile) deployEnvoyAIGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	crdOpts := helm.InstallOptions{
		ReleaseName: releaseAIGatewayCRD,
		Chart:       chartAIGatewayCRD,
		Namespace:   namespaceAIGateway,
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     timeoutHelmInstall,
	}

	if err := deployer.Install(ctx, crdOpts); err != nil {
		return err
	}

	installOpts := helm.InstallOptions{
		ReleaseName: releaseAIGateway,
		Chart:       chartAIGateway,
		Namespace:   namespaceAIGateway,
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     timeoutHelmInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, namespaceAIGateway, deploymentAIGateway, timeoutDeploymentWait)
}

func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.kubectlApply(ctx, opts.KubeConfig, baseModelManifest); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}

	if err := p.kubectlApply(ctx, opts.KubeConfig, gatewayAPIManifest); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

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

	startTime := time.Now()
	p.log("Waiting for Envoy Gateway service to be ready...")

	var envoyService string
	for {
		envoyService, err = helpers.GetEnvoyServiceName(ctx, client, labelSelectorGateway, p.verbose)
		if err == nil {
			podErr := helpers.VerifyServicePodsRunning(ctx, client, namespaceEnvoyGateway, envoyService, p.verbose)
			if podErr == nil {
				p.log("Envoy Gateway service is ready: %s", envoyService)
				break
			}
			if p.verbose {
				p.log("Envoy service found but pods not ready: %v", podErr)
			}
			err = fmt.Errorf("service pods not ready: %w", podErr)
		}

		if time.Since(startTime) >= timeoutServiceRetry {
			return fmt.Errorf("failed to get Envoy service with running pods after %v: %w", timeoutServiceRetry, err)
		}

		if p.verbose {
			p.log("Envoy service not ready, retrying in %v... (elapsed: %v)",
				intervalServiceRetry, time.Since(startTime).Round(time.Second))
		}

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(intervalServiceRetry):
		}
	}

	p.log("Verifying all deployments are healthy...")

	if err := helpers.CheckDeployment(ctx, client, namespaceSemanticRouter, deploymentSemanticRouter, p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	if err := helpers.CheckDeployment(ctx, client, namespaceEnvoyGateway, deploymentEnvoyGateway, p.verbose); err != nil {
		return fmt.Errorf("envoy-gateway deployment not healthy: %w", err)
	}

	if err := helpers.CheckDeployment(ctx, client, namespaceAIGateway, deploymentAIGateway, p.verbose); err != nil {
		return fmt.Errorf("ai-gateway-controller deployment not healthy: %w", err)
	}

	p.log("All deployments are healthy")

	return nil
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

	p.log("Scaling vllm-llama3-8b-instruct deployment to 2 replicas")
	if err := p.kubectl(ctx, opts.KubeConfig, "scale", "deployment", "vllm-llama3-8b-instruct", "-n", namespaceDefault, "--replicas=2"); err != nil {
		return fmt.Errorf("failed to scale vllm demo deployment: %w", err)
	}

	if err := deployer.WaitForDeployment(ctx, namespaceDefault, "vllm-llama3-8b-instruct", timeoutDeploymentWait); err != nil {
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
	p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/service.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/deployment.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/pvc.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/configmap.yaml", "-n", namespaceDefault, "--ignore-not-found=true")
	p.kubectl(ctx, opts.KubeConfig, "delete", "-f", prometheusDir+"/rbac.yaml", "--ignore-not-found=true")
	p.kubectl(ctx, opts.KubeConfig, "delete", "serviceaccount", "prometheus", "-n", namespaceDefault, "--ignore-not-found=true")
	return nil
}

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) error {
	p.kubectlDelete(ctx, opts.KubeConfig, gatewayAPIManifest)
	p.kubectlDelete(ctx, opts.KubeConfig, baseModelManifest)
	return nil
}

func (p *Profile) kubectl(ctx context.Context, kubeConfig string, args ...string) error {
	return p.runKubectl(ctx, kubeConfig, args...)
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

func (p *Profile) kubectlApplyWithNamespace(ctx context.Context, kubeConfig, namespace, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest, "-n", namespace)
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
		fmt.Printf("[Production-Stack] "+format+"\n", args...)
	}
}
