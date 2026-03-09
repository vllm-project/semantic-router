package productionstack

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

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
