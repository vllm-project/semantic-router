package dashboard

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"
	"k8s.io/client-go/tools/clientcmd"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	profileName = "dashboard"

	// Deploy the Dashboard in the Router namespace so setup can bind it to the
	// exact immutable config snapshot referenced by the Router Deployment.
	namespaceRouter = "vllm-semantic-router-system"

	valuesFile = "e2e/profiles/dashboard/values.yaml"

	dashboardManifestDir = "deploy/kubernetes/observability/dashboard"

	// dashboardE2EDeploymentManifest is the E2E-specific deployment for the dashboard.
	// It intentionally omits the ml-training-service sidecar (unavailable in CI)
	// and sets ML_PIPELINE_ENABLED=false. All other spec comes from the same image
	// and configmap as the production deployment.
	dashboardE2EDeploymentManifest  = "e2e/profiles/dashboard/dashboard-deployment.yaml"
	dashboardE2EPVCManifest         = "e2e/profiles/dashboard/dashboard-pvc.yaml"
	dashboardIssuerServiceManifest  = "e2e/profiles/dashboard/dashboard-issuer-service.yaml"
	managedStoresManifest           = "e2e/profiles/dashboard/managed-stores.yaml"
	dashboardE2EEgressManifest      = "e2e/profiles/dashboard/router-egress-policy.yaml"
	managedInferenceGatewayManifest = "e2e/profiles/dashboard/managed-inference-gateway.yaml"

	dashboardE2EManagedLabel = "vllm.ai/e2e-managed=true"

	deploymentDashboard = "semantic-router-dashboard"
	serviceDashboard    = "semantic-router-dashboard"

	// ServicePort is the container port used by the E2E framework for direct
	// pod port-forwarding. The service maps 80 → 8700 (targetPort: http), but
	// the framework connects directly to the pod, so use the container port.
	dashboardPort               = "8700"
	dashboardBootstrapTokenPath = "/tmp/vllm-sr-dashboard/bootstrap/router-token"

	timeoutDashboardWait = 10 * time.Minute

	semanticRouterReplicaCount = int32(2)

	setupStepManagedPrerequisites = "managed prerequisites"
	setupStepGatewayFoundations   = "gateway foundations"
	setupStepPublicInference      = "public inference front door"
	setupStepDashboardClient      = "Dashboard control-plane client"
	setupStepFirstPublication     = "first managed publication"
	setupStepRouterReadiness      = "Router inference readiness"
)

var resourceManifests = []string{
	"deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
	managedInferenceGatewayManifest,
}

// Profile implements the dashboard E2E test profile.
// It deploys the base gateway stack (router + Envoy) and then the dashboard
// on top, so testcases can exercise the full /api/* surface.
type Profile struct {
	verbose bool
	stack   *gatewaystack.Stack
}

type setupStep struct {
	name string
	run  func(context.Context, *framework.SetupOptions) error
}

// NewProfile creates a new dashboard profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                         profileName,
			SemanticRouterValuesFile:     valuesFile,
			PrerequisiteManifests:        []string{managedStoresManifest, dashboardE2EEgressManifest},
			ResourceManifests:            resourceManifests,
			DeferSemanticRouterReadiness: true,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return profileName }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the Dashboard as a Router Management client, including identity bootstrap, Agent continuity, and restart recovery"
}

// Setup provisions the managed dependencies, starts the Router without waiting
// on inference readiness, and lets the Dashboard publish the first revision
// before closing the Router /ready gate.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Dashboard test environment")

	for _, step := range p.setupSteps() {
		p.log("Starting %s", step.name)
		if err := step.run(ctx, opts); err != nil {
			return fmt.Errorf("%s: %w", step.name, err)
		}
	}

	p.log("Dashboard test environment setup complete")
	return nil
}

func (p *Profile) setupSteps() []setupStep {
	return []setupStep{
		{name: setupStepManagedPrerequisites, run: p.prepareManagedPrerequisites},
		{name: setupStepGatewayFoundations, run: p.stack.Setup},
		{name: setupStepPublicInference, run: p.preparePublicInferenceFrontDoor},
		{name: setupStepDashboardClient, run: p.deployDashboard},
		{name: setupStepFirstPublication, run: p.bootstrapFirstPublication},
		{name: setupStepRouterReadiness, run: p.waitForRouterReadiness},
	}
}

// Teardown removes the dashboard then tears down the shared gateway stack.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down Dashboard test environment")

	if err := p.cleanupDashboard(ctx, opts); err != nil {
		p.log("Warning: failed to cleanup dashboard resources: %v", err)
	}

	if err := p.stack.Teardown(ctx, opts); err != nil {
		return err
	}

	p.log("Dashboard test environment teardown complete")
	return nil
}

// GetTestCases returns the dashboard E2E test contract.
func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(testmatrix.DashboardContract)
}

// GetServiceConfig returns the dashboard service used by the health, status,
// evaluation, and restart-recovery contracts.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		Name:        serviceDashboard,
		Namespace:   namespaceRouter,
		ServicePort: dashboardPort,
	}
}

// ---------------------------------------------------------------------------
// dashboard lifecycle
// ---------------------------------------------------------------------------

func (p *Profile) deployDashboard(ctx context.Context, opts *framework.SetupOptions) error {
	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceRouter, dashboardManifestDir+"/configmap.yaml"); err != nil {
		return fmt.Errorf("failed to apply dashboard configmap: %w", err)
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceRouter, dashboardE2EPVCManifest); err != nil {
		return fmt.Errorf("failed to apply dashboard PVC: %w", err)
	}

	if err := p.applyDashboardDeployment(ctx, opts); err != nil {
		return fmt.Errorf("failed to apply dashboard deployment: %w", err)
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceRouter, dashboardIssuerServiceManifest); err != nil {
		return fmt.Errorf("failed to apply dashboard issuer service: %w", err)
	}

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceRouter, dashboardManifestDir+"/service.yaml"); err != nil {
		return fmt.Errorf("failed to apply dashboard service: %w", err)
	}

	p.log("Waiting for dashboard deployment to be ready")
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	if err := deployer.WaitForDeployment(ctx, namespaceRouter, deploymentDashboard, timeoutDashboardWait); err != nil {
		// Use a fresh context: the parent ctx may be cancelled/killed (causing
		// "signal: killed") so diagnostic commands must run independently.
		diagCtx, diagCancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer diagCancel()

		_ = p.runKubectlAlways(diagCtx, opts.KubeConfig, "describe", "pods",
			"-l", "app="+deploymentDashboard,
			"-n", namespaceRouter,
		)
		// --previous=true shows the crash logs of the last terminated container,
		// which is the most useful output (contains the actual fatal error).
		_ = p.runKubectlAlways(diagCtx, opts.KubeConfig, "logs",
			"-l", "app="+deploymentDashboard,
			"-n", namespaceRouter,
			"--all-containers=true",
			"--previous=true",
		)
		// Also grab current container logs in case it hasn't restarted yet.
		_ = p.runKubectlAlways(diagCtx, opts.KubeConfig, "logs",
			"-l", "app="+deploymentDashboard,
			"-n", namespaceRouter,
			"--all-containers=true",
		)
		return fmt.Errorf("dashboard deployment not ready: %w", err)
	}

	// Give the dashboard a moment to finish initialising its HTTP handlers.
	time.Sleep(3 * time.Second)
	return nil
}

func (p *Profile) bootstrapFirstPublication(ctx context.Context, opts *framework.SetupOptions) error {
	restConfig, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("load Kubernetes REST config: %w", err)
	}
	session, err := fixtures.OpenServiceSessionForConfig(
		ctx,
		opts.KubeClient,
		restConfig,
		p.GetServiceConfig(),
		opts.Verbose,
	)
	if err != nil {
		return fmt.Errorf("connect to Dashboard bootstrap API: %w", err)
	}
	defer session.Close()
	sessionToken, err := fixtures.EnsureDashboardAdmin(
		ctx,
		session.HTTPClient(45*time.Second),
		session.BaseURL(),
		opts.Verbose,
	)
	if err != nil {
		return fmt.Errorf("complete Router-backed Dashboard installation: %w", err)
	}
	if err := fixtures.WaitForFirstRouterPublication(
		ctx,
		session.HTTPClient(45*time.Second),
		session.BaseURL(),
		sessionToken,
		90*time.Second,
		opts.Verbose,
	); err != nil {
		return err
	}
	// A successful first-admin transaction must consume the one-time Router
	// bootstrap credential. Router /ready remains fail-closed while it exists.
	if err := p.runKubectl(
		ctx,
		opts.KubeConfig,
		"exec",
		"deployment/"+deploymentDashboard,
		"-n",
		namespaceRouter,
		"--",
		"test",
		"!",
		"-e",
		dashboardBootstrapTokenPath,
	); err != nil {
		return fmt.Errorf("one-time Router bootstrap credential was not consumed: %w", err)
	}
	return nil
}

func (p *Profile) waitForRouterReadiness(ctx context.Context, opts *framework.SetupOptions) error {
	return p.stack.WaitForSemanticRouterReady(ctx, opts, semanticRouterReplicaCount)
}

func (p *Profile) cleanupDashboard(ctx context.Context, opts *framework.TeardownOptions) error {
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardManifestDir+"/service.yaml", "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardIssuerServiceManifest, "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardE2EDeploymentManifest, "-n", namespaceRouter, "--ignore-not-found=true")
	// Delete only PVCs created by this E2E profile (label vllm.ai/e2e-managed=true).
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "pvc", "-l", dashboardE2EManagedLabel, "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardManifestDir+"/configmap.yaml", "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "secret", dashboardRouterSecretName, dashboardIdentitySecretName, "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "service", dashboardPublicInferenceServiceName, "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "secret", dashboardStoreSecretName, "-n", "default", "--ignore-not-found=true")
	return nil
}

// ---------------------------------------------------------------------------
// kubectl helpers (same pattern as production-stack)
// ---------------------------------------------------------------------------

func (p *Profile) kubectl(ctx context.Context, kubeConfig string, args ...string) error {
	return p.runKubectl(ctx, kubeConfig, args...)
}

func (p *Profile) kubectlApplyWithNamespace(ctx context.Context, kubeConfig, namespace, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest, "-n", namespace)
}

func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	cmd := exec.CommandContext(ctx, "kubectl", dashboardKubectlArgs(kubeConfig, args...)...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

// runKubectlAlways runs kubectl and always streams output, regardless of verbose.
// Use for diagnostic commands (describe, logs) called in error paths.
func (p *Profile) runKubectlAlways(ctx context.Context, kubeConfig string, args ...string) error {
	cmd := exec.CommandContext(ctx, "kubectl", dashboardKubectlArgs(kubeConfig, args...)...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func dashboardKubectlArgs(kubeConfig string, args ...string) []string {
	result := make([]string, 0, len(args)+2)
	result = append(result, "--kubeconfig", kubeConfig)
	return append(result, args...)
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[Dashboard] "+format+"\n", args...)
	}
}
