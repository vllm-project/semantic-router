package dashboard

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	gatewaystack "github.com/vllm-project/semantic-router/e2e/pkg/stacks/gateway"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	profileName = "dashboard"

	// Deploy the dashboard in the same namespace as the router so it can
	// mount the semantic-router-config ConfigMap created by the Helm chart.
	namespaceRouter = "vllm-semantic-router-system"

	valuesFile = "e2e/profiles/dashboard/values.yaml"

	dashboardManifestDir = "deploy/kubernetes/observability/dashboard"

	// dashboardE2EDeploymentManifest is the E2E-specific deployment for the dashboard.
	// It intentionally omits the ml-training-service sidecar (unavailable in CI)
	// and sets ML_PIPELINE_ENABLED=false. All other spec comes from the same image
	// and configmap as the production deployment.
	dashboardE2EDeploymentManifest = "e2e/profiles/dashboard/dashboard-deployment.yaml"

	deploymentDashboard = "semantic-router-dashboard"
	serviceDashboard    = "semantic-router-dashboard"

	// ServicePort is the container port used by the E2E framework for direct
	// pod port-forwarding. The service maps 80 → 8700 (targetPort: http), but
	// the framework connects directly to the pod, so use the container port.
	dashboardPort = "8700"

	timeoutDashboardWait = 10 * time.Minute
)

var resourceManifests = []string{
	"deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml",
	"deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml",
}

// Profile implements the dashboard E2E test profile.
// It deploys the base gateway stack (router + Envoy) and then the dashboard
// on top, so testcases can exercise the full /api/* surface.
type Profile struct {
	verbose bool
	stack   *gatewaystack.Stack
}

// NewProfile creates a new dashboard profile.
func NewProfile() *Profile {
	return &Profile{
		stack: gatewaystack.New(gatewaystack.Config{
			Name:                     profileName,
			SemanticRouterValuesFile: valuesFile,
			ResourceManifests:        resourceManifests,
		}),
	}
}

// Name returns the profile name.
func (p *Profile) Name() string { return profileName }

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Tests the dashboard API surface: health, status, config read, deploy preview, config versions, and input validation"
}

// Setup deploys the shared gateway stack then the standalone dashboard.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up Dashboard test environment")

	if err := p.stack.Setup(ctx, opts); err != nil {
		return err
	}

	p.log("Deploying dashboard")
	if err := p.deployDashboard(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy dashboard: %w", err)
	}

	p.log("Dashboard test environment setup complete")
	return nil
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

// GetServiceConfig returns the dashboard service — testcases port-forward here
// to call /healthz, /api/status, /api/router/config/*, etc.
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

	if err := p.kubectlApplyWithNamespace(ctx, opts.KubeConfig, namespaceRouter, dashboardE2EDeploymentManifest); err != nil {
		return fmt.Errorf("failed to apply dashboard deployment: %w", err)
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

func (p *Profile) cleanupDashboard(ctx context.Context, opts *framework.TeardownOptions) error {
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardManifestDir+"/service.yaml", "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardE2EDeploymentManifest, "-n", namespaceRouter, "--ignore-not-found=true")
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "-f", dashboardManifestDir+"/configmap.yaml", "-n", namespaceRouter, "--ignore-not-found=true")
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
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

// runKubectlAlways runs kubectl and always streams output, regardless of verbose.
// Use for diagnostic commands (describe, logs) called in error paths.
func (p *Profile) runKubectlAlways(ctx context.Context, kubeConfig string, args ...string) error {
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	return cmd.Run()
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[Dashboard] "+format+"\n", args...)
	}
}
