package agentgateway

import (
	"context"
	"fmt"
	"os/exec"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
)

const (
	gatewayAPICRDsURL           = "https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml"
	timeoutAgentGatewayInstall  = 10 * time.Minute
	timeoutSemanticRouterDeploy = 20 * time.Minute
	timeoutDemoLLMDeploy        = 10 * time.Minute
	gatewayAPIInstallAttempts   = 3
	gatewayAPIRetryBaseDelay    = 5 * time.Second
)

func (p *Profile) installGatewayAPICRDs(ctx context.Context, opts *framework.SetupOptions) error {
	var lastErr error
	for attempt := 1; attempt <= gatewayAPIInstallAttempts; attempt++ {
		lastErr = p.runKubectl(
			ctx,
			opts.KubeConfig,
			"apply",
			"--server-side",
			"--force-conflicts",
			"-f",
			gatewayAPICRDsURL,
		)
		if lastErr == nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(time.Duration(attempt) * gatewayAPIRetryBaseDelay):
		}
	}
	return fmt.Errorf("apply Gateway API CRDs after retries: %w", lastErr)
}

func (p *Profile) installAgentGateway(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	crdsRelease := helm.InstallOptions{
		ReleaseName: "agentgateway-crds",
		Chart:       "oci://cr.agentgateway.dev/charts/agentgateway-crds",
		Namespace:   agentGatewayNamespace,
		Version:     agentGatewayVersion,
		Set: map[string]string{
			"controller.image.pullPolicy": "Always",
		},
		Wait:    true,
		Timeout: "10m",
	}
	if err := deployer.Install(ctx, crdsRelease); err != nil {
		return fmt.Errorf("install agentgateway-crds: %w", err)
	}

	controllerRelease := helm.InstallOptions{
		ReleaseName: "agentgateway",
		Chart:       "oci://cr.agentgateway.dev/charts/agentgateway",
		Namespace:   agentGatewayNamespace,
		Version:     agentGatewayVersion,
		Set: map[string]string{
			"controller.image.pullPolicy":                                      "Always",
			"controller.extraEnv.KGW_ENABLE_GATEWAY_API_EXPERIMENTAL_FEATURES": "true",
		},
		Wait:    true,
		Timeout: "10m",
	}
	return deployer.Install(ctx, controllerRelease)
}

func (p *Profile) deployDemoLLM(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	if err := p.applyManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/demo-llm.yaml"); err != nil {
		return err
	}
	return deployer.WaitForDeployment(ctx, "default", "vllm-llama3-8b-instruct", timeoutDemoLLMDeploy)
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	release := helm.SemanticRouterRelease.Clone()
	release.Namespace = agentGatewayNamespace
	release.ValuesFiles = []string{semanticRouterValuesFile}
	// The profile owns readiness diagnostics below; avoiding Helm's opaque
	// release-wide wait keeps failures bounded and surfaces pod state.
	release.Wait = false
	release.Timeout = "10m"
	release.Set = map[string]string{
		"image.repository": "ghcr.io/vllm-project/semantic-router/extproc",
		"image.tag":        opts.ImageTag,
		"image.pullPolicy": "Never",
		"config.global.router.streamed_body.enabled":     "true",
		"config.global.router.streamed_body.max_bytes":   "10485760",
		"config.global.router.streamed_body.timeout_sec": "30",
	}
	if err := deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("install semantic-router: %w", err)
	}
	return deployer.WaitForDeployment(ctx, agentGatewayNamespace, semanticRouterDeployment, timeoutSemanticRouterDeploy)
}

func (p *Profile) failSetup(ctx context.Context, opts *framework.SetupOptions, state *setupState, err error) error {
	p.log("ERROR: %v", err)
	p.collectSetupFailureDiagnostics(opts)
	p.cleanupPartialDeployment(ctx, opts, state)
	return err
}

func (p *Profile) collectSetupFailureDiagnostics(opts *framework.SetupOptions) {
	if opts.CollectRouterDiagnostics == nil {
		return
	}
	diagnosticsCtx, cancel := context.WithTimeout(context.Background(), 45*time.Second)
	defer cancel()
	if err := opts.CollectRouterDiagnostics(diagnosticsCtx, agentGatewayNamespace); err != nil {
		p.log("Warning: failed to collect Router diagnostics before cleanup: %v", err)
	}
}

func (p *Profile) cleanupPartialDeployment(ctx context.Context, opts *framework.SetupOptions, state *setupState) {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	if state.extProcPolicyApplied {
		_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/extproc-policy.yaml")
	}
	if state.routingResourcesApplied {
		_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/routing-resources.yaml")
	}
	if state.semanticRouterDeployed {
		_ = deployer.Uninstall(ctx, semanticRouterDeployment, agentGatewayNamespace)
	}
	if state.demoLLMDeployed {
		_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/demo-llm.yaml")
	}
	if state.gatewayProxyCreated {
		_ = p.deleteManifest(ctx, opts.KubeConfig, "deploy/kubernetes/agentgateway/gateway.yaml")
	}
	if state.agentGatewayInstalled {
		_ = deployer.Uninstall(ctx, "agentgateway", agentGatewayNamespace)
		_ = deployer.Uninstall(ctx, "agentgateway-crds", agentGatewayNamespace)
	}
}

func (p *Profile) applyManifest(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

func (p *Profile) deleteManifest(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "-f", manifest, "--ignore-not-found=true")
}

func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf(
			"kubectl %s: %w: %s",
			strings.Join(args, " "),
			err,
			strings.TrimSpace(string(output)),
		)
	}
	return nil
}
