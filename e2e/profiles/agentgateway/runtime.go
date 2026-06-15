package agentgateway

import (
	"context"
	"fmt"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
)

const (
	gatewayAPICRDsURL           = "https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.5.0/standard-install.yaml"
	timeoutAgentGatewayInstall  = 10 * time.Minute
	timeoutSemanticRouterDeploy = 20 * time.Minute
	timeoutDemoLLMDeploy        = 10 * time.Minute
)

func (p *Profile) installGatewayAPICRDs(ctx context.Context, opts *framework.SetupOptions) error {
	return p.runKubectl(ctx, opts.KubeConfig,
		"apply", "--server-side", "--force-conflicts", "-f", gatewayAPICRDsURL,
	)
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
	release.Set = map[string]string{
		"image.repository": "ghcr.io/vllm-project/semantic-router/extproc",
		"image.tag":        opts.ImageTag,
		"image.pullPolicy": "Never",
	}
	if err := deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("install semantic-router: %w", err)
	}
	return deployer.WaitForDeployment(ctx, agentGatewayNamespace, semanticRouterDeployment, timeoutSemanticRouterDeploy)
}

func (p *Profile) failSetup(ctx context.Context, opts *framework.SetupOptions, state *setupState, err error) error {
	p.log("ERROR: %v", err)
	p.cleanupPartialDeployment(ctx, opts, state)
	return err
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
	return cmd.Run()
}
