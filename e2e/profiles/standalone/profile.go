package standalone

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	valuesFile          = "e2e/profiles/standalone/values.yaml"
	unreachableBackend  = "e2e/profiles/standalone/unreachable-backend.yaml"
	providerMockerKusto = "deploy/kubernetes/provider-mocker"
	listenerPort        = "8899"
)

// Profile runs the Router on the Helm chart's defaults: standalone mode, where
// clients reach the Router's own listener through its Service and no gateway
// runs in the cluster.
type Profile struct {
	verbose bool
}

// NewProfile creates the standalone profile.
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name.
func (p *Profile) Name() string {
	return "standalone"
}

// Description returns the profile description.
func (p *Profile) Description() string {
	return "Helm defaults: the standalone Router serves chat, models, fallback and config rollouts through its Service"
}

// Setup deploys the model backends and the chart with only its image overridden.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	for _, args := range [][]string{{"-k", providerMockerKusto}, {"-f", unreachableBackend}} {
		if err := p.kubectl(ctx, opts.KubeConfig, append([]string{"apply"}, args...)...); err != nil {
			return fmt.Errorf("apply model backends: %w", err)
		}
	}
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	release := helm.SemanticRouterRelease.Clone()
	release.ValuesFiles = []string{valuesFile}
	release.Set = map[string]string{
		"image.tag":        opts.ImageTag,
		"image.pullPolicy": "Never",
	}
	if err := deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("install the chart on its defaults: %w", err)
	}
	client, err := helpers.NewKubeClient(opts.KubeConfig)
	if err != nil {
		return err
	}
	for _, ref := range []helpers.DeploymentRef{
		{Namespace: "default", Name: "provider-mocker"},
		{Namespace: release.Namespace, Name: release.ReleaseName},
	} {
		if err := helpers.WaitForDeploymentReady(ctx, client, ref.Namespace, ref.Name, 30*time.Minute, 5*time.Second, opts.Verbose); err != nil {
			return err
		}
	}
	return nil
}

// Teardown removes the release and the backends.
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	release := helm.SemanticRouterRelease
	if err := deployer.Uninstall(ctx, release.ReleaseName, release.Namespace); err != nil {
		p.log("uninstall %s: %v", release.ReleaseName, err)
	}
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "--ignore-not-found", "-f", unreachableBackend)
	_ = p.kubectl(ctx, opts.KubeConfig, "delete", "--ignore-not-found", "-k", providerMockerKusto)
	return nil
}

// GetTestCases returns the standalone cases; the config rollout runs last.
func (p *Profile) GetTestCases() []string {
	return []string{
		"standalone-chat-completions",
		"standalone-models",
		"standalone-fallback",
		"routing-error-codes",
		"standalone-decision-model",
		"standalone-config-rollout",
	}
}

// GetServiceConfig points the cases at the Router's Service and listener port.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		Name:        helm.SemanticRouterRelease.ReleaseName,
		Namespace:   helm.SemanticRouterRelease.Namespace,
		ServicePort: listenerPort,
	}
}

func (p *Profile) kubectl(ctx context.Context, kubeconfig string, args ...string) error {
	cmd := exec.CommandContext(ctx, "kubectl", append(args, "--kubeconfig", kubeconfig)...) //nolint:gosec // The profile's own manifests and the run's kubeconfig.
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[standalone] "+format+"\n", args...)
	}
}
