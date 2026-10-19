package standalone

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/cluster"
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

// Profile runs the Router on the Helm chart's defaults, its image included:
// standalone mode, where clients reach the Router's own listener through its
// Service and no gateway runs in the cluster.
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

// Setup deploys the model backends and the chart with nothing but its config
// overridden. In a development cycle the chart's default image is the
// development image, so this run's Router image answers to that name inside
// the cluster.
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	for _, args := range [][]string{{"-k", providerMockerKusto}, {"-f", unreachableBackend}} {
		if err := p.kubectl(ctx, opts.KubeConfig, append([]string{"apply"}, args...)...); err != nil {
			return fmt.Errorf("apply model backends: %w", err)
		}
	}
	if err := cluster.TagLoadedImage(ctx, opts.ClusterName, helm.RouterImageRepository+":"+opts.ImageTag, helm.DevelopmentRouterImage); err != nil {
		return fmt.Errorf("name this run's Router image %s: %w", helm.DevelopmentRouterImage, err)
	}
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	release := chartDefaults()
	if err := deployer.Install(ctx, release); err != nil {
		return fmt.Errorf("install the chart on its defaults: %w", err)
	}
	client, err := helpers.NewKubeClient(opts.KubeConfig)
	if err != nil {
		return err
	}
	if err := requireDevelopmentImage(ctx, client, release); err != nil {
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

// chartDefaults installs the chart with only the profile's config: no image,
// pull policy, args or probe overrides. Setup waits for readiness itself, after
// it has checked the image.
func chartDefaults() helm.InstallOptions {
	release := helm.SemanticRouterRelease.Clone()
	release.ValuesFiles = []string{valuesFile}
	release.Wait = false
	return release
}

// requireDevelopmentImage fails at once when the chart's default Router image
// isn't the development image, instead of waiting out a Router that can't
// start.
func requireDevelopmentImage(ctx context.Context, client *kubernetes.Clientset, release helm.InstallOptions) error {
	deployment, err := client.AppsV1().Deployments(release.Namespace).Get(ctx, release.ReleaseName, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("read the Router deployment: %w", err)
	}
	containers := deployment.Spec.Template.Spec.Containers
	if len(containers) == 0 {
		return fmt.Errorf("the Router deployment has no container")
	}
	if image := containers[0].Image; image != helm.DevelopmentRouterImage {
		return fmt.Errorf("the chart's default Router image is %s, want the development image %s: a development cycle's chart deploys the image main publishes (tools/release/check_version_contract.py)", image, helm.DevelopmentRouterImage)
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
		"standalone-chart-defaults",
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
