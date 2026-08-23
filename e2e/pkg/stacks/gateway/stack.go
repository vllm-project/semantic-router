package gateway

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
)

const (
	deploymentSemanticRouter = "semantic-router"
	deploymentEnvoyGateway   = "envoy-gateway"
	deploymentAIGateway      = "ai-gateway-controller"

	imageRepository = "ghcr.io/vllm-project/semantic-router/extproc"
	imagePullPolicy = "Never"

	timeoutServiceRetry   = 10 * time.Minute
	intervalServiceRetry  = 5 * time.Second
	timeoutDeploymentWait = 30 * time.Minute
)

var defaultVerifyDeployments = []helpers.DeploymentRef{
	{Namespace: helm.SemanticRouterRelease.Namespace, Name: deploymentSemanticRouter},
	{Namespace: helm.EnvoyGatewayRelease.Namespace, Name: deploymentEnvoyGateway},
	{Namespace: helm.AIGatewayRelease.Namespace, Name: deploymentAIGateway},
}

// Config describes a reusable semantic-router + Envoy Gateway + AI Gateway test stack.
type Config struct {
	Name string

	SemanticRouterValuesFile string
	SemanticRouterSet        map[string]string

	ServiceConfig framework.ServiceConfig

	PrerequisiteManifests []string
	ResourceManifests     []string
	WaitDeployments       []helpers.DeploymentRef
	VerifyDeployments     []helpers.DeploymentRef

	// DeferSemanticRouterReadiness supports a managed first installation: the
	// private Management listener must accept the first publication before the
	// Router can become inference-ready. The caller must finish by invoking
	// WaitForSemanticRouterReady after its Management client has bootstrapped.
	DeferSemanticRouterReadiness bool
}

// Stack composes the common gateway-based E2E deployment lifecycle.
type Stack struct {
	config Config
}

// New creates a stack with normalized defaults.
func New(config Config) *Stack {
	config.ServiceConfig = normalizeServiceConfig(config.ServiceConfig)
	if len(config.VerifyDeployments) == 0 {
		config.VerifyDeployments = append([]helpers.DeploymentRef(nil), defaultVerifyDeployments...)
	}
	if config.DeferSemanticRouterReadiness {
		config.VerifyDeployments = withoutDeployment(
			config.VerifyDeployments,
			helm.SemanticRouterRelease.Namespace,
			deploymentSemanticRouter,
		)
	}
	return &Stack{config: config}
}

// DefaultServiceConfig returns the common Envoy Gateway access configuration.
func DefaultServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		LabelSelector: "gateway.envoyproxy.io/owning-gateway-namespace=default,gateway.envoyproxy.io/owning-gateway-name=semantic-router",
		Namespace:     helm.EnvoyGatewayRelease.Namespace,
		ServicePort:   "80",
	}
}

// ServiceConfig returns the normalized service configuration for this stack.
func (s *Stack) ServiceConfig() framework.ServiceConfig {
	return s.config.ServiceConfig
}

// Setup deploys prerequisites, core releases, gateway resources, and shared readiness checks.
func (s *Stack) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	if err := s.ApplyPrerequisites(ctx, opts); err != nil {
		return err
	}
	if err := s.DeployCore(ctx, opts); err != nil {
		return err
	}
	if err := s.ApplyResources(ctx, opts); err != nil {
		return err
	}
	return s.Verify(ctx, opts)
}

// Teardown step names, in the order Teardown runs them.
const (
	teardownStepResources     = "resources"
	teardownStepCoreReleases  = "core releases"
	teardownStepPrerequisites = "prerequisites"
)

// teardownStep is one named, best-effort teardown action.
type teardownStep struct {
	name string
	run  func(context.Context, *framework.TeardownOptions) error
}

// teardownSteps returns the teardown actions in order. It mirrors setup in
// reverse - resources, then the Helm releases, then the prerequisites those
// releases were installed on top of.
//
// Prerequisites have to go last. A profile may create the release's own
// namespace there (jailbreak-onerror does, so its mapping ConfigMap exists
// before an install that blocks on --wait), and deleting that namespace before
// `helm uninstall` deletes the release secret with it: the uninstall then fails
// with "release: not found", which is swallowed as best-effort, leaving the
// chart's cluster-scoped RBAC orphaned.
func (s *Stack) teardownSteps() []teardownStep {
	return []teardownStep{
		{name: teardownStepResources, run: s.CleanupResources},
		{name: teardownStepCoreReleases, run: func(ctx context.Context, opts *framework.TeardownOptions) error {
			s.UninstallCore(ctx, opts)
			return nil
		}},
		{name: teardownStepPrerequisites, run: s.CleanupPrerequisites},
	}
}

// Teardown removes gateway resources, shared Helm releases, and prerequisites.
func (s *Stack) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	for _, step := range s.teardownSteps() {
		if err := step.run(ctx, opts); err != nil {
			s.log(opts.Verbose, "Warning: failed to cleanup %s: %v", step.name, err)
		}
	}
	return nil
}

// ApplyPrerequisites applies manifests that must exist before semantic-router is installed.
func (s *Stack) ApplyPrerequisites(ctx context.Context, opts *framework.SetupOptions) error {
	if len(s.config.PrerequisiteManifests) == 0 {
		return nil
	}
	s.log(opts.Verbose, "Applying prerequisite manifests")
	return s.applyManifests(ctx, opts.KubeConfig, s.config.PrerequisiteManifests)
}

// DeployCore installs semantic-router, Envoy Gateway, and Envoy AI Gateway.
func (s *Stack) DeployCore(ctx context.Context, opts *framework.SetupOptions) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	envoyGatewayRelease := helm.EnvoyGatewayRelease.Clone()
	aiGatewayCRDRelease := helm.AIGatewayCRDRelease.Clone()
	aiGatewayRelease := helm.AIGatewayRelease.Clone()

	s.log(opts.Verbose, "Installing semantic-router")
	if err := deployer.Install(ctx, s.semanticRouterInstallOptions(opts)); err != nil {
		return fmt.Errorf("install semantic-router: %w", err)
	}
	if !s.config.DeferSemanticRouterReadiness {
		if err := deployer.WaitForDeployment(ctx, helm.SemanticRouterRelease.Namespace, deploymentSemanticRouter, timeoutDeploymentWait); err != nil {
			return fmt.Errorf("wait for semantic-router: %w", err)
		}
	}

	s.log(opts.Verbose, "Installing Envoy Gateway")
	if err := deployer.Install(ctx, envoyGatewayRelease); err != nil {
		return fmt.Errorf("install Envoy Gateway: %w", err)
	}
	if err := deployer.WaitForDeployment(ctx, helm.EnvoyGatewayRelease.Namespace, deploymentEnvoyGateway, timeoutDeploymentWait); err != nil {
		return fmt.Errorf("wait for Envoy Gateway: %w", err)
	}

	s.log(opts.Verbose, "Installing Envoy AI Gateway")
	if err := deployer.Install(ctx, aiGatewayCRDRelease); err != nil {
		return fmt.Errorf("install Envoy AI Gateway CRDs: %w", err)
	}
	if err := deployer.Install(ctx, aiGatewayRelease); err != nil {
		return fmt.Errorf("install Envoy AI Gateway: %w", err)
	}
	if err := deployer.WaitForDeployment(ctx, helm.AIGatewayRelease.Namespace, deploymentAIGateway, timeoutDeploymentWait); err != nil {
		return fmt.Errorf("wait for Envoy AI Gateway: %w", err)
	}

	return nil
}

// ApplyResources applies the profile-specific gateway resources and waits for extra deployments.
func (s *Stack) ApplyResources(ctx context.Context, opts *framework.SetupOptions) error {
	if len(s.config.ResourceManifests) > 0 {
		s.log(opts.Verbose, "Applying gateway resource manifests")
		if err := s.applyManifests(ctx, opts.KubeConfig, s.config.ResourceManifests); err != nil {
			return err
		}
	}

	if len(s.config.WaitDeployments) == 0 {
		return nil
	}

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	for _, deployment := range s.config.WaitDeployments {
		s.log(opts.Verbose, "Waiting for deployment %s/%s", deployment.Namespace, deployment.Name)
		if err := deployer.WaitForDeployment(ctx, deployment.Namespace, deployment.Name, timeoutDeploymentWait); err != nil {
			return fmt.Errorf("wait for deployment %s/%s: %w", deployment.Namespace, deployment.Name, err)
		}
	}

	return nil
}

// Verify waits for the shared gateway service and the configured healthy deployments.
func (s *Stack) Verify(ctx context.Context, opts *framework.SetupOptions) error {
	if opts.KubeClient == nil {
		return fmt.Errorf("kube client is required for verification")
	}

	svcConfig := s.config.ServiceConfig
	serviceName, err := s.waitForService(ctx, opts.KubeClient, svcConfig, opts.Verbose)
	if err != nil {
		return err
	}
	s.log(opts.Verbose, "Gateway service is ready: %s", serviceName)

	s.log(opts.Verbose, "Verifying deployments")
	if err := helpers.VerifyDeployments(ctx, opts.KubeClient, s.config.VerifyDeployments, opts.Verbose); err != nil {
		return fmt.Errorf("required deployment not healthy: %w", err)
	}
	return nil
}

// WaitForSemanticRouterReady closes a deferred managed bootstrap. The Router
// chart probes /ready; requiring the complete expected rollout therefore proves
// every replica has crossed the inference-readiness gate after publication.
func (s *Stack) WaitForSemanticRouterReady(
	ctx context.Context,
	opts *framework.SetupOptions,
	expectedReplicas int32,
) error {
	if opts == nil || opts.KubeClient == nil {
		return fmt.Errorf("kube client is required for Router readiness verification")
	}
	s.log(opts.Verbose, "Waiting for semantic-router /ready on %d replicas", expectedReplicas)
	if err := helpers.WaitForDeploymentReadyReplicas(
		ctx,
		opts.KubeClient,
		helm.SemanticRouterRelease.Namespace,
		deploymentSemanticRouter,
		expectedReplicas,
		timeoutDeploymentWait,
		5*time.Second,
		opts.Verbose,
	); err != nil {
		return fmt.Errorf("wait for semantic-router publication readiness: %w", err)
	}
	return nil
}

// CleanupResources removes resource manifests in reverse order.
func (s *Stack) CleanupResources(ctx context.Context, opts *framework.TeardownOptions) error {
	return s.cleanupManifests(ctx, opts.KubeConfig, reverseCopy(s.config.ResourceManifests))
}

// CleanupPrerequisites removes prerequisite manifests in reverse order.
func (s *Stack) CleanupPrerequisites(ctx context.Context, opts *framework.TeardownOptions) error {
	return s.cleanupManifests(ctx, opts.KubeConfig, reverseCopy(s.config.PrerequisiteManifests))
}

// UninstallCore removes the shared Helm releases with best-effort cleanup semantics.
func (s *Stack) UninstallCore(ctx context.Context, opts *framework.TeardownOptions) {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	bestEffort := []helm.InstallOptions{
		helm.AIGatewayCRDRelease,
		helm.AIGatewayRelease,
		helm.EnvoyGatewayRelease,
		helm.SemanticRouterRelease,
	}
	for _, release := range bestEffort {
		if err := deployer.Uninstall(ctx, release.ReleaseName, release.Namespace); err != nil {
			s.log(opts.Verbose, "Warning: failed to uninstall %s/%s: %v", release.Namespace, release.ReleaseName, err)
		}
	}
}

func (s *Stack) semanticRouterInstallOptions(opts *framework.SetupOptions) helm.InstallOptions {
	installOptions := helm.SemanticRouterRelease.Clone()
	setValues := map[string]string{
		"image.repository": imageRepository,
		"image.tag":        opts.ImageTag,
		"image.pullPolicy": imagePullPolicy,
	}
	for key, value := range s.config.SemanticRouterSet {
		setValues[key] = value
	}

	valuesFiles := []string{s.config.SemanticRouterValuesFile}
	if extraValuesFile, ok := opts.ValuesFiles[helm.SemanticRouterRelease.ReleaseName]; ok && extraValuesFile != "" {
		valuesFiles = append(valuesFiles, extraValuesFile)
	}

	installOptions.ValuesFiles = valuesFiles
	installOptions.Set = setValues
	if s.config.DeferSemanticRouterReadiness {
		installOptions.Wait = false
		installOptions.Timeout = ""
	}
	return installOptions
}

func (s *Stack) waitForService(
	ctx context.Context,
	client *kubernetes.Clientset,
	svcConfig framework.ServiceConfig,
	verbose bool,
) (string, error) {
	serviceName := svcConfig.Name
	var err error
	if serviceName == "" {
		serviceName, err = helpers.WaitForServiceByLabelWithReadyPods(
			ctx,
			client,
			svcConfig.Namespace,
			svcConfig.LabelSelector,
			timeoutServiceRetry,
			intervalServiceRetry,
			verbose,
			func(format string, args ...interface{}) {
				s.log(verbose, format, args...)
			},
		)
		if err != nil {
			return "", fmt.Errorf("wait for gateway service: %w", err)
		}
		return serviceName, nil
	}

	if err := helpers.VerifyServicePodsRunning(ctx, client, svcConfig.Namespace, serviceName, verbose); err != nil {
		return "", fmt.Errorf("verify gateway service pods: %w", err)
	}
	return serviceName, nil
}

func (s *Stack) applyManifests(ctx context.Context, kubeConfig string, manifests []string) error {
	for _, manifest := range manifests {
		if err := s.runKubectl(ctx, kubeConfig, "apply", "-f", manifest); err != nil {
			return fmt.Errorf("apply manifest %s: %w", manifest, err)
		}
	}
	return nil
}

func (s *Stack) cleanupManifests(ctx context.Context, kubeConfig string, manifests []string) error {
	var errs []error
	for _, manifest := range manifests {
		if manifest == "" {
			continue
		}
		if err := s.runKubectl(ctx, kubeConfig, "delete", "-f", manifest, "--ignore-not-found=true"); err != nil {
			errs = append(errs, fmt.Errorf("delete manifest %s: %w", manifest, err))
		}
	}
	return errors.Join(errs...)
}

func (s *Stack) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append(args, "--kubeconfig", kubeConfig)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	return cmd.Run()
}

func (s *Stack) log(verbose bool, format string, args ...interface{}) {
	if verbose {
		fmt.Printf("[%s] "+format+"\n", append([]interface{}{s.config.Name}, args...)...)
	}
}

func normalizeServiceConfig(config framework.ServiceConfig) framework.ServiceConfig {
	defaults := DefaultServiceConfig()
	if config.LabelSelector == "" {
		config.LabelSelector = defaults.LabelSelector
	}
	if config.Namespace == "" {
		config.Namespace = defaults.Namespace
	}
	if config.ServicePort == "" && config.PortMapping == "" {
		config.ServicePort = defaults.ServicePort
	}
	return config
}

func reverseCopy(items []string) []string {
	if len(items) == 0 {
		return nil
	}
	out := append([]string(nil), items...)
	for left, right := 0, len(out)-1; left < right; left, right = left+1, right-1 {
		out[left], out[right] = out[right], out[left]
	}
	return out
}

func withoutDeployment(items []helpers.DeploymentRef, namespace, name string) []helpers.DeploymentRef {
	filtered := make([]helpers.DeploymentRef, 0, len(items))
	for _, item := range items {
		if item.Namespace == namespace && item.Name == name {
			continue
		}
		filtered = append(filtered, item)
	}
	return filtered
}
