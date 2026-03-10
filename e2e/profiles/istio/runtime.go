package istio

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
)

const (
	timeoutIstioInstall         = 5 * time.Minute
	timeoutSemanticRouterDeploy = 20 * time.Minute
	timeoutGatewayReady         = 5 * time.Minute
	timeoutStabilization        = 60 * time.Second
)

func (p *Profile) failSetup(ctx context.Context, opts *framework.SetupOptions, state *setupState, err error) error {
	p.log("ERROR: %v", err)
	p.cleanupPartialDeployment(ctx, opts, state)
	return err
}

func (p *Profile) setupMeshControlPlane(
	ctx context.Context,
	opts *framework.SetupOptions,
	deployer *helm.Deployer,
	state *setupState,
) error {
	p.log("Step 1/9: Installing Istio control plane (version: %s)", p.istioVersion)
	if err := p.installIstio(ctx, opts); err != nil {
		return fmt.Errorf("failed to install Istio: %w", err)
	}
	state.istioInstalled = true

	p.log("Step 2/9: Configuring namespace for sidecar injection")
	if err := p.configureNamespace(ctx, opts); err != nil {
		return fmt.Errorf("failed to configure namespace: %w", err)
	}
	state.namespaceConfigured = true

	p.log("Step 3/9: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}
	state.semanticRouterDeployed = true
	return nil
}

func (p *Profile) setupGatewayPlane(
	ctx context.Context,
	opts *framework.SetupOptions,
	deployer *helm.Deployer,
	state *setupState,
) error {
	p.log("Step 4/9: Deploying Envoy Gateway")
	if err := p.deployEnvoyGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy gateway: %w", err)
	}
	state.envoyGatewayDeployed = true

	p.log("Step 5/9: Deploying Envoy AI Gateway")
	if err := p.deployEnvoyAIGateway(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy envoy ai gateway: %w", err)
	}
	state.envoyAIGatewayDeployed = true

	p.log("Step 6/9: Deploying Demo LLM backend")
	if err := p.deployDemoLLM(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy demo LLM: %w", err)
	}
	state.demoLLMDeployed = true

	p.log("Step 7/9: Deploying Gateway API Resources")
	if err := p.deployGatewayResources(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy gateway resources: %w", err)
	}
	state.gatewayResources = true
	return nil
}

func (p *Profile) finishSetup(ctx context.Context, opts *framework.SetupOptions, _ *setupState) error {
	p.log("Step 8/9: Creating Istio resources for service mesh testing")
	if err := p.createIstioResources(ctx, opts); err != nil {
		p.log("Warning: Failed to create Istio resources (non-critical): %v", err)
	}

	p.log("Step 9/9: Verifying environment")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}
	return nil
}

func (p *Profile) installIstio(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Installing Istio with Helm (version: %s)", p.istioVersion)

	deployer := helm.NewDeployer(opts.KubeConfig, p.verbose)
	baseOpts := helm.InstallOptions{
		ReleaseName: "istio-base",
		Chart:       fmt.Sprintf("https://istio-release.storage.googleapis.com/charts/base-%s.tgz", p.istioVersion),
		Namespace:   istioNamespace,
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, baseOpts); err != nil {
		return fmt.Errorf("failed to install Istio base: %w", err)
	}

	istiodOpts := helm.InstallOptions{
		ReleaseName: "istiod",
		Chart:       fmt.Sprintf("https://istio-release.storage.googleapis.com/charts/istiod-%s.tgz", p.istioVersion),
		Namespace:   istioNamespace,
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, istiodOpts); err != nil {
		return fmt.Errorf("failed to install Istiod: %w", err)
	}

	p.log("Waiting for istiod to be ready...")
	if err := p.waitForDeployment(ctx, opts, istioNamespace, "istiod", timeoutIstioInstall); err != nil {
		return err
	}

	gatewayOpts := helm.InstallOptions{
		ReleaseName: "istio-ingressgateway",
		Chart:       fmt.Sprintf("https://istio-release.storage.googleapis.com/charts/gateway-%s.tgz", p.istioVersion),
		Namespace:   istioNamespace,
		Wait:        false,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, gatewayOpts); err != nil {
		return fmt.Errorf("failed to install Istio Ingress Gateway: %w", err)
	}

	p.log("Waiting for Istio Ingress Gateway to be ready...")
	return p.waitForDeployment(ctx, opts, istioNamespace, istioIngressGateway, timeoutGatewayReady)
}

func (p *Profile) configureNamespace(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Creating namespace: %s", semanticRouterNamespace)
	createCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"create", "namespace", semanticRouterNamespace)
	if p.verbose {
		createCmd.Stdout = os.Stdout
		createCmd.Stderr = os.Stderr
	}
	if err := createCmd.Run(); err != nil {
		p.log("Warning: Namespace creation failed (may already exist): %v", err)
	}

	p.log("Enabling automatic sidecar injection for namespace: %s", semanticRouterNamespace)
	labelCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"label", "namespace", semanticRouterNamespace, "istio-injection=enabled", "--overwrite")
	if p.verbose {
		labelCmd.Stdout = os.Stdout
		labelCmd.Stderr = os.Stderr
	}
	if err := labelCmd.Run(); err != nil {
		return fmt.Errorf("failed to label namespace for sidecar injection: %w", err)
	}
	return nil
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: semanticRouterDeployment,
		Chart:       "deploy/helm/semantic-router",
		Namespace:   semanticRouterNamespace,
		ValuesFiles: []string{"e2e/profiles/ai-gateway/values.yaml"},
		Set: map[string]string{
			"image.repository": "ghcr.io/vllm-project/semantic-router/extproc",
			"image.tag":        opts.ImageTag,
			"image.pullPolicy": "Never",
		},
		Wait:    true,
		Timeout: "30m",
	}
	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	p.log("Waiting for Semantic Router deployment to be ready...")
	if err := deployer.WaitForDeployment(ctx, semanticRouterNamespace, semanticRouterDeployment, timeoutSemanticRouterDeploy); err != nil {
		return err
	}

	p.log("Verifying Istio sidecar injection...")
	return p.verifySidecarInjection(ctx, opts)
}

func (p *Profile) deployEnvoyGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: "eg",
		Chart:       "oci://docker.io/envoyproxy/gateway-helm",
		Namespace:   "envoy-gateway-system",
		Version:     "v1.6.0",
		ValuesFiles: []string{"https://raw.githubusercontent.com/envoyproxy/ai-gateway/main/manifests/envoy-gateway-values.yaml"},
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}
	return deployer.WaitForDeployment(ctx, "envoy-gateway-system", "envoy-gateway", 10*time.Minute)
}

func (p *Profile) deployEnvoyAIGateway(ctx context.Context, deployer *helm.Deployer, _ *framework.SetupOptions) error {
	crdOpts := helm.InstallOptions{
		ReleaseName: "aieg-crd",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-crds-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, crdOpts); err != nil {
		return err
	}

	installOpts := helm.InstallOptions{
		ReleaseName: "aieg",
		Chart:       "oci://docker.io/envoyproxy/ai-gateway-helm",
		Namespace:   "envoy-ai-gateway-system",
		Version:     "v0.4.0",
		Wait:        true,
		Timeout:     "10m",
	}
	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}
	return deployer.WaitForDeployment(ctx, "envoy-ai-gateway-system", "ai-gateway-controller", 10*time.Minute)
}

func (p *Profile) deployGatewayResources(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Applying Gateway API resources...")

	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml"); err != nil {
		return fmt.Errorf("failed to apply base model: %w", err)
	}
	if err := p.kubectlApply(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/gwapi-resources.yaml"); err != nil {
		return fmt.Errorf("failed to apply gateway API resources: %w", err)
	}

	client, err := helpers.NewKubeClient(opts.KubeConfig)
	if err != nil {
		return err
	}

	p.log("Waiting for Envoy Gateway service pods to be ready...")
	_, err = helpers.WaitForServiceByLabelWithReadyPods(
		ctx,
		client,
		"envoy-gateway-system",
		p.GetServiceConfig().LabelSelector,
		5*time.Minute,
		5*time.Second,
		p.verbose,
		p.log,
	)
	return err
}

func (p *Profile) cleanupGatewayResources(ctx context.Context, opts *framework.TeardownOptions) {
	_ = p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/httproute.yaml")
	_ = p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/gateway.yaml")
}

func (p *Profile) createIstioResources(ctx context.Context, opts *framework.SetupOptions) error {
	gatewayYAML := `apiVersion: networking.istio.io/v1beta1
kind: Gateway
metadata:
  name: semantic-router-gateway
  namespace: ` + semanticRouterNamespace + `
spec:
  selector:
    istio: ingressgateway
  servers:
  - port:
      number: 80
      name: http
      protocol: HTTP
    hosts:
    - "*"
---
apiVersion: networking.istio.io/v1beta1
kind: VirtualService
metadata:
  name: semantic-router
  namespace: ` + semanticRouterNamespace + `
spec:
  hosts:
  - "*"
  gateways:
  - semantic-router-gateway
  http:
  - match:
    - uri:
        prefix: /v1
    route:
    - destination:
        host: ` + semanticRouterService + `
        port:
          number: 8080
---
apiVersion: networking.istio.io/v1beta1
kind: DestinationRule
metadata:
  name: semantic-router
  namespace: ` + semanticRouterNamespace + `
spec:
  host: ` + semanticRouterService + `
  trafficPolicy:
    tls:
      mode: ISTIO_MUTUAL
`

	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig, "apply", "-f", "-")
	cmd.Stdin = strings.NewReader(gatewayYAML)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to create Istio resources: %w", err)
	}

	p.log("Waiting for Istio ingress gateway to be ready...")
	return p.waitForDeployment(ctx, opts, istioNamespace, istioIngressGateway, timeoutGatewayReady)
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Verifying istiod is running...")
	if err := p.verifyDeployment(ctx, opts, istioNamespace, "istiod"); err != nil {
		return fmt.Errorf("istiod verification failed: %w", err)
	}

	p.log("Verifying Istio ingress gateway is running...")
	if err := p.verifyDeployment(ctx, opts, istioNamespace, istioIngressGateway); err != nil {
		return fmt.Errorf("ingress gateway verification failed: %w", err)
	}

	p.log("Verifying Semantic Router is running...")
	if err := p.verifyDeployment(ctx, opts, semanticRouterNamespace, semanticRouterDeployment); err != nil {
		return fmt.Errorf("semantic router verification failed: %w", err)
	}

	p.log("Verifying sidecar injection...")
	if err := p.verifySidecarInjection(ctx, opts); err != nil {
		return fmt.Errorf("sidecar injection verification failed: %w", err)
	}

	p.log("Verifying Semantic Router service health...")
	if err := p.verifyServiceHealth(ctx, opts); err != nil {
		p.log("Warning: Service health check failed: %v", err)
		p.log("This may cause traffic routing tests to fail")
	}

	p.log("Allowing %v for environment stabilization...", timeoutStabilization)
	time.Sleep(timeoutStabilization)
	p.log("Environment verification complete")
	return nil
}

func (p *Profile) verifyServiceHealth(ctx context.Context, opts *framework.SetupOptions) error {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "pods",
		"-n", semanticRouterNamespace,
		"-l", "app.kubernetes.io/name=semantic-router",
		"-o", "jsonpath={.items[*].status.containerStatuses[*].ready}")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("failed to check pod readiness: %w (output: %s)", err, string(output))
	}

	readyStatus := strings.TrimSpace(string(output))
	if !strings.Contains(readyStatus, "true") {
		return fmt.Errorf("semantic-router pod containers not all ready: %s", readyStatus)
	}
	readyCount := strings.Count(readyStatus, "true")
	if readyCount < 2 {
		return fmt.Errorf("expected 2 ready containers (main + sidecar), got %d", readyCount)
	}

	p.log("Semantic Router service health check passed: %d/2 containers ready", readyCount)
	p.log("Waiting additional 10s for service to be fully ready...")
	time.Sleep(10 * time.Second)
	return nil
}

func (p *Profile) verifySidecarInjection(ctx context.Context, opts *framework.SetupOptions) error {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "pods",
		"-n", semanticRouterNamespace,
		"-l", "app.kubernetes.io/name=semantic-router",
		"-o", "jsonpath={.items[0].spec.containers[*].name}")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get pod containers: %w", err)
	}

	containers := string(output)
	if !strings.Contains(containers, "istio-proxy") {
		return fmt.Errorf("istio-proxy sidecar not found in pod. Containers: %s", containers)
	}
	p.log("✓ Istio sidecar successfully injected")
	return nil
}

func (p *Profile) waitForDeployment(ctx context.Context, opts *framework.SetupOptions, namespace, name string, timeout time.Duration) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	return deployer.WaitForDeployment(ctx, namespace, name, timeout)
}

func (p *Profile) verifyDeployment(ctx context.Context, opts *framework.SetupOptions, namespace, name string) error {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"get", "deployment", name, "-n", namespace, "-o", "jsonpath={.status.readyReplicas}")
	output, err := cmd.Output()
	if err != nil {
		return fmt.Errorf("failed to get deployment status: %w", err)
	}
	if string(output) == "0" || string(output) == "" {
		return fmt.Errorf("deployment %s/%s has no ready replicas", namespace, name)
	}
	p.log("✓ Deployment %s/%s is ready", namespace, name)
	return nil
}

func (p *Profile) cleanupPartialDeployment(ctx context.Context, opts *framework.SetupOptions, state *setupState) {
	p.log("Cleaning up partial deployment...")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	teardownOpts := &framework.TeardownOptions{
		KubeClient:  opts.KubeClient,
		KubeConfig:  opts.KubeConfig,
		ClusterName: opts.ClusterName,
		Verbose:     opts.Verbose,
	}

	if state.gatewayResources {
		p.log("Cleaning up Gateway API resources")
		p.cleanupGatewayResources(ctx, teardownOpts)
	}
	if state.demoLLMDeployed {
		p.log("Cleaning up Demo LLM")
		_ = p.kubectlDelete(ctx, opts.KubeConfig, "deploy/kubernetes/ai-gateway/aigw-resources/base-model.yaml")
	}
	if state.envoyAIGatewayDeployed {
		_ = deployer.Uninstall(ctx, "aieg", "envoy-ai-gateway-system")
		_ = deployer.Uninstall(ctx, "aieg-crd", "envoy-ai-gateway-system")
	}
	if state.envoyGatewayDeployed {
		_ = deployer.Uninstall(ctx, "eg", "envoy-gateway-system")
	}
	if state.semanticRouterDeployed {
		_ = deployer.Uninstall(ctx, semanticRouterDeployment, semanticRouterNamespace)
	}
	if state.namespaceConfigured {
		p.removeSidecarInjection(ctx, teardownOpts)
	}
	if state.istioInstalled {
		p.uninstallIstio(ctx, teardownOpts)
	}
}

func (p *Profile) removeSidecarInjection(ctx context.Context, opts *framework.TeardownOptions) {
	cmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"label", "namespace", semanticRouterNamespace, "istio-injection-")
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	if err := cmd.Run(); err != nil {
		p.log("Warning: Failed to remove sidecar injection label: %v", err)
	}
}

func (p *Profile) uninstallIstio(ctx context.Context, opts *framework.TeardownOptions) {
	deployer := helm.NewDeployer(opts.KubeConfig, p.verbose)

	p.log("Uninstalling Istio Ingress Gateway...")
	_ = deployer.Uninstall(ctx, "istio-ingressgateway", istioNamespace)
	p.log("Uninstalling Istiod...")
	_ = deployer.Uninstall(ctx, "istiod", istioNamespace)
	p.log("Uninstalling Istio base...")
	_ = deployer.Uninstall(ctx, "istio-base", istioNamespace)

	p.log("Deleting istio-system namespace...")
	deleteNsCmd := exec.CommandContext(ctx, "kubectl", "--kubeconfig", opts.KubeConfig,
		"delete", "namespace", istioNamespace, "--ignore-not-found", "--timeout=60s")
	if p.verbose {
		deleteNsCmd.Stdout = os.Stdout
		deleteNsCmd.Stderr = os.Stderr
	}
	if err := deleteNsCmd.Run(); err != nil {
		p.log("Warning: Failed to delete istio-system namespace: %v", err)
	}
}

func (p *Profile) deployDemoLLM(ctx context.Context, opts *framework.SetupOptions) error {
	p.log("Demo LLM will be deployed with Gateway API resources")
	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "--server-side", "-f", manifest)
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "-f", manifest)
}

func (p *Profile) runKubectl(ctx context.Context, kubeConfig string, args ...string) error {
	args = append([]string{"--kubeconfig", kubeConfig}, args...)
	cmd := exec.CommandContext(ctx, "kubectl", args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}
