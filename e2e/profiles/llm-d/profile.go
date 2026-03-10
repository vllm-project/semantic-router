package llmd

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	v1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
	"github.com/vllm-project/semantic-router/e2e/pkg/testmatrix"

	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	kindNamespace     = "default"
	semanticNamespace = "vllm-semantic-router-system"
	gatewayNamespace  = "istio-system"
	istioVersion      = "1.28.0"
	gatewayCRDURL     = "https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml"
	inferenceCRDURL   = "https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v1.1.0/manifests.yaml"
)

type Profile struct {
	verbose bool
}

type rollbackStack struct {
	funcs []func()
}

type apiCheck struct {
	groupVersion      string
	expectedResources []string
	optional          bool
}

var requiredAPIChecks = []apiCheck{
	{groupVersion: "gateway.networking.k8s.io/v1", expectedResources: []string{"gateways", "httproutes"}},
	{groupVersion: "inference.networking.k8s.io/v1", expectedResources: []string{"inferencepools"}},
	// EndpointPickerConfig CRD is optional in some environments; treat as best-effort.
	{groupVersion: "inference.networking.x-k8s.io/v1alpha1", expectedResources: []string{"endpointpickerconfigs"}, optional: true},
}

func NewProfile() *Profile {
	return &Profile{}
}

func (p *Profile) Name() string {
	return "llm-d"
}

func (p *Profile) Description() string {
	return "Tests Semantic Router with LLM-D distributed inference"
}

func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose

	fmt.Printf("[Profile] llm-d setup start (istio=%s, gatewayCRD=%s, inferenceCRD=%s)\\n",
		istioVersion, gatewayCRDURL, inferenceCRDURL)

	rollbacks := &rollbackStack{}

	istioctlPath, err := p.ensureIstioctl(ctx)
	if err != nil {
		return err
	}
	if p.verbose {
		fmt.Printf("[Profile] istioctl ready at %s\n", istioctlPath)
	}

	if err := p.setupGatewayAPIs(ctx, rollbacks); err != nil {
		return err
	}
	if err := p.installLLMDIstio(ctx, istioctlPath, rollbacks); err != nil {
		return failWithRollback(rollbacks, "install istio: %w", err)
	}
	if err := p.deployLLMDRuntime(ctx, opts, rollbacks); err != nil {
		return failWithRollback(rollbacks, "%w", err)
	}
	if err := p.deployRoutesAndWait(ctx, rollbacks); err != nil {
		return failWithRollback(rollbacks, "%w", err)
	}

	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return failWithRollback(rollbacks, "verify environment: %w", err)
	}

	if p.verbose {
		fmt.Println("[Profile] llm-d setup complete")
	}
	return nil
}

func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	fmt.Println("[Profile] llm-d teardown start")
	_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/httproute-services.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/inferencepool-llama.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/inferencepool-phi4.yaml")
	_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/inference-sim.yaml")
	_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/rbac.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/envoyfilter.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/destinationrule.yaml")
	_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/gateway.yaml")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	_ = deployer.Uninstall(ctx, "semantic-router", semanticNamespace)

	_ = p.uninstallIstio(ctx)
	_ = p.kubectlDelete(ctx, gatewayCRDURL)
	_ = p.kubectlDelete(ctx, inferenceCRDURL)
	fmt.Println("[Profile] llm-d teardown complete")

	return nil
}

func (p *Profile) GetTestCases() []string {
	return testmatrix.Combine(
		testmatrix.RouterSmoke,
		[]string{
			"llm-d-inference-gateway-health",
		},
	)
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		Name:        "inference-gateway-istio",
		Namespace:   kindNamespace,
		PortMapping: "8080:80",
	}
}

func (r *rollbackStack) Add(fn func()) {
	r.funcs = append(r.funcs, fn)
}

func (r *rollbackStack) Run() {
	for i := len(r.funcs) - 1; i >= 0; i-- {
		r.funcs[i]()
	}
}

func failWithRollback(rollbacks *rollbackStack, format string, err error) error {
	rollbacks.Run()
	return fmt.Errorf(format, err)
}

func (p *Profile) setupGatewayAPIs(ctx context.Context, rollbacks *rollbackStack) error {
	if err := p.applyManifestWithRollback(ctx, gatewayCRDURL, "applied gateway CRDs", rollbacks); err != nil {
		return fmt.Errorf("gateway CRDs: %w", err)
	}
	if err := p.applyManifestWithRollback(ctx, inferenceCRDURL, "applied inference CRDs", rollbacks); err != nil {
		return fmt.Errorf("inference CRDs: %w", err)
	}
	return nil
}

func (p *Profile) applyManifestWithRollback(ctx context.Context, target, successMsg string, rollbacks *rollbackStack) error {
	if err := p.kubectlApply(ctx, target); err != nil {
		return err
	}
	rollbacks.Add(func() { _ = p.kubectlDelete(ctx, target) })
	if p.verbose {
		fmt.Println("[Profile]", successMsg)
	}
	return nil
}

func (p *Profile) installLLMDIstio(ctx context.Context, istioctlPath string, rollbacks *rollbackStack) error {
	if err := p.installIstio(ctx, istioctlPath); err != nil {
		return err
	}
	rollbacks.Add(func() { _ = p.uninstallIstio(ctx) })
	if p.verbose {
		fmt.Println("[Profile] istio installed")
	}
	return nil
}

func (p *Profile) deployLLMDRuntime(ctx context.Context, opts *framework.SetupOptions, rollbacks *rollbackStack) error {
	if err := p.deploySemanticRouter(ctx, opts); err != nil {
		return fmt.Errorf("deploy semantic router: %w", err)
	}
	rollbacks.Add(func() {
		deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
		_ = deployer.Uninstall(ctx, "semantic-router", semanticNamespace)
	})
	if p.verbose {
		fmt.Println("[Profile] semantic-router deployed")
	}

	if err := p.deployInferenceSim(ctx, opts); err != nil {
		return fmt.Errorf("deploy inference sim: %w", err)
	}
	rollbacks.Add(func() { _ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/inference-sim.yaml") })
	if p.verbose {
		fmt.Println("[Profile] inference simulators deployed")
	}

	if err := p.deployLLMD(ctx); err != nil {
		return fmt.Errorf("deploy llm-d resources: %w", err)
	}
	rollbacks.Add(func() {
		_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/rbac.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/inferencepool-llama.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/inferencepool-phi4.yaml")
	})
	if p.verbose {
		fmt.Println("[Profile] llm-d schedulers and pools deployed")
	}
	return nil
}

func (p *Profile) deployRoutesAndWait(ctx context.Context, rollbacks *rollbackStack) error {
	if err := p.deployGatewayRoutes(ctx); err != nil {
		return fmt.Errorf("deploy gateway routes: %w", err)
	}
	rollbacks.Add(func() {
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/envoyfilter.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/destinationrule.yaml")
		_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/httproute-services.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/gateway.yaml")
	})
	if p.verbose {
		fmt.Println("[Profile] gateway routes deployed")
	}
	return p.waitForGatewayRoutes(ctx)
}

func (p *Profile) waitForGatewayRoutes(ctx context.Context) error {
	for _, routeName := range []string{"vsr-llama8b-svc", "vsr-phi4-mini-svc"} {
		if err := p.waitHTTPRouteAccepted(ctx, routeName, kindNamespace, 2*time.Minute); err != nil {
			return err
		}
		if err := p.waitHTTPRouteResolvedRefs(ctx, routeName, kindNamespace, 2*time.Minute); err != nil {
			return err
		}
	}
	return nil
}

func (p *Profile) ensureIstioctl(ctx context.Context) (string, error) {
	if path, err := exec.LookPath("istioctl"); err == nil {
		return path, nil
	}

	osPart := runtime.GOOS
	if osPart == "darwin" {
		osPart = "osx"
	}
	arch := runtime.GOARCH
	platform := fmt.Sprintf("%s-%s", osPart, arch)

	cacheDir := filepath.Join(os.TempDir(), "istioctl-"+istioVersion+"-"+platform)
	bin := filepath.Join(cacheDir, "istioctl")
	if _, err := os.Stat(bin); err == nil {
		return bin, nil
	}

	if err := os.MkdirAll(cacheDir, 0o755); err != nil {
		return "", err
	}

	url := fmt.Sprintf("https://github.com/istio/istio/releases/download/%s/istioctl-%s-%s.tar.gz", istioVersion, istioVersion, platform)
	tgz := filepath.Join(cacheDir, "istioctl.tgz")

	if err := p.runCmd(ctx, "curl", "-fL", "-o", tgz, url); err != nil {
		return "", err
	}
	if err := p.runCmd(ctx, "tar", "-xzf", tgz, "-C", cacheDir); err != nil {
		return "", err
	}
	if err := os.Chmod(bin, 0o755); err != nil {
		return "", err
	}
	return bin, nil
}

func (p *Profile) installIstio(ctx context.Context, istioctl string) error {
	return p.runCmd(ctx, istioctl, "install", "-y", "--set", "profile=minimal", "--set", "values.pilot.env.ENABLE_GATEWAY_API=true", "--set", "values.pilot.env.ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true")
}

func (p *Profile) uninstallIstio(ctx context.Context) error {
	istioctl, err := exec.LookPath("istioctl")
	if err != nil {
		return nil
	}
	return p.runCmd(ctx, istioctl, "x", "uninstall", "--purge", "-y")
}

func (p *Profile) deploySemanticRouter(ctx context.Context, opts *framework.SetupOptions) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	installOpts := helm.InstallOptions{
		ReleaseName: "semantic-router",
		Chart:       "deploy/helm/semantic-router",
		Namespace:   semanticNamespace,
		ValuesFiles: []string{"e2e/profiles/llm-d/values.yaml"},
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
	return deployer.WaitForDeployment(ctx, semanticNamespace, "semantic-router", 30*time.Minute)
}

func (p *Profile) deployInferenceSim(ctx context.Context, opts *framework.SetupOptions) error {
	return p.kubectlApply(ctx, "e2e/profiles/llm-d/manifests/inference-sim.yaml")
}

func (p *Profile) deployLLMD(ctx context.Context) error {
	if err := p.kubectlApply(ctx, "deploy/kubernetes/llmd-base/inferencepool-llama.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "deploy/kubernetes/llmd-base/inferencepool-phi4.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "e2e/profiles/llm-d/manifests/rbac.yaml"); err != nil {
		return err
	}
	return nil
}

func (p *Profile) deployGatewayRoutes(ctx context.Context) error {
	if err := p.kubectlApply(ctx, "deploy/kubernetes/istio/gateway.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "e2e/profiles/llm-d/manifests/httproute-services.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "deploy/kubernetes/istio/destinationrule.yaml"); err != nil {
		return err
	}
	if err := p.kubectlApply(ctx, "deploy/kubernetes/istio/envoyfilter.yaml"); err != nil {
		return err
	}
	// Ensure EnvoyFilter ext-proc matches Gateway listener context for this e2e run
	_ = p.patchEnvoyFilterForGateway(ctx)
	return nil
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	client, err := helpers.NewKubeClient(opts.KubeConfig)
	if err != nil {
		return err
	}
	if err := p.verifyRequiredAPIGroups(client); err != nil {
		return err
	}
	if err := p.waitForCriticalDeployments(ctx, opts); err != nil {
		return err
	}
	if err := p.verifyCriticalDeployments(ctx, client); err != nil {
		return err
	}
	if err := helpers.VerifyServicePodsRunning(ctx, client, kindNamespace, "inference-gateway-istio", p.verbose); err != nil {
		return err
	}
	return p.verifyInferencePoolEndpoints(ctx, client)
}

func (p *Profile) verifyRequiredAPIGroups(client *kubernetes.Clientset) error {
	for _, check := range requiredAPIChecks {
		if err := p.verifyAPIGroup(client, check); err != nil {
			return err
		}
	}
	return nil
}

func (p *Profile) verifyAPIGroup(client *kubernetes.Clientset, check apiCheck) error {
	resources, err := client.Discovery().ServerResourcesForGroupVersion(check.groupVersion)
	if err != nil {
		if check.optional {
			if p.verbose {
				fmt.Printf("[Verify] API group %s not found (optional): %v\n", check.groupVersion, err)
			}
			return nil
		}
		return fmt.Errorf("discover %s: %w", check.groupVersion, err)
	}

	found := make(map[string]bool, len(resources.APIResources))
	for _, resource := range resources.APIResources {
		found[resource.Name] = true
	}
	for _, resourceName := range check.expectedResources {
		if found[resourceName] {
			continue
		}
		if check.optional {
			if p.verbose {
				fmt.Printf("[Verify] Missing optional resource %s in %s\n", resourceName, check.groupVersion)
			}
			return nil
		}
		return fmt.Errorf("missing %s in %s", resourceName, check.groupVersion)
	}

	if p.verbose {
		fmt.Printf("[Verify] API group %s present with %v\n", check.groupVersion, check.expectedResources)
	}
	return nil
}

func (p *Profile) waitForCriticalDeployments(ctx context.Context, opts *framework.SetupOptions) error {
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	for _, deployment := range []helpers.DeploymentRef{
		{Namespace: semanticNamespace, Name: "semantic-router"},
		{Namespace: gatewayNamespace, Name: "istiod"},
		{Namespace: kindNamespace, Name: "vllm-llama3-8b-instruct"},
		{Namespace: kindNamespace, Name: "phi4-mini"},
		{Namespace: kindNamespace, Name: "llm-d-inference-scheduler-llama3-8b"},
		{Namespace: kindNamespace, Name: "llm-d-inference-scheduler-phi4-mini"},
		{Namespace: kindNamespace, Name: "inference-gateway-istio"},
	} {
		if err := deployer.WaitForDeployment(ctx, deployment.Namespace, deployment.Name, 10*time.Minute); err != nil {
			return fmt.Errorf("wait for deployment %s/%s: %w", deployment.Namespace, deployment.Name, err)
		}
	}
	return nil
}

func (p *Profile) verifyCriticalDeployments(ctx context.Context, client *kubernetes.Clientset) error {
	return helpers.VerifyDeployments(
		ctx,
		client,
		[]helpers.DeploymentRef{
			{Namespace: semanticNamespace, Name: "semantic-router"},
			{Namespace: gatewayNamespace, Name: "istiod"},
			{Namespace: kindNamespace, Name: "vllm-llama3-8b-instruct"},
			{Namespace: kindNamespace, Name: "phi4-mini"},
			{Namespace: kindNamespace, Name: "llm-d-inference-scheduler-llama3-8b"},
			{Namespace: kindNamespace, Name: "llm-d-inference-scheduler-phi4-mini"},
		},
		p.verbose,
	)
}

func (p *Profile) verifyInferencePoolEndpoints(ctx context.Context, client *kubernetes.Clientset) error {
	for _, serviceName := range []string{"vllm-llama3-8b-instruct", "phi4-mini"} {
		if err := p.checkInferencePoolEndpointReady(ctx, client, kindNamespace, serviceName, 2*time.Minute); err != nil {
			return err
		}
	}
	return nil
}

// Note: GAIE controller is shipped by some providers (e.g., kgateway, nginx-gateway) or via provider-specific enable flags.
// For Istio-based profile we rely on pilot env ENABLE_GATEWAY_API_INFERENCE_EXTENSION=true instead of a standalone controller manifest.

func (p *Profile) runCmdOutput(ctx context.Context, name string, args ...string) (string, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	return string(out), nil
}

func (p *Profile) waitHTTPRouteAccepted(ctx context.Context, name, ns string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		out, err := p.runCmdOutput(ctx, "kubectl", "get", "httproute", name, "-n", ns, "-o", "jsonpath={.status.parents[*].conditions[?(@.type==\"Accepted\")].status}")
		if err == nil && strings.Contains(out, "True") {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	if p.verbose {
		_ = p.runCmd(ctx, "kubectl", "-n", "gateway-inference-system", "logs", "deploy/gateway-api-inference-extension-controller", "--tail=100")
		_ = p.runCmd(ctx, "kubectl", "-n", "default", "logs", "deploy/inference-gateway-istio", "--tail=100")
	}
	return fmt.Errorf("HTTPRoute %s/%s not Accepted", ns, name)
}

func (p *Profile) waitHTTPRouteResolvedRefs(ctx context.Context, name, ns string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		out, err := p.runCmdOutput(ctx, "kubectl", "get", "httproute", name, "-n", ns, "-o", "jsonpath={.status.parents[*].conditions[?(@.type==\"ResolvedRefs\")].status}")
		if err == nil && strings.Contains(out, "True") {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	if p.verbose {
		_ = p.runCmd(ctx, "kubectl", "-n", "gateway-inference-system", "logs", "deploy/gateway-api-inference-extension-controller", "--tail=100")
	}
	return fmt.Errorf("HTTPRoute %s/%s not ResolvedRefs", ns, name)
}

func (p *Profile) checkInferencePoolEndpointReady(ctx context.Context, client *kubernetes.Clientset, ns, name string, timeout time.Duration) error {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		ep, err := client.CoreV1().Endpoints(ns).Get(ctx, name, v1.GetOptions{})
		if err != nil {
			return err
		}
		addrs := 0
		for _, s := range ep.Subsets {
			addrs += len(s.Addresses)
		}
		if addrs > 0 {
			return nil
		}
		time.Sleep(2 * time.Second)
	}
	return fmt.Errorf("endpoints %s/%s empty", ns, name)
}

func (p *Profile) runCmd(ctx context.Context, name string, args ...string) error {
	cmd := exec.CommandContext(ctx, name, args...)
	if p.verbose {
		cmd.Stdout = os.Stdout
		cmd.Stderr = os.Stderr
	}
	return cmd.Run()
}

func (p *Profile) kubectlApply(ctx context.Context, target string) error {
	return p.runCmd(ctx, "kubectl", "apply", "-f", target)
}

func (p *Profile) kubectlDelete(ctx context.Context, target string) error {
	return p.runCmd(ctx, "kubectl", "delete", "-f", target, "--ignore-not-found")
}
func (p *Profile) patchEnvoyFilterForGateway(ctx context.Context) error {
	// Add match.context=GATEWAY and listener.portNumber=80 to the first configPatch via JSON patch
	patch := `[
      {"op":"add","path":"/spec/configPatches/0/match/context","value":"GATEWAY"},
      {"op":"add","path":"/spec/configPatches/0/match/listener/portNumber","value":80}
    ]`
	return p.runCmd(ctx, "kubectl", "-n", "default", "patch", "envoyfilter", "semantic-router", "--type=json", "-p", patch)
}
