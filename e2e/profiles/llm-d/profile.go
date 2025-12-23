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
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

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

	rollback := []func(){}
	rollbackAll := func() {
		for i := len(rollback) - 1; i >= 0; i-- {
			rollback[i]()
		}
	}

	istioctlPath, err := p.ensureIstioctl(ctx)
	if err != nil {
		return err
	}
	if p.verbose {
		fmt.Printf("[Profile] istioctl ready at %s\n", istioctlPath)
	}

	if err := p.kubectlApply(ctx, gatewayCRDURL); err != nil {
		return fmt.Errorf("gateway CRDs: %w", err)
	}
	rollback = append(rollback, func() { _ = p.kubectlDelete(ctx, gatewayCRDURL) })
	if p.verbose {
		fmt.Println("[Profile] applied gateway CRDs")
	}
	if err := p.kubectlApply(ctx, inferenceCRDURL); err != nil {
		rollbackAll()
		return fmt.Errorf("inference CRDs: %w", err)
	}
	rollback = append(rollback, func() { _ = p.kubectlDelete(ctx, inferenceCRDURL) })
	if p.verbose {
		fmt.Println("[Profile] applied inference CRDs")
	}

	if err := p.installIstio(ctx, istioctlPath); err != nil {
		rollbackAll()
		return fmt.Errorf("install istio: %w", err)
	}
	rollback = append(rollback, func() { _ = p.uninstallIstio(ctx) })
	if p.verbose {
		fmt.Println("[Profile] istio installed")
	}

	if err := p.deploySemanticRouter(ctx, opts); err != nil {
		rollbackAll()
		return fmt.Errorf("deploy semantic router: %w", err)
	}
	rollback = append(rollback, func() {
		deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
		_ = deployer.Uninstall(ctx, "semantic-router", semanticNamespace)
	})
	if p.verbose {
		fmt.Println("[Profile] semantic-router deployed")
	}

	if err := p.deployInferenceSim(ctx, opts); err != nil {
		rollbackAll()
		return fmt.Errorf("deploy inference sim: %w", err)
	}
	rollback = append(rollback, func() { _ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/inference-sim.yaml") })
	if p.verbose {
		fmt.Println("[Profile] inference simulators deployed")
	}

	if err := p.deployLLMD(ctx); err != nil {
		rollbackAll()
		return fmt.Errorf("deploy llm-d resources: %w", err)
	}
	rollback = append(rollback, func() {
		_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/rbac.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-llama.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/dest-rule-epp-phi4.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/inferencepool-llama.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/llmd-base/inferencepool-phi4.yaml")
	})
	if p.verbose {
		fmt.Println("[Profile] llm-d schedulers and pools deployed")
	}

	if err := p.deployGatewayRoutes(ctx); err != nil {
		rollbackAll()
		return fmt.Errorf("deploy gateway routes: %w", err)
	}
	rollback = append(rollback, func() {
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/envoyfilter.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/destinationrule.yaml")
		_ = p.kubectlDelete(ctx, "e2e/profiles/llm-d/manifests/httproute-services.yaml")
		_ = p.kubectlDelete(ctx, "deploy/kubernetes/istio/gateway.yaml")
	})
	if p.verbose {
		fmt.Println("[Profile] gateway routes deployed")
	}

	if err := p.waitHTTPRouteAccepted(ctx, "vsr-llama8b-svc", "default", 2*time.Minute); err != nil {
		rollbackAll()
		return err
	}
	if err := p.waitHTTPRouteResolvedRefs(ctx, "vsr-llama8b-svc", "default", 2*time.Minute); err != nil {
		rollbackAll()
		return err
	}
	if err := p.waitHTTPRouteAccepted(ctx, "vsr-phi4-mini-svc", "default", 2*time.Minute); err != nil {
		rollbackAll()
		return err
	}
	if err := p.waitHTTPRouteResolvedRefs(ctx, "vsr-phi4-mini-svc", "default", 2*time.Minute); err != nil {
		rollbackAll()
		return err
	}

	if err := p.verifyEnvironment(ctx, opts); err != nil {
		rollbackAll()
		return fmt.Errorf("verify environment: %w", err)
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
	deployer.Uninstall(ctx, "semantic-router", semanticNamespace)

	_ = p.uninstallIstio(ctx)
	_ = p.kubectlDelete(ctx, gatewayCRDURL)
	_ = p.kubectlDelete(ctx, inferenceCRDURL)
	fmt.Println("[Profile] llm-d teardown complete")

	return nil
}

func (p *Profile) GetTestCases() []string {
	// Shared router testcases that we also want to validate in the llm-d environment
	shared := []string{
		"chat-completions-request",
		"chat-completions-stress-request",
		"chat-completions-progressive-stress",
		"domain-classify",
	}

	// For llm-d we currently only reuse shared router testcases.
	// llm-d-specific HA/traffic semantics are expected to be covered in LLM-D / infra tests.
	return shared
}

func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		Name:        "inference-gateway-istio",
		Namespace:   kindNamespace,
		PortMapping: "8080:80",
	}
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
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return err
	}
	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return err
	}

	// Verify required CRDs/APIs from Gateway API and Inference Extension are registered.
	type apiCheck struct {
		groupVersion      string
		expectedResources []string
		optional          bool
	}
	checkAPIGroup := func(c apiCheck) error {
		resources, err := client.Discovery().ServerResourcesForGroupVersion(c.groupVersion)
		if err != nil {
			if c.optional {
				if p.verbose {
					fmt.Printf("[Verify] API group %s not found (optional): %v\n", c.groupVersion, err)
				}
				return nil
			}
			return fmt.Errorf("discover %s: %w", c.groupVersion, err)
		}
		found := make(map[string]bool, len(resources.APIResources))
		for _, r := range resources.APIResources {
			found[r.Name] = true
		}
		for _, r := range c.expectedResources {
			if !found[r] {
				if c.optional {
					if p.verbose {
						fmt.Printf("[Verify] Missing optional resource %s in %s\n", r, c.groupVersion)
					}
					return nil
				}
				return fmt.Errorf("missing %s in %s", r, c.groupVersion)
			}
		}
		if p.verbose {
			fmt.Printf("[Verify] API group %s present with %v\n", c.groupVersion, c.expectedResources)
		}
		return nil
	}

	for _, c := range []apiCheck{
		{groupVersion: "gateway.networking.k8s.io/v1", expectedResources: []string{"gateways", "httproutes"}},
		{groupVersion: "inference.networking.k8s.io/v1", expectedResources: []string{"inferencepools"}},
		// EndpointPickerConfig CRD is optional in some environments; treat as best-effort.
		{groupVersion: "inference.networking.x-k8s.io/v1alpha1", expectedResources: []string{"endpointpickerconfigs"}, optional: true},
	} {
		if err := checkAPIGroup(c); err != nil {
			return err
		}
	}

	// endpoints readiness check moved after deployments ready

	// Actively wait for critical deployments to become Available before checking readiness counts.
	// This avoids flakiness when resources are still pulling images just after creation.
	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)
	deploymentsToWait := []struct {
		ns, name string
	}{
		{semanticNamespace, "semantic-router"},
		{gatewayNamespace, "istiod"},
		{"default", "vllm-llama3-8b-instruct"},
		{"default", "phi4-mini"},
		{"default", "llm-d-inference-scheduler-llama3-8b"},
		{"default", "llm-d-inference-scheduler-phi4-mini"},
		{"default", "inference-gateway-istio"},
	}
	for _, d := range deploymentsToWait {
		if err := deployer.WaitForDeployment(ctx, d.ns, d.name, 10*time.Minute); err != nil {
			return fmt.Errorf("wait for deployment %s/%s: %w", d.ns, d.name, err)
		}
	}

	if err := helpers.CheckDeployment(ctx, client, semanticNamespace, "semantic-router", p.verbose); err != nil {
		return err
	}
	if err := helpers.CheckDeployment(ctx, client, gatewayNamespace, "istiod", p.verbose); err != nil {
		return err
	}
	if err := helpers.CheckDeployment(ctx, client, "default", "vllm-llama3-8b-instruct", p.verbose); err != nil {
		return err
	}
	if err := helpers.CheckDeployment(ctx, client, "default", "phi4-mini", p.verbose); err != nil {
		return err
	}
	if err := helpers.CheckDeployment(ctx, client, "default", "llm-d-inference-scheduler-llama3-8b", p.verbose); err != nil {
		return err
	}
	if err := helpers.CheckDeployment(ctx, client, "default", "llm-d-inference-scheduler-phi4-mini", p.verbose); err != nil {
		return err
	}
	if err := helpers.VerifyServicePodsRunning(ctx, client, "default", "inference-gateway-istio", p.verbose); err != nil {
		return err
	}
	if err := p.checkInferencePoolEndpointReady(ctx, client, "default", "vllm-llama3-8b-instruct", 2*time.Minute); err != nil {
		return err
	}
	if err := p.checkInferencePoolEndpointReady(ctx, client, "default", "phi4-mini", 2*time.Minute); err != nil {
		return err
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
