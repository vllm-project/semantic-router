package ragllamastack

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"time"

	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	"github.com/vllm-project/semantic-router/e2e/pkg/helm"
	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"

	// Import testcases package to register all test cases via their init() functions
	_ "github.com/vllm-project/semantic-router/e2e/testcases"
)

const (
	// Chart and file path constants
	chartPathSemanticRouter = "deploy/helm/semantic-router"
	valuesFile              = "e2e/profiles/rag-llama-stack/values.yaml"
	llamaStackManifest      = "e2e/profiles/rag-llama-stack/manifests/llama-stack.yaml"

	// Timeout constants
	timeoutSemanticRouterInstall = "30m"
	timeoutDeploymentWait        = 30 * time.Minute
	timeoutLlamaStackReady       = 10 * time.Minute
	intervalServiceRetry         = 5 * time.Second
	timeoutHTTPRequest           = 60 * time.Second
	delayPortForwardReady        = 2 * time.Second

	// Image constants
	imageRepository = "ghcr.io/vllm-project/semantic-router/extproc"
	imagePullPolicy = "Never"
	llamaStackImage = "llamastack/distribution-starter:0.5.0"

	// Namespace constants
	namespaceSemanticRouter = "vllm-semantic-router-system"
)

// Profile implements the RAG with Llama Stack test profile
type Profile struct {
	verbose bool
}

// NewProfile creates a new RAG Llama Stack profile
func NewProfile() *Profile {
	return &Profile{}
}

// Name returns the profile name
func (p *Profile) Name() string {
	return "rag-llama-stack"
}

// Description returns the profile description
func (p *Profile) Description() string {
	return "Tests RAG vector store pipeline with Llama Stack backend"
}

// Setup deploys all required components for RAG Llama Stack testing.
// This profile does not deploy Envoy Gateway because the rag-vectorstore
// test connects directly to the Semantic Router API server (port 8080).
func (p *Profile) Setup(ctx context.Context, opts *framework.SetupOptions) error {
	p.verbose = opts.Verbose
	p.log("Setting up RAG Llama Stack test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Step 1: Deploy Llama Stack
	p.log("Step 1/3: Deploying Llama Stack")
	if err := p.deployLlamaStack(ctx, opts); err != nil {
		return fmt.Errorf("failed to deploy llama stack: %w", err)
	}

	// Step 2: Deploy Semantic Router
	p.log("Step 2/3: Deploying Semantic Router")
	if err := p.deploySemanticRouter(ctx, deployer, opts); err != nil {
		return fmt.Errorf("failed to deploy semantic router: %w", err)
	}

	// Step 3: Verify all components are ready
	p.log("Step 3/3: Verifying all components are ready")
	if err := p.verifyEnvironment(ctx, opts); err != nil {
		return fmt.Errorf("failed to verify environment: %w", err)
	}

	p.log("RAG Llama Stack test environment setup complete")
	return nil
}

// Teardown cleans up all deployed resources
func (p *Profile) Teardown(ctx context.Context, opts *framework.TeardownOptions) error {
	p.verbose = opts.Verbose
	p.log("Tearing down RAG Llama Stack test environment")

	deployer := helm.NewDeployer(opts.KubeConfig, opts.Verbose)

	// Clean up in reverse order
	p.log("Uninstalling Semantic Router")
	deployer.Uninstall(ctx, "semantic-router", namespaceSemanticRouter)

	p.log("Removing Llama Stack")
	p.kubectlDelete(ctx, opts.KubeConfig, llamaStackManifest)

	p.log("RAG Llama Stack test environment teardown complete")
	return nil
}

// GetTestCases returns the list of test cases for this profile
func (p *Profile) GetTestCases() []string {
	return []string{
		"rag-vectorstore",
	}
}

// GetServiceConfig returns the service configuration for accessing the deployed service.
func (p *Profile) GetServiceConfig() framework.ServiceConfig {
	return framework.ServiceConfig{
		Name:        "semantic-router",
		Namespace:   namespaceSemanticRouter,
		PortMapping: "8080:8080",
	}
}

func (p *Profile) deployLlamaStack(ctx context.Context, opts *framework.SetupOptions) error {
	// Pre-pull Llama Stack image and load into Kind cluster to avoid
	// slow or failing image pulls from within the cluster nodes
	p.log("Pulling Llama Stack image: %s", llamaStackImage)
	pullCmd := exec.CommandContext(ctx, "docker", "pull", llamaStackImage)
	if p.verbose {
		pullCmd.Stdout = os.Stdout
		pullCmd.Stderr = os.Stderr
	}
	if err := pullCmd.Run(); err != nil {
		return fmt.Errorf("failed to pull llama stack image: %w", err)
	}

	p.log("Loading Llama Stack image into Kind cluster: %s", opts.ClusterName)
	loadCmd := exec.CommandContext(ctx, "kind", "load", "docker-image", llamaStackImage, "--name", opts.ClusterName)
	if p.verbose {
		loadCmd.Stdout = os.Stdout
		loadCmd.Stderr = os.Stderr
	}
	if err := loadCmd.Run(); err != nil {
		return fmt.Errorf("failed to load llama stack image to Kind: %w", err)
	}

	// Apply Llama Stack Kubernetes manifests
	if err := p.kubectlApply(ctx, opts.KubeConfig, llamaStackManifest); err != nil {
		return fmt.Errorf("failed to apply llama stack manifests: %w", err)
	}

	// Wait for Llama Stack deployment to be ready
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kube client: %w", err)
	}

	startTime := time.Now()
	for {
		if err := helpers.CheckDeployment(ctx, client, "llama-stack-system", "llama-stack", p.verbose); err == nil {
			break
		}
		if time.Since(startTime) >= timeoutLlamaStackReady {
			return fmt.Errorf("llama stack deployment not ready after %v", timeoutLlamaStackReady)
		}

		p.log("Llama Stack not ready, retrying in %v...", intervalServiceRetry)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(intervalServiceRetry):
		}
	}

	// Register embedding model via port-forward
	p.log("Registering embedding model in Llama Stack")
	if err := p.registerEmbeddingModel(ctx, opts); err != nil {
		p.log("Warning: failed to register embedding model: %v", err)
		// Don't fail — model may already be registered
	}

	return nil
}

func (p *Profile) registerEmbeddingModel(ctx context.Context, opts *framework.SetupOptions) error {
	// Port-forward to Llama Stack to register the model
	cmd := exec.CommandContext(ctx, "kubectl",
		"--kubeconfig", opts.KubeConfig,
		"port-forward", "-n", "llama-stack-system", "svc/llama-stack", "18321:8321")
	if err := cmd.Start(); err != nil {
		return fmt.Errorf("failed to start port-forward: %w", err)
	}
	defer func() {
		if cmd.Process != nil {
			cmd.Process.Kill() //nolint:errcheck
		}
		cmd.Wait() //nolint:errcheck
	}()

	// Wait for port-forward to be ready
	time.Sleep(delayPortForwardReady)

	// Register the embedding model
	body, err := json.Marshal(map[string]interface{}{
		"model_id":    "all-MiniLM-L6-v2",
		"provider_id": "sentence-transformers",
		"model_type":  "embedding",
		"metadata":    map[string]interface{}{"embedding_dimension": 384},
	})
	if err != nil {
		return fmt.Errorf("failed to marshal model registration body: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "http://localhost:18321/v1/models", bytes.NewReader(body))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")

	httpClient := &http.Client{Timeout: timeoutHTTPRequest}
	resp, err := httpClient.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	if resp.StatusCode == http.StatusOK || resp.StatusCode == http.StatusConflict {
		p.log("Embedding model registered successfully")
		return nil
	}

	return fmt.Errorf("unexpected status %d when registering model", resp.StatusCode)
}

func (p *Profile) deploySemanticRouter(ctx context.Context, deployer *helm.Deployer, opts *framework.SetupOptions) error {
	installOpts := helm.InstallOptions{
		ReleaseName: "semantic-router",
		Chart:       chartPathSemanticRouter,
		Namespace:   namespaceSemanticRouter,
		ValuesFiles: []string{valuesFile},
		Set: map[string]string{
			"image.repository": imageRepository,
			"image.tag":        opts.ImageTag,
			"image.pullPolicy": imagePullPolicy,
		},
		Wait:    true,
		Timeout: timeoutSemanticRouterInstall,
	}

	if err := deployer.Install(ctx, installOpts); err != nil {
		return err
	}

	return deployer.WaitForDeployment(ctx, namespaceSemanticRouter, "semantic-router", timeoutDeploymentWait)
}

func (p *Profile) verifyEnvironment(ctx context.Context, opts *framework.SetupOptions) error {
	config, err := clientcmd.BuildConfigFromFlags("", opts.KubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	client, err := kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create kube client: %w", err)
	}

	p.log("Verifying all deployments are healthy...")

	if err := helpers.CheckDeployment(ctx, client, "llama-stack-system", "llama-stack", p.verbose); err != nil {
		return fmt.Errorf("llama-stack deployment not healthy: %w", err)
	}

	if err := helpers.CheckDeployment(ctx, client, namespaceSemanticRouter, "semantic-router", p.verbose); err != nil {
		return fmt.Errorf("semantic-router deployment not healthy: %w", err)
	}

	p.log("All deployments are healthy")
	return nil
}

func (p *Profile) kubectlApply(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "apply", "-f", manifest)
}

func (p *Profile) kubectlDelete(ctx context.Context, kubeConfig, manifest string) error {
	return p.runKubectl(ctx, kubeConfig, "delete", "--ignore-not-found", "-f", manifest)
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

func (p *Profile) log(format string, args ...interface{}) {
	if p.verbose {
		fmt.Printf("[RAG-Llama-Stack] "+format+"\n", args...)
	}
}
