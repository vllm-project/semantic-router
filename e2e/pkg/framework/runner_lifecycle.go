package framework

import (
	"context"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/cluster"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/clientcmd"
)

type runState struct {
	exitCode    int
	kubeClient  *kubernetes.Clientset
	kubeConfig  string
	valuesFiles map[string]string
	cleanup     []func()
}

func (s *runState) addCleanup(fn func()) {
	s.cleanup = append(s.cleanup, fn)
}

func (s *runState) runCleanup() {
	for i := len(s.cleanup) - 1; i >= 0; i-- {
		s.cleanup[i]()
	}
}

type describedProfile interface {
	Description() string
}

// Run executes the E2E tests.
func (r *Runner) Run(ctx context.Context) error {
	state := &runState{}

	r.log("Starting E2E tests for profile: %s", r.profile.Name())
	r.logProfileDescription()
	r.initializeReport()
	defer r.finalizeReport(state)
	defer state.runCleanup()

	if err := r.prepareRuntime(ctx, state); err != nil {
		state.exitCode = 1
		return err
	}

	if r.opts.SetupOnly {
		r.logSetupOnlyHints()
		return nil
	}

	results, err := r.runTests(ctx, state.kubeClient)
	if err != nil {
		state.exitCode = 1
		return fmt.Errorf("failed to run tests: %w", err)
	}

	if err := r.finishRun(ctx, state, results); err != nil {
		state.exitCode = 1
		r.logPreservedEnvironmentHints()
		return err
	}

	r.log("✅ All tests passed!")
	r.logPreservedEnvironmentHints()
	return nil
}

func (r *Runner) logProfileDescription() {
	profile, ok := r.profile.(describedProfile)
	if !ok {
		return
	}
	r.log("Description: %s", profile.Description())
}

func (r *Runner) initializeReport() {
	r.reporter = NewReportGenerator(r.opts.Profile, r.opts.ClusterName)
	r.reporter.SetEnvironment("go_version", "1.24")
	r.reporter.SetEnvironment("verbose", fmt.Sprintf("%v", r.opts.Verbose))
	r.reporter.SetEnvironment("parallel", fmt.Sprintf("%v", r.opts.Parallel))
	r.reporter.SetEnvironment("use_workspace_models", fmt.Sprintf("%v", r.opts.UseWorkspaceModels))
}

func (r *Runner) finalizeReport(state *runState) {
	r.reporter.Finalize(state.exitCode)

	if err := r.reporter.WriteJSON("test-report.json"); err != nil {
		r.log("Warning: failed to write JSON report: %v", err)
	} else {
		r.log("Test report written to: test-report.json")
	}

	if err := r.reporter.WriteMarkdown("test-report.md"); err != nil {
		r.log("Warning: failed to write Markdown report: %v", err)
	} else {
		r.log("Test report written to: test-report.md")
	}
}

func (r *Runner) prepareRuntime(ctx context.Context, state *runState) error {
	if err := r.prepareCluster(ctx, state); err != nil {
		return err
	}
	if err := r.buildAndLoadImages(ctx); err != nil {
		return fmt.Errorf("failed to build and load images: %w", err)
	}
	if err := r.prepareKubeClient(ctx, state); err != nil {
		return err
	}

	r.configureHFTokenSecret(ctx, state.kubeClient)
	if err := r.prepareWorkspaceModelsValues(state); err != nil {
		return err
	}

	if err := r.setupProfile(ctx, state); err != nil {
		return err
	}

	return nil
}

func (r *Runner) prepareCluster(ctx context.Context, state *runState) error {
	if r.opts.UseExistingCluster {
		if r.opts.UseWorkspaceModels {
			r.log("Workspace model reuse requires the existing cluster to have been created with -use-workspace-models enabled")
		}
		return nil
	}

	if err := r.setupCluster(ctx); err != nil {
		return fmt.Errorf("failed to setup cluster: %w", err)
	}

	if !r.opts.KeepCluster {
		state.addCleanup(func() {
			r.cleanupClusterWithLogs(state)
		})
	}

	return nil
}

func (r *Runner) cleanupClusterWithLogs(state *runState) {
	if state.kubeClient != nil {
		r.log("📝 Collecting semantic-router logs before cluster cleanup...")
		logCtx, logCancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer logCancel()
		if err := r.collectSemanticRouterLogs(logCtx, state.kubeClient); err != nil {
			r.log("Warning: failed to collect semantic-router logs before cleanup: %v", err)
		}
	}
	r.cleanupCluster(context.Background())
}

func (r *Runner) prepareKubeClient(ctx context.Context, state *runState) error {
	kubeConfig, err := r.cluster.GetKubeConfig(ctx)
	if err != nil {
		return fmt.Errorf("failed to get kubeconfig: %w", err)
	}
	state.kubeConfig = kubeConfig

	config, err := clientcmd.BuildConfigFromFlags("", kubeConfig)
	if err != nil {
		return fmt.Errorf("failed to build kubeconfig: %w", err)
	}

	r.restConfig = config
	state.kubeClient, err = kubernetes.NewForConfig(config)
	if err != nil {
		return fmt.Errorf("failed to create Kubernetes client: %w", err)
	}

	r.reporter.SetKubeClient(state.kubeClient)
	return nil
}

func (r *Runner) configureHFTokenSecret(ctx context.Context, kubeClient *kubernetes.Clientset) {
	if os.Getenv("HF_TOKEN") == "" {
		r.log("ℹ️  HF_TOKEN not set - gated models (e.g., embeddinggemma-300m) may not be downloadable")
		return
	}

	if err := r.createHFTokenSecret(ctx, kubeClient); err != nil {
		r.log("⚠️  Warning: Failed to create HF_TOKEN secret: %v", err)
		r.log("   Model downloads may fail if gated models (e.g., embeddinggemma-300m) are required")
		return
	}

	r.log("✅ Created HF_TOKEN secret for gated model downloads")
}

func (r *Runner) setupProfile(ctx context.Context, state *runState) error {
	if r.opts.SkipSetup {
		r.log("⏭️  Skipping profile setup (--skip-setup enabled)")
		return nil
	}

	setupOpts := &SetupOptions{
		KubeClient:  state.kubeClient,
		KubeConfig:  state.kubeConfig,
		ClusterName: r.opts.ClusterName,
		ImageTag:    r.opts.ImageTag,
		Verbose:     r.opts.Verbose,
		ValuesFiles: state.valuesFiles,
	}

	if err := r.profile.Setup(ctx, setupOpts); err != nil {
		return fmt.Errorf("failed to setup profile: %w", err)
	}

	if !r.opts.SetupOnly && !r.opts.KeepCluster {
		state.addCleanup(func() {
			r.teardownProfile(state)
		})
	}

	return nil
}

func (r *Runner) prepareWorkspaceModelsValues(state *runState) error {
	if !r.opts.UseWorkspaceModels || r.opts.SkipSetup {
		return nil
	}

	valuesFile, cleanup, err := createWorkspaceModelsValuesFile()
	if err != nil {
		return err
	}

	if state.valuesFiles == nil {
		state.valuesFiles = make(map[string]string)
	}
	state.valuesFiles["semantic-router"] = valuesFile
	state.addCleanup(cleanup)
	r.log("Using semantic-router workspace models overlay: %s", valuesFile)
	return nil
}

func (r *Runner) teardownProfile(state *runState) {
	teardownOpts := &TeardownOptions{
		KubeClient:  state.kubeClient,
		KubeConfig:  state.kubeConfig,
		ClusterName: r.opts.ClusterName,
		Verbose:     r.opts.Verbose,
	}
	if err := r.profile.Teardown(context.Background(), teardownOpts); err != nil {
		r.log("Warning: failed to teardown profile: %v", err)
	}
}

func (r *Runner) logSetupOnlyHints() {
	r.log("✅ Profile setup complete (--setup-only mode)")
	r.log("💡 Cluster is ready. You can now:")
	r.log("   - Run tests manually: ./bin/e2e -profile %s -skip-setup -use-existing-cluster", r.opts.Profile)
	r.log("   - Inspect the cluster: kubectl --context kind-%s get pods -A", r.opts.ClusterName)
	r.log("   - Clean up when done: make e2e-cleanup")
}

func (r *Runner) logPreservedEnvironmentHints() {
	if !r.opts.KeepCluster {
		return
	}

	r.log("💡 Preserved E2E environment for cluster %s", r.opts.ClusterName)
	r.log("   - Re-run tests without setup: ./bin/e2e -profile %s -skip-setup -use-existing-cluster", r.opts.Profile)
	r.log("   - Inspect the cluster: kubectl --context kind-%s get pods -A", r.opts.ClusterName)
	r.log("   - Clean up when done: make e2e-cleanup E2E_CLUSTER_NAME=%s", r.opts.ClusterName)
}

func (r *Runner) finishRun(ctx context.Context, state *runState, results []TestResult) error {
	r.reporter.AddTestResults(results)

	if err := r.reporter.CollectClusterInfo(ctx); err != nil {
		r.log("Warning: failed to collect cluster info: %v", err)
	}

	r.printResults(results)
	if !hasFailures(results) {
		return nil
	}

	r.log("❌ Some tests failed, printing all pods status for debugging...")
	r.printAllPodsDebugInfo(ctx, state.kubeClient)
	if r.opts.Verbose {
		PrintAllPodsStatus(ctx, state.kubeClient)
	}

	return fmt.Errorf("some tests failed")
}

func hasFailures(results []TestResult) bool {
	for _, result := range results {
		if !result.Passed {
			return true
		}
	}
	return false
}

// createHFTokenSecret creates a Kubernetes secret for HF_TOKEN if it's available in the environment.
func (r *Runner) createHFTokenSecret(ctx context.Context, kubeClient *kubernetes.Clientset) error {
	hfToken := os.Getenv("HF_TOKEN")
	if hfToken == "" {
		return nil
	}

	nsName := "vllm-semantic-router-system"

	_, err := kubeClient.CoreV1().Namespaces().Get(ctx, nsName, metav1.GetOptions{})
	if err != nil {
		ns := &corev1.Namespace{
			ObjectMeta: metav1.ObjectMeta{Name: nsName},
		}
		_, err = kubeClient.CoreV1().Namespaces().Create(ctx, ns, metav1.CreateOptions{})
		if err != nil && !strings.Contains(err.Error(), "already exists") {
			r.log("⚠️  Could not create namespace %s (will be created by profile): %v", nsName, err)
		}
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "hf-token-secret",
			Namespace: nsName,
		},
		Type: corev1.SecretTypeOpaque,
		StringData: map[string]string{
			"token": hfToken,
		},
	}

	_, err = kubeClient.CoreV1().Secrets(nsName).Create(ctx, secret, metav1.CreateOptions{})
	if err == nil {
		return nil
	}

	return r.handleHFTokenSecretCreateError(ctx, kubeClient, nsName, secret, err)
}

func (r *Runner) handleHFTokenSecretCreateError(
	ctx context.Context,
	kubeClient *kubernetes.Clientset,
	nsName string,
	secret *corev1.Secret,
	createErr error,
) error {
	if strings.Contains(createErr.Error(), "already exists") {
		_, err := kubeClient.CoreV1().Secrets(nsName).Update(ctx, secret, metav1.UpdateOptions{})
		if err != nil {
			return fmt.Errorf("failed to update existing HF_TOKEN secret in %s: %w", nsName, err)
		}
		return nil
	}
	if strings.Contains(createErr.Error(), "not found") {
		r.log("⚠️  Namespace %s not found yet (will be created by profile)", nsName)
		return nil
	}
	return fmt.Errorf("failed to create HF_TOKEN secret in %s: %w", nsName, createErr)
}

func createWorkspaceModelsValuesFile() (string, func(), error) {
	tmpFile, err := os.CreateTemp("", "semantic-router-workspace-models-*.yaml")
	if err != nil {
		return "", nil, fmt.Errorf("create workspace models values file: %w", err)
	}

	content := fmt.Sprintf(`persistence:
  enabled: false
extraVolumes:
  - name: workspace-models
    hostPath:
      path: %s
      type: DirectoryOrCreate
extraVolumeMounts:
  - name: workspace-models
    mountPath: /app/models
`, cluster.WorkspaceModelsNodeMountPath)

	if _, err := tmpFile.WriteString(content); err != nil {
		_ = tmpFile.Close()
		_ = os.Remove(tmpFile.Name())
		return "", nil, fmt.Errorf("write workspace models values file: %w", err)
	}
	if err := tmpFile.Close(); err != nil {
		_ = os.Remove(tmpFile.Name())
		return "", nil, fmt.Errorf("close workspace models values file: %w", err)
	}

	cleanup := func() {
		_ = os.Remove(tmpFile.Name())
	}
	return tmpFile.Name(), cleanup, nil
}
