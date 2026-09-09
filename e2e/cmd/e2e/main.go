package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/banner"
	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
)

const version = "v1.0.0"

func main() {
	// Parse command line flags
	var (
		profile            = flag.String("profile", "envoy-ai-gateway", fmt.Sprintf("Test profile to run (%s)", strings.Join(framework.RegisteredProfileNames(), ", ")))
		clusterName        = flag.String("cluster", "", "Kind cluster name (default: unique per run)")
		imageTag           = flag.String("image-tag", "", "Docker image tag (default: unique per run)")
		outputDir          = flag.String("output-dir", os.Getenv("E2E_OUTPUT_DIR"), "Directory for this run's reports and logs")
		keepCluster        = flag.Bool("keep-cluster", false, "Keep cluster after tests complete")
		useExistingCluster = flag.Bool("use-existing-cluster", false, "Use existing cluster instead of creating a new one")
		verbose            = flag.Bool("verbose", false, "Enable verbose logging")
		parallel           = flag.Bool("parallel", false, "Run tests in parallel")
		testCases          = flag.String("tests", "", "Comma-separated list of test cases to run (empty means all)")
		setupOnly          = flag.Bool("setup-only", false, "Only setup the profile without running tests")
		skipSetup          = flag.Bool("skip-setup", false, "Skip profile setup and only run tests (assumes environment is already deployed)")
		useWorkspaceModels = flag.Bool(
			"use-workspace-models",
			boolEnv("E2E_USE_WORKSPACE_MODELS", false),
			"Mount the workspace models/ directory into semantic-router for reuse instead of the default PVC-backed path",
		)
	)

	flag.Parse()

	// Show banner
	if isInteractive() {
		banner.Show(version)
	} else {
		banner.ShowQuick(version)
	}

	// Validate flags
	if *setupOnly && *skipSetup {
		fmt.Fprintf(os.Stderr, "Error: --setup-only and --skip-setup cannot be used together\n")
		os.Exit(1)
	}

	// Setup-only mode always keeps the cluster
	if *setupOnly {
		*keepCluster = true
	}

	// Parse test cases
	var testCasesList []string
	if *testCases != "" {
		testCasesList = strings.Split(*testCases, ",")
		for i := range testCasesList {
			testCasesList[i] = strings.TrimSpace(testCasesList[i])
		}
	}

	// Create test options
	opts := &framework.TestOptions{
		Profile:            *profile,
		ClusterName:        *clusterName,
		ImageTag:           *imageTag,
		OutputDir:          *outputDir,
		KeepCluster:        *keepCluster,
		UseExistingCluster: *useExistingCluster,
		Verbose:            *verbose,
		Parallel:           *parallel,
		TestCases:          testCasesList,
		SetupOnly:          *setupOnly,
		SkipSetup:          *skipSetup,
		UseWorkspaceModels: *useWorkspaceModels,
	}
	if err := framework.ResolveRunResources(opts); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Get the profile implementation
	profileImpl, err := framework.NewProfileByName(*profile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Create and run the test runner
	runner := framework.NewRunner(opts, profileImpl)

	ctx := context.Background()
	if err := runner.Run(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

// isInteractive checks if the program is running in an interactive terminal
func isInteractive() bool {
	// Check if CI environment variable is set
	if os.Getenv("CI") != "" {
		return false
	}
	// Check if stdout is a terminal
	fileInfo, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fileInfo.Mode() & os.ModeCharDevice) != 0
}

func boolEnv(key string, fallback bool) bool {
	raw, ok := os.LookupEnv(key)
	if !ok || strings.TrimSpace(raw) == "" {
		return fallback
	}

	parsed, err := strconv.ParseBool(raw)
	if err != nil {
		return fallback
	}

	return parsed
}
