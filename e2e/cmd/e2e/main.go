package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/vllm-project/semantic-router/e2e/pkg/banner"
	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	aigateway "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"
	aibrix "github.com/vllm-project/semantic-router/e2e/profiles/aibrix"
	dynamicconfig "github.com/vllm-project/semantic-router/e2e/profiles/dynamic-config"
	dynamo "github.com/vllm-project/semantic-router/e2e/profiles/dynamo"
	istio "github.com/vllm-project/semantic-router/e2e/profiles/istio"
	llmd "github.com/vllm-project/semantic-router/e2e/profiles/llm-d"
	productionstack "github.com/vllm-project/semantic-router/e2e/profiles/production-stack"
	responseapi "github.com/vllm-project/semantic-router/e2e/profiles/response-api"
	responseapiredis "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis"
	responseapirediscluster "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis-cluster"
	routingstrategies "github.com/vllm-project/semantic-router/e2e/profiles/routing-strategies"

	// Import profiles to register test cases
	_ "github.com/vllm-project/semantic-router/e2e/profiles/ai-gateway"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/aibrix"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/dynamo"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/istio"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/llm-d"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/production-stack"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/response-api"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/response-api-redis-cluster"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/routing-strategies"
)

const version = "v1.0.0"

func main() {
	// Parse command line flags
	var (
		profile            = flag.String("profile", "ai-gateway", "Test profile to run (ai-gateway, dynamo, istio, etc.)")
		clusterName        = flag.String("cluster", "semantic-router-e2e", "Kind cluster name")
		imageTag           = flag.String("image-tag", "e2e-test", "Docker image tag")
		keepCluster        = flag.Bool("keep-cluster", false, "Keep cluster after tests complete")
		useExistingCluster = flag.Bool("use-existing-cluster", false, "Use existing cluster instead of creating a new one")
		verbose            = flag.Bool("verbose", false, "Enable verbose logging")
		parallel           = flag.Bool("parallel", false, "Run tests in parallel")
		testCases          = flag.String("tests", "", "Comma-separated list of test cases to run (empty means all)")
		setupOnly          = flag.Bool("setup-only", false, "Only setup the profile without running tests")
		skipSetup          = flag.Bool("skip-setup", false, "Skip profile setup and only run tests (assumes environment is already deployed)")
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
		KeepCluster:        *keepCluster,
		UseExistingCluster: *useExistingCluster,
		Verbose:            *verbose,
		Parallel:           *parallel,
		TestCases:          testCasesList,
		SetupOnly:          *setupOnly,
		SkipSetup:          *skipSetup,
	}

	// Get the profile implementation
	profileImpl, err := getProfile(*profile)
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

func getProfile(name string) (framework.Profile, error) {
	switch name {
	case "ai-gateway":
		return aigateway.NewProfile(), nil
	case "dynamic-config":
		return dynamicconfig.NewProfile(), nil
	case "dynamo":
		return dynamo.NewProfile(), nil
	case "aibrix":
		return aibrix.NewProfile(), nil
	case "istio":
		return istio.NewProfile(), nil
	case "llm-d":
		return llmd.NewProfile(), nil
	case "production-stack":
		return productionstack.NewProfile(), nil
	case "response-api":
		return responseapi.NewProfile(), nil
	case "response-api-redis":
		return responseapiredis.NewProfile(), nil
	case "response-api-redis-cluster":
		return responseapirediscluster.NewProfile(), nil
	case "routing-strategies":
		return routingstrategies.NewProfile(), nil
	default:
		return nil, fmt.Errorf("unknown profile: %s", name)
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
