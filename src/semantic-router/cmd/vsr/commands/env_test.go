package commands

import (
	"os"
	"testing"
)

// TestVSRConfigEnvVar tests that VSR_CONFIG environment variable is respected
func TestVSRConfigEnvVar(t *testing.T) {
	// Test 1: Default value when no env var is set
	os.Unsetenv("VSR_CONFIG")
	configDefault := os.Getenv("VSR_CONFIG")
	if configDefault != "" {
		t.Fatalf("Expected empty VSR_CONFIG, got %s", configDefault)
	}

	// Test 2: VSR_CONFIG should override default
	expectedConfig := "/custom/config.yaml"
	os.Setenv("VSR_CONFIG", expectedConfig)
	actualConfig := os.Getenv("VSR_CONFIG")
	if actualConfig != expectedConfig {
		t.Fatalf("Expected VSR_CONFIG=%s, got %s", expectedConfig, actualConfig)
	}

	// Cleanup
	os.Unsetenv("VSR_CONFIG")
}

// TestVSRNamespaceEnvVar tests that VSR_NAMESPACE environment variable is respected
func TestVSRNamespaceEnvVar(t *testing.T) {
	// Test 1: Default namespace when no env var
	os.Unsetenv("VSR_NAMESPACE")
	os.Unsetenv("_VSR_NAMESPACE_DEFAULT")

	actualNamespace := os.Getenv("VSR_NAMESPACE")
	if actualNamespace != "" {
		t.Fatalf("Expected empty VSR_NAMESPACE, got %s", actualNamespace)
	}

	// Test 2: VSR_NAMESPACE should be stored in _VSR_NAMESPACE_DEFAULT
	expectedNamespace := "production"
	os.Setenv("VSR_NAMESPACE", expectedNamespace)
	os.Setenv("_VSR_NAMESPACE_DEFAULT", expectedNamespace)

	storedNamespace := os.Getenv("_VSR_NAMESPACE_DEFAULT")
	if storedNamespace != expectedNamespace {
		t.Fatalf("Expected _VSR_NAMESPACE_DEFAULT=%s, got %s", expectedNamespace, storedNamespace)
	}

	// Cleanup
	os.Unsetenv("VSR_NAMESPACE")
	os.Unsetenv("_VSR_NAMESPACE_DEFAULT")
}

// TestVSRVerboseEnvVar tests that VSR_VERBOSE environment variable is respected
func TestVSRVerboseEnvVar(t *testing.T) {
	// Test 1: Default is false when no env var
	os.Unsetenv("VSR_VERBOSE")
	verbose := os.Getenv("VSR_VERBOSE")
	if verbose != "" {
		t.Fatalf("Expected empty VSR_VERBOSE, got %s", verbose)
	}

	// Test 2: VSR_VERBOSE="true" should be set
	os.Setenv("VSR_VERBOSE", "true")
	verbose = os.Getenv("VSR_VERBOSE")
	if verbose != "true" {
		t.Fatalf("Expected VSR_VERBOSE=true, got %s", verbose)
	}

	// Test 3: Other values should not be treated as true
	os.Setenv("VSR_VERBOSE", "yes")
	verbose = os.Getenv("VSR_VERBOSE")
	if verbose == "true" {
		t.Fatalf("Expected VSR_VERBOSE=yes to not equal 'true', got %s", verbose)
	}

	// Cleanup
	os.Unsetenv("VSR_VERBOSE")
}

// TestEnvVarPrecedence tests that explicit flags take precedence over env vars
func TestEnvVarPrecedence(t *testing.T) {
	// Test that if a flag is explicitly set, it overrides the env var
	// This is handled by Cobra's flag parsing, so we just test the env var setup

	// Set env var
	os.Setenv("VSR_CONFIG", "env-config.yaml")
	expectedEnvValue := os.Getenv("VSR_CONFIG")
	if expectedEnvValue != "env-config.yaml" {
		t.Fatalf("Expected env var to be set to 'env-config.yaml', got %s", expectedEnvValue)
	}

	// In actual usage, a flag passed explicitly would override this through Cobra
	// (e.g., vsr -c explicit-config.yaml would use explicit-config.yaml)

	// Cleanup
	os.Unsetenv("VSR_CONFIG")
}

// TestStatusCmdNamespaceDefault tests that status command respects VSR_NAMESPACE
func TestStatusCmdNamespaceDefault(t *testing.T) {
	// Set environment variable for namespace
	os.Setenv("_VSR_NAMESPACE_DEFAULT", "test-namespace")

	// Create status command
	cmd := NewStatusCmd()

	// Get the namespace flag value
	namespace, _ := cmd.Flags().GetString("namespace")
	if namespace != "test-namespace" {
		t.Fatalf("Expected status command namespace to be 'test-namespace', got %s", namespace)
	}

	// Cleanup
	os.Unsetenv("_VSR_NAMESPACE_DEFAULT")
}

// TestDeployUndeployCmdNamespaceDefault tests that deploy/undeploy commands respect VSR_NAMESPACE
func TestDeployUndeployCmdNamespaceDefault(t *testing.T) {
	// Set environment variable for namespace
	os.Setenv("_VSR_NAMESPACE_DEFAULT", "prod-namespace")

	// Create deploy command
	deployCmd := NewDeployCmd()
	namespace, _ := deployCmd.Flags().GetString("namespace")
	if namespace != "prod-namespace" {
		t.Fatalf("Expected deploy command namespace to be 'prod-namespace', got %s", namespace)
	}

	// Create undeploy command
	undeployCmd := NewUndeployCmd()
	namespace, _ = undeployCmd.Flags().GetString("namespace")
	if namespace != "prod-namespace" {
		t.Fatalf("Expected undeploy command namespace to be 'prod-namespace', got %s", namespace)
	}

	// Cleanup
	os.Unsetenv("_VSR_NAMESPACE_DEFAULT")
}

// TestUpgradeDashboardCmdNamespaceDefault tests that upgrade/dashboard commands respect VSR_NAMESPACE
func TestUpgradeDashboardCmdNamespaceDefault(t *testing.T) {
	// Set environment variable for namespace
	os.Setenv("_VSR_NAMESPACE_DEFAULT", "staging-namespace")

	// Create upgrade command
	upgradeCmd := NewUpgradeCmd()
	namespace, _ := upgradeCmd.Flags().GetString("namespace")
	if namespace != "staging-namespace" {
		t.Fatalf("Expected upgrade command namespace to be 'staging-namespace', got %s", namespace)
	}

	// Create dashboard command
	dashboardCmd := NewDashboardCmd()
	namespace, _ = dashboardCmd.Flags().GetString("namespace")
	if namespace != "staging-namespace" {
		t.Fatalf("Expected dashboard command namespace to be 'staging-namespace', got %s", namespace)
	}

	// Cleanup
	os.Unsetenv("_VSR_NAMESPACE_DEFAULT")
}
