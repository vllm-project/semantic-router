package all

import (
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

// TestStickyToolSelectionContractIsRegistered keeps the maintained recovery
// case attached to the profile that owns the sticky runtime configuration.
// Without this guard, a testcase can remain compilable while silently
// dropping out of the profile's live E2E matrix.
func TestStickyToolSelectionContractIsRegistered(t *testing.T) {
	profile, err := framework.NewProfileByName("envoy-ai-gateway")
	if err != nil {
		t.Fatalf("envoy-ai-gateway profile failed: %v", err)
	}

	for _, required := range []string{
		"sticky-tool-selection",
		"sticky-tool-selection-recovery",
	} {
		if !profileHasTestCase(profile.GetTestCases(), required) {
			t.Fatalf("envoy-ai-gateway profile is missing sticky testcase %q: %v", required, profile.GetTestCases())
		}
	}

	recovery, ok := pkgtestcases.Get("sticky-tool-selection-recovery")
	if !ok {
		t.Fatal("sticky-tool-selection-recovery is not registered")
	}
	for _, requiredTag := range []string{"sticky", "restart", "recovery"} {
		if !containsString(recovery.Tags, requiredTag) {
			t.Fatalf("sticky recovery testcase tags = %v, want %q", recovery.Tags, requiredTag)
		}
	}
}

// TestStickyProviderPrefixContractIsRegistered keeps the Anthropic-specific
// prefix case attached to the profile that provides its backend fixture.
func TestStickyProviderPrefixContractIsRegistered(t *testing.T) {
	profile, err := framework.NewProfileByName("anthropic-shim")
	if err != nil {
		t.Fatalf("anthropic-shim profile failed: %v", err)
	}

	const required = "sticky-tool-selection-provider-prefix"
	if !profileHasTestCase(profile.GetTestCases(), required) {
		t.Fatalf("anthropic-shim profile is missing sticky testcase %q: %v", required, profile.GetTestCases())
	}

	testCase, ok := pkgtestcases.Get(required)
	if !ok {
		t.Fatalf("%s is not registered", required)
	}
	for _, requiredTag := range []string{"anthropic", "cache", "sticky"} {
		if !containsString(testCase.Tags, requiredTag) {
			t.Fatalf("sticky provider-prefix testcase tags = %v, want %q", testCase.Tags, requiredTag)
		}
	}
}

func profileHasTestCase(testCases []string, want string) bool {
	for _, name := range testCases {
		if name == want {
			return true
		}
	}
	return false
}

func containsString(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
