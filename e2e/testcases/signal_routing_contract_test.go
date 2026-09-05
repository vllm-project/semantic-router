package testcases_test

import (
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"

	// Registers every profile (and, transitively, every testcase in this
	// package) so framework.NewProfileByName resolves the same way it does
	// inside the real e2e binary.
	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
)

// signalRoutingContracts locks in each signal-routing E2E test stays selected
// by its intended profile, so a dropped entry fails `go test` instead of only
// surfacing during a full E2E run.
var signalRoutingContracts = []struct {
	profile  string
	testCase string
}{
	{profile: "envoy-ai-gateway", testCase: "event-routing"},
	{profile: "envoy-ai-gateway", testCase: "language-routing"},
}

func TestProfilesSelectSignalRoutingContracts(t *testing.T) {
	for _, tc := range signalRoutingContracts {
		t.Run(tc.profile+"/"+tc.testCase, func(t *testing.T) {
			profile, err := framework.NewProfileByName(tc.profile)
			if err != nil {
				t.Fatalf("%s profile failed: %v", tc.profile, err)
			}

			for _, name := range profile.GetTestCases() {
				if name == tc.testCase {
					return
				}
			}
			t.Fatalf("%s profile test cases = %v, want %q included", tc.profile, profile.GetTestCases(), tc.testCase)
		})
	}
}
