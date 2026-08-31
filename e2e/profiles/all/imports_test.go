package all

import (
	"reflect"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
)

func TestDashboardProfileBuildsLocalImage(t *testing.T) {
	registration, ok := framework.LookupProfileRegistration("dashboard")
	if !ok {
		t.Fatal("dashboard profile is not registered")
	}

	want := []framework.LocalImageBuild{{
		Dockerfile:   "dashboard/backend/Dockerfile",
		Tag:          "ghcr.io/vllm-project/semantic-router/dashboard:e2e-test",
		BuildContext: ".",
	}}
	if !reflect.DeepEqual(registration.Capabilities.LocalImages, want) {
		t.Fatalf("dashboard local images = %#v, want %#v", registration.Capabilities.LocalImages, want)
	}
}

func TestEnvoyAIGatewayProfileKeepsKubernetesAlias(t *testing.T) {
	if _, ok := framework.LookupProfileRegistration("envoy-ai-gateway"); !ok {
		t.Fatal("envoy-ai-gateway profile is not registered")
	}
	profile, err := framework.NewProfileByName("kubernetes")
	if err != nil {
		t.Fatalf("legacy kubernetes profile alias failed: %v", err)
	}
	if profile.Name() != "envoy-ai-gateway" {
		t.Fatalf("legacy alias resolved to %q, want envoy-ai-gateway", profile.Name())
	}
}

func TestProtocolCodecE2EMatrixProfilesAreClosed(t *testing.T) {
	profiles := map[string][]string{
		"response-api": {
			"protocol-codec-chat-backend-buffered-matrix",
			"protocol-codec-chat-backend-streaming-matrix",
			"protocol-codec-responses-backend-buffered-matrix",
			"protocol-codec-responses-backend-streaming-matrix",
		},
		"anthropic-shim": {
			"protocol-codec-anthropic-backend-buffered-matrix",
			"protocol-codec-anthropic-backend-streaming-matrix",
		},
	}

	for profileName, required := range profiles {
		profileName, required := profileName, required
		t.Run(profileName, func(t *testing.T) {
			profile, err := framework.NewProfileByName(profileName)
			if err != nil {
				t.Fatal(err)
			}
			actual := make(map[string]struct{}, len(profile.GetTestCases()))
			for _, name := range profile.GetTestCases() {
				actual[name] = struct{}{}
			}
			for _, name := range required {
				if _, ok := actual[name]; !ok {
					t.Errorf("profile %q is missing protocol matrix testcase %q", profileName, name)
				}
			}
		})
	}
}
