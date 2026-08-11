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
