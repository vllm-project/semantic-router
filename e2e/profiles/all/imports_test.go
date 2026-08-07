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
