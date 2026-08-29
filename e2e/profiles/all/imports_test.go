package all

import (
	"bufio"
	"bytes"
	"encoding/json"
	"io"
	"os"
	"reflect"
	"testing"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	anthropicshim "github.com/vllm-project/semantic-router/e2e/profiles/anthropic-shim"
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

func TestAnthropicShimProfileBuildsLocalImage(t *testing.T) {
	registration, ok := framework.LookupProfileRegistration("anthropic-shim")
	if !ok {
		t.Fatal("anthropic-shim profile is not registered")
	}

	localImages := anthropicshim.LocalImages()
	if len(localImages) != 1 {
		t.Fatalf("LocalImages() returned %d entries, want 1", len(localImages))
	}
	if len(registration.Capabilities.LocalImages) != 1 || registration.Capabilities.LocalImages[0].Tag != localImages[0].Tag {
		t.Fatalf("registered local images = %#v, want Tag %q", registration.Capabilities.LocalImages, localImages[0].Tag)
	}

	image, pullPolicy := anthropicShimContainerFromBackendYAML(t)
	if image != localImages[0].Tag {
		t.Fatalf("backend.yaml anthropic-shim image = %q, want LocalImages Tag %q", image, localImages[0].Tag)
	}
	if pullPolicy != corev1.PullNever {
		t.Fatalf("backend.yaml anthropic-shim imagePullPolicy = %q, want %q", pullPolicy, corev1.PullNever)
	}
}

func anthropicShimContainerFromBackendYAML(t *testing.T) (string, corev1.PullPolicy) {
	t.Helper()

	raw, err := os.ReadFile("../anthropic-shim/gateway-resources/backend.yaml")
	if err != nil {
		t.Fatalf("read backend.yaml: %v", err)
	}

	reader := utilyaml.NewYAMLReader(bufio.NewReader(bytes.NewReader(raw)))
	for {
		doc, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("read backend.yaml document: %v", err)
		}
		if len(bytes.TrimSpace(doc)) == 0 {
			continue
		}

		jsonDocument, err := utilyaml.ToJSON(doc)
		if err != nil {
			t.Fatalf("convert backend.yaml document to JSON: %v", err)
		}

		var typeMeta struct {
			Kind string `json:"kind"`
		}
		if err := json.Unmarshal(jsonDocument, &typeMeta); err != nil {
			t.Fatalf("decode backend.yaml type meta: %v", err)
		}
		if typeMeta.Kind != "Deployment" {
			continue
		}

		var deployment appsv1.Deployment
		if err := json.Unmarshal(jsonDocument, &deployment); err != nil {
			t.Fatalf("decode backend.yaml deployment: %v", err)
		}
		for _, container := range deployment.Spec.Template.Spec.Containers {
			if container.Name == "anthropic-shim" {
				return container.Image, container.ImagePullPolicy
			}
		}
		t.Fatal("anthropic-shim container not found in backend.yaml Deployment")
	}

	t.Fatal("Deployment not found in backend.yaml")
	return "", ""
}
