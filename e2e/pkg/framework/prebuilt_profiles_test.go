package framework_test

import (
	"context"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/e2e/pkg/framework"
	_ "github.com/vllm-project/semantic-router/e2e/profiles/all"
)

var fixtureEnvironments = map[string]string{
	"mock-vllm":      "E2E_PREBUILT_MOCK_VLLM_IMAGE",
	"dashboard":      "VLLM_SR_DASHBOARD_IMAGE",
	"llm-katan":      "E2E_PREBUILT_LLM_KATAN_IMAGE",
	"anthropic-shim": "E2E_PREBUILT_ANTHROPIC_SHIM_IMAGE",
}

func sameLocalImageFixture(a, b framework.LocalImageBuild) bool {
	return a.Dockerfile == b.Dockerfile && a.Tag == b.Tag && a.BuildContext == b.BuildContext
}

func registeredFixtureImages(t *testing.T) map[string]framework.LocalImageBuild {
	t.Helper()
	images := map[string]framework.LocalImageBuild{}
	for _, name := range framework.RegisteredProfileNames() {
		registration, ok := framework.LookupProfileRegistration(name)
		if !ok {
			t.Fatalf("profile %s disappeared", name)
		}
		for _, image := range registration.Capabilities.LocalImages {
			id := strings.Split(filepath.Base(image.Tag), ":")[0]
			if _, known := fixtureEnvironments[id]; !known {
				t.Fatalf("profile %s requires unqualified fixture %s", name, id)
			}
			if previous, exists := images[id]; exists && !sameLocalImageFixture(previous, image) {
				t.Fatalf("profiles disagree on fixture %s: %+v and %+v", id, previous, image)
			}
			images[id] = image
		}
	}
	// The loader also supports this maintained service outside registered profiles.
	if _, registered := images["llm-katan"]; !registered {
		images["llm-katan"] = framework.LocalImageBuild{
			Dockerfile: "e2e/testing/llm-katan/Dockerfile", Tag: "fixture/llm-katan:test",
			BuildContext: "e2e/testing/llm-katan",
		}
	}
	if len(images) != len(fixtureEnvironments) {
		t.Fatalf("incomplete fixture inventory: %+v", images)
	}
	for id, image := range images {
		path := filepath.Join("..", "..", "..", image.Dockerfile)
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			t.Fatalf("fixture %s references missing Dockerfile %s: %v", id, image.Dockerfile, err)
		}
	}
	return images
}

func prebuiltCommands(t *testing.T) func() []string {
	t.Helper()
	directory := t.TempDir()
	log := filepath.Join(directory, "commands")
	t.Setenv("PATH", directory+string(os.PathListSeparator)+os.Getenv("PATH"))
	t.Setenv("COMMAND_LOG", log)
	t.Setenv("UNAVAILABLE_IMAGE", "")
	t.Setenv("PREBUILT_RUNTIME_IMAGES", "1")
	t.Setenv("E2E_PREBUILT_EXT_PROC_IMAGE", "verified:extproc")
	for id, name := range fixtureEnvironments {
		t.Setenv(name, "verified:"+id)
	}
	script := `#!/bin/sh
printf '%s %s\n' "${0##*/}" "$*" >> "$COMMAND_LOG"
if [ "$1 $2" = 'image inspect' ] && [ "$3" = "$UNAVAILABLE_IMAGE" ]; then exit 1; fi
`
	for _, name := range []string{"docker", "kind"} {
		if err := os.WriteFile(filepath.Join(directory, name), []byte(script), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	return func() []string {
		raw, err := os.ReadFile(log)
		if os.IsNotExist(err) {
			return nil
		}
		if err != nil {
			t.Fatal(err)
		}
		return strings.Split(strings.TrimSpace(string(raw)), "\n")
	}
}

func TestPrebuiltImagesConsumeRegisteredProfileFixturesWithoutBuild(t *testing.T) {
	for id, image := range registeredFixtureImages(t) {
		t.Run(id, func(t *testing.T) {
			commands := prebuiltCommands(t)
			if err := framework.BuildPrebuiltFixturesForTest(context.Background(), []framework.LocalImageBuild{image}); err != nil {
				t.Fatal(err)
			}
			want := []string{
				"docker image inspect verified:extproc",
				"docker tag verified:extproc ghcr.io/vllm-project/semantic-router/extproc:test",
				"kind load docker-image ghcr.io/vllm-project/semantic-router/extproc:test --name fixture-test",
				"docker image inspect verified:" + id,
				"docker tag verified:" + id + " " + image.Tag,
				"kind load docker-image " + image.Tag + " --name fixture-test",
			}
			if got := commands(); !slices.Equal(got, want) {
				t.Fatalf("unexpected image operations: got %q, want %q", got, want)
			}
		})
	}
}

func TestPrebuiltImagesRefuseMissingInputsWithoutBuildFallback(t *testing.T) {
	images := registeredFixtureImages(t)
	for id, image := range images {
		t.Run("missing-"+id, func(t *testing.T) {
			commands := prebuiltCommands(t)
			t.Setenv(fixtureEnvironments[id], "")
			err := framework.BuildPrebuiltFixturesForTest(context.Background(), []framework.LocalImageBuild{image})
			if err == nil || !strings.Contains(err.Error(), "required prebuilt fixture missing") || len(commands()) != 3 {
				t.Fatalf("missing fixture was not rejected before load/build: %v, %q", err, commands())
			}
		})
	}
	for _, scenario := range []string{"unknown-dockerfile", "missing-extproc", "missing-artifact"} {
		t.Run(scenario, func(t *testing.T) {
			commands := prebuiltCommands(t)
			image := images["mock-vllm"]
			wantCommands := 3
			switch scenario {
			case "unknown-dockerfile":
				image.Dockerfile = "unknown/Dockerfile"
			case "missing-extproc":
				t.Setenv("E2E_PREBUILT_EXT_PROC_IMAGE", "")
				wantCommands = 0
			case "missing-artifact":
				t.Setenv("UNAVAILABLE_IMAGE", "verified:mock-vllm")
				wantCommands = 4
			}
			err := framework.BuildPrebuiltFixturesForTest(context.Background(), []framework.LocalImageBuild{image})
			if err == nil || !strings.Contains(err.Error(), "prebuilt") || len(commands()) != wantCommands {
				t.Fatalf("invalid input was not rejected: %v, %q", err, commands())
			}
			for _, command := range commands() {
				if strings.HasPrefix(command, "docker build") {
					t.Fatalf("missing prebuilt input fell back to build: %q", command)
				}
			}
		})
	}
}
