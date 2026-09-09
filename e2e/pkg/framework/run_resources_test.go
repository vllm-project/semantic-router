package framework

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunResourcesSeparateInvocationsAndRequireExplicitReuse(t *testing.T) {
	a := &TestOptions{OutputDir: t.TempDir()}
	b := &TestOptions{OutputDir: t.TempDir()}
	for _, opts := range []*TestOptions{a, b} {
		if err := ResolveRunResources(opts); err != nil {
			t.Fatal(err)
		}
	}
	if a.ClusterName == b.ClusterName || a.ImageTag == b.ImageTag {
		t.Fatal("independent invocations shared cluster or image identity")
	}
	if err := ResolveRunResources(&TestOptions{UseExistingCluster: true}); err == nil {
		t.Fatal("reuse without a named cluster was accepted")
	}
	if err := ResolveRunResources(&TestOptions{SkipSetup: true}); err == nil {
		t.Fatal("skip setup without explicit reuse was accepted")
	}
}

func TestManifestImageStagingIsIsolatedAndCleanedAfterFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "deployment.yaml")
	source := "image: registry:5000/dashboard:e2e-test\n"
	if err := os.WriteFile(path, []byte(source), 0o600); err != nil {
		t.Fatal(err)
	}
	wantError := errors.New("apply failed")
	var staged string
	err := WithLocalImages(path, map[string]string{"registry:5000/dashboard:e2e-test": "registry:5000/dashboard:run-a"}, func(rewritten string) error {
		staged = rewritten
		data, err := os.ReadFile(rewritten)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), "dashboard:run-a") {
			t.Fatalf("wrong image: %s", data)
		}
		return wantError
	})
	if !errors.Is(err, wantError) {
		t.Fatalf("lost apply failure: %v", err)
	}
	if _, statErr := os.Stat(staged); !os.IsNotExist(statErr) {
		t.Fatalf("staged manifest leaked: %v", statErr)
	}
	unchanged, err := os.ReadFile(path)
	if err != nil || string(unchanged) != source {
		t.Fatal("modified source manifest")
	}
}

func TestAllLocalImagesReceiveRunTag(t *testing.T) {
	r := &Runner{opts: &TestOptions{ImageTag: "run-b"}, profileCapabilities: ProfileCapabilities{LocalImages: []LocalImageBuild{
		{Tag: "registry:5000/mock:latest"}, {Tag: "registry/dashboard:e2e-test"}, {Tag: "registry/anthropic-shim:e2e-test"},
	}}}
	for original, image := range r.localImageReferences() {
		if !strings.HasSuffix(image, ":run-b") {
			t.Fatalf("%s was not isolated: %s", original, image)
		}
	}
	if got := runImageReference("registry:5000/mock:latest", "run-a"); got != "registry:5000/mock:run-a" {
		t.Fatal(got)
	}
}
