package docker

import (
	"reflect"
	"testing"
)

func TestBuildCommandArgsIncludesSortedBuildArgs(t *testing.T) {
	t.Parallel()

	opts := BuildOptions{
		Dockerfile:   "tools/docker/Dockerfile.extproc",
		Tag:          "ghcr.io/vllm-project/semantic-router/extproc:e2e-test",
		BuildContext: ".",
		BuildArgs: map[string]string{
			"TARGETARCH":    "arm64",
			"BUILDPLATFORM": "linux/arm64",
		},
	}

	got := buildCommandArgs(opts)
	want := []string{
		"build",
		"-f", "tools/docker/Dockerfile.extproc",
		"--build-arg", "BUILDPLATFORM=linux/arm64",
		"--build-arg", "TARGETARCH=arm64",
		"-t", "ghcr.io/vllm-project/semantic-router/extproc:e2e-test",
		".",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("build args mismatch\nwant: %#v\ngot:  %#v", want, got)
	}
}
