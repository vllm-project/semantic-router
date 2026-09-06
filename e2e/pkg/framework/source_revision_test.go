package framework

import "testing"

const testSourceTreeDigest = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

func TestDashboardLocalImageReceivesExplicitImmutableSourceRevision(t *testing.T) {
	t.Setenv("VLLM_SR_SOURCE_REVISION", testSourceTreeDigest)
	args, err := localImageDockerBuildArgs(LocalImageBuild{Dockerfile: dashboardDockerfile})
	if err != nil {
		t.Fatalf("localImageDockerBuildArgs: %v", err)
	}
	if args["VLLM_SR_SOURCE_REVISION"] != testSourceTreeDigest {
		t.Fatalf("source revision build arg=%q", args["VLLM_SR_SOURCE_REVISION"])
	}
	if args["TARGETARCH"] == "" || args["BUILDPLATFORM"] == "" {
		t.Fatalf("platform build args missing: %#v", args)
	}
}

func TestDashboardLocalImageRejectsMutableSourceRevision(t *testing.T) {
	for _, revision := range []string{"unavailable", "main", "abc1234", "sha256:short"} {
		t.Run(revision, func(t *testing.T) {
			t.Setenv("VLLM_SR_SOURCE_REVISION", revision)
			if _, err := localImageDockerBuildArgs(LocalImageBuild{Dockerfile: dashboardDockerfile}); err == nil {
				t.Fatalf("mutable source revision %q was accepted", revision)
			}
		})
	}
}

func TestNonDashboardLocalImageDoesNotReceiveEvaluationRevision(t *testing.T) {
	t.Setenv("VLLM_SR_SOURCE_REVISION", "invalid-for-dashboard")
	args, err := localImageDockerBuildArgs(LocalImageBuild{Dockerfile: "tools/mock-vllm/Dockerfile"})
	if err != nil {
		t.Fatalf("localImageDockerBuildArgs: %v", err)
	}
	if _, exists := args["VLLM_SR_SOURCE_REVISION"]; exists {
		t.Fatalf("non-Dashboard image received evaluation revision: %#v", args)
	}
}
