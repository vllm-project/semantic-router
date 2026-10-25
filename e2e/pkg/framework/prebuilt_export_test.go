package framework

import (
	"context"

	"github.com/vllm-project/semantic-router/e2e/pkg/docker"
)

// BuildPrebuiltFixturesForTest exposes the runner only to external profile tests.
func BuildPrebuiltFixturesForTest(ctx context.Context, images []LocalImageBuild) error {
	runner := &Runner{
		opts:                &TestOptions{ClusterName: "fixture-test", ImageTag: "test"},
		profileCapabilities: ProfileCapabilities{LocalImages: images},
		builder:             docker.NewBuilder(false),
	}
	return runner.buildAndLoadImages(ctx)
}
