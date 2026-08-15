package anthropicshim

import "github.com/vllm-project/semantic-router/e2e/pkg/framework"

// LocalImages returns the fixture images required by this profile.
func LocalImages() []framework.LocalImageBuild {
	return []framework.LocalImageBuild{
		{
			Dockerfile:   "e2e/testing/anthropic-shim/Dockerfile",
			Tag:          "ghcr.io/vllm-project/semantic-router/anthropic-shim:e2e-test",
			BuildContext: "e2e/testing/anthropic-shim",
		},
	}
}
