package classification

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving/servingtest"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// testModelRuntime prepares a recipe's model runtime whose deployments, named
// as the bindings name them (implicit module defaults are "@<module>"), are
// served by a fake runtime process.
func testModelRuntime(t *testing.T, cfg *config.RouterConfig, deployments map[string]runtimetest.Model) *classifierModelRuntime {
	t.Helper()
	runtime, _ := servingtest.Runtime(t, deployments)
	models, err := newClassifierModelRuntime(cfg, RecipeRuntimeOptions{Runtime: runtime})
	if err != nil {
		t.Fatal(err)
	}
	return models
}
