package services

import (
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// serviceModelRuntimes prepares the model runtimes of a service-owned
// generation: a lease on the configuration's model_runtime deployments from
// the process manager (none when the process has no manager) and the
// service's resource pool. The caller owns the lease.
func serviceModelRuntimes(cfg *config.RouterConfig, pool *binding.Pool) (classification.RecipeRuntimeOptions, *modelservice.Lease, error) {
	var services serving.Services
	var lease *modelservice.Lease
	if manager := modelservice.DefaultManager(); manager != nil {
		acquired, err := manager.Acquire(cfg)
		if err != nil {
			return classification.RecipeRuntimeOptions{}, nil, err
		}
		lease, services = acquired, acquired
	}
	return classification.RecipeRuntimeOptions{Runtime: serving.New(services, pool)}, lease, nil
}
