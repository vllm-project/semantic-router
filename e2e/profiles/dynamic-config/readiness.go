package dynamicconfig

import (
	"context"
	"os/exec"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/helpers"
)

const (
	routerNamespace         = "vllm-semantic-router-system"
	activationStartupBudget = 60 * time.Minute
)

func (p *Profile) waitForCRDReady(ctx context.Context, kubeconfig string) error {
	ctx, cancel := context.WithTimeout(ctx, activationStartupBudget)
	defer cancel()
	p.log("Waiting for the current IntelligentPool and IntelligentRoute generations to activate...")
	resources := []string{"intelligentpool/ai-gateway-pool", "intelligentroute/ai-gateway-route"}
	return helpers.WaitForActivatedResources(ctx, 5*time.Second, resources, func(ctx context.Context, resource string) ([]byte, error) {
		return exec.CommandContext(ctx, "kubectl", "get", resource, "-n", routerNamespace, "--kubeconfig", kubeconfig, "-o", "json").Output()
	})
}
