package dynamicconfig

import (
	"context"
	"encoding/json"
	"fmt"
	"os/exec"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	routerNamespace         = "vllm-semantic-router-system"
	activationStartupBudget = 60 * time.Minute
)

func (p *Profile) waitForCRDReady(ctx context.Context, kubeconfig string) error {
	ctx, cancel := context.WithTimeout(ctx, activationStartupBudget)
	defer cancel()
	p.log("Waiting for the current IntelligentPool and IntelligentRoute generations to activate...")
	return waitForActivatedCRs(ctx, 5*time.Second, func(ctx context.Context, resource string) ([]byte, error) {
		return exec.CommandContext(ctx, "kubectl", "get", resource, "-n", routerNamespace, "--kubeconfig", kubeconfig, "-o", "json").Output()
	})
}

func waitForActivatedCRs(ctx context.Context, interval time.Duration, fetch func(context.Context, string) ([]byte, error)) error {
	ticker := time.NewTicker(interval)
	defer ticker.Stop()
	for {
		ready := true
		for _, resource := range []string{"intelligentpool/ai-gateway-pool", "intelligentroute/ai-gateway-route"} {
			data, err := fetch(ctx, resource)
			if err != nil || !currentGenerationReady(data) {
				ready = false
			}
		}
		if ready {
			return nil
		}
		select {
		case <-ctx.Done():
			return fmt.Errorf("waiting for CR runtime activation: %w", ctx.Err())
		case <-ticker.C:
		}
	}
}

func currentGenerationReady(data []byte) bool {
	var resource struct {
		Metadata struct {
			Generation int64 `json:"generation"`
		} `json:"metadata"`
		Status struct {
			Conditions []metav1.Condition `json:"conditions"`
		} `json:"status"`
	}
	if json.Unmarshal(data, &resource) != nil || resource.Metadata.Generation < 1 {
		return false
	}
	for _, condition := range resource.Status.Conditions {
		if condition.Type == "Ready" {
			return condition.Status == metav1.ConditionTrue && condition.ObservedGeneration == resource.Metadata.Generation
		}
	}
	return false
}
