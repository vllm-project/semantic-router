package modelservice

import (
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestWorkerIdentityIndependentOfDemandAndReplicaOrder(t *testing.T) {
	resource := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-4B", Replicas: []config.ModelReplica{{Device: "rocm:0"}, {Device: "rocm:0"}, {Device: "rocm:1"}}}
	before := planProcesses(map[string]config.ModelDeployment{"primary": resource}, nil, "", 16, "")
	if len(before) != 3 {
		t.Fatal("every placement, including same-GPU duplicates, needs an independent worker")
	}
	keys := map[string]bool{}
	for _, plan := range before {
		if keys[plan.key] || len(plan.models) != 1 {
			t.Fatalf("replicas coalesced: %+v", plan)
		}
		keys[plan.key] = true
	}
	resource.Replicas[0], resource.Replicas[2] = resource.Replicas[2], resource.Replicas[0]
	after := planProcesses(map[string]config.ModelDeployment{"primary": resource, "other": {Provider: config.ModelRuntimeProvider, Artifact: resource.Artifact, Device: "rocm:0"}}, nil, "", 16, "")
	for _, plan := range after {
		if plan.logical == "primary" && !keys[plan.key] {
			t.Fatal("placement reorder or unrelated resource restarted a replica")
		}
		if plan.logical == "other" && keys[plan.key] {
			t.Fatal("independent logical deployments must not secretly coalesce")
		}
	}
}

func TestCPUWorkerBudgetStableAcrossDemandSets(t *testing.T) {
	t.Setenv(CPUProcessesEnv, "2")
	resource := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B"}
	single := planProcesses(map[string]config.ModelDeployment{"primary": resource}, nil, "", 16, "cpu")[0]
	multiple := planProcesses(map[string]config.ModelDeployment{"primary": resource, "secondary": resource}, nil, "", 16, "cpu")
	if single.models[0].Device != "cpu" || single.threads != 8 {
		t.Fatalf("CPU budget = %+v", single)
	}
	for _, plan := range multiple {
		if plan.logical == "primary" && plan.key != single.key {
			t.Fatal("Router/Engine consumer changes must retain the primary CPU process")
		}
	}
}
