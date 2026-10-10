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
	for _, test := range []struct {
		name, override string
		threads        int
	}{{name: "default", threads: 16}, {name: "operator", override: "8", threads: 8}} {
		t.Run(test.name, func(t *testing.T) {
			t.Setenv(CPUThreadsEnv, test.override)
			resource := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B"}
			single := planProcesses(map[string]config.ModelDeployment{"primary": resource}, nil, "", 32, "cpu")[0]
			multiple := planProcesses(map[string]config.ModelDeployment{"primary": resource, "secondary": resource}, nil, "", 32, "cpu")
			if single.models[0].Device != "cpu" || single.threads != test.threads {
				t.Fatalf("CPU budget = %+v", single)
			}
			for _, plan := range multiple {
				if plan.logical == "primary" && plan.key != single.key {
					t.Fatal("Router/Engine consumer changes must retain the primary CPU process")
				}
			}
		})
	}
}

func TestCPUThreadOverrideChangesOnlyOwnedCPUWorkerIdentity(t *testing.T) {
	deployments := map[string]config.ModelDeployment{
		"cpu":      {Artifact: "model", Device: "cpu"},
		"gpu":      {Artifact: "model", Device: "rocm:0"},
		"attached": {Endpoint: "http://runtime.example", Device: "cpu"},
	}
	t.Setenv(CPUThreadsEnv, "4")
	before := planProcesses(deployments, nil, "", 32, "cpu")
	t.Setenv(CPUThreadsEnv, "8")
	after := planProcesses(deployments, nil, "", 32, "cpu")
	for _, old := range before {
		for _, next := range after {
			if old.logical != next.logical {
				continue
			}
			if old.logical == "cpu" {
				if old.threads != 4 || next.threads != 8 || old.key == next.key {
					t.Fatal("changing a CPU worker's thread budget must prepare a new process")
				}
			} else if old.threads != 0 || next.threads != 0 || old.key != next.key {
				t.Fatalf("CPU thread budget changed %s process", old.logical)
			}
		}
	}
}
