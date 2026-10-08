package modelservice

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// fakeAutoManager is a manager whose runtime command is this test binary,
// with the given cores; auto is the device its devices command reports
// ("" fails the command). It returns the file that logs the device queries.
func fakeAutoManager(t *testing.T, auto string, cores int) (*Manager, string) {
	t.Helper()
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	queries := filepath.Join(t.TempDir(), "queries")
	t.Setenv(fakeRuntimeEnv, "1")
	t.Setenv(fakeAutoEnv, auto)
	t.Setenv(fakeDevicesLogEnv, queries)
	t.Setenv(RuntimeCommandEnv, binary)
	t.Setenv(RuntimeDirEnv, filepath.Join(t.TempDir(), "run"))
	manager := NewManager()
	manager.cores = cores
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	return manager, queries
}

func autoEncoders() map[string]config.ModelDeployment {
	return map[string]config.ModelDeployment{
		"domain": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain"},
		"guard":  {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Guard", Device: "auto"},
	}
}

func processes(t *testing.T, lease *Lease) map[string]string {
	t.Helper()
	byDeployment := map[string]string{}
	for _, status := range lease.Statuses() {
		byDeployment[status.Name] = status.Process
	}
	return byDeployment
}

func TestManagerPlansAutoDeploymentsAsCPUWhereTheRuntimeResolvesAutoToTheCPU(t *testing.T) {
	manager, queries := fakeAutoManager(t, "cpu", 8)
	cfg := runtimeConfig(autoEncoders())
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	lease := manager.Published()
	waitReady(t, lease, "domain")
	waitReady(t, lease, "guard")
	if got := processes(t, lease); got["domain"] != "cpu-0" || got["guard"] != "cpu-1" {
		t.Fatalf("each auto model gets a CPU process of its own: %v", got)
	}
	for _, name := range []string{"domain", "guard"} {
		group := lease.members[name].group
		data, err := os.ReadFile(group.modelsFile)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.Contains(string(data), `"device": "cpu"`) || group.plan.threads != 4 {
			t.Fatalf("%s runs on the CPU with a thread share of 8 cores (threads %d): %s", name, group.plan.threads, data)
		}
	}
	// A reload plans with the answer it already has.
	if err := manager.Reconcile(cfg); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(queries); strings.Count(string(data), "devices\n") != 1 {
		t.Fatalf("the runtime is asked once per manager, got %q", data)
	}
}

func TestManagerKeepsOneAutoProcessWhenTheDeviceQueryFails(t *testing.T) {
	manager, queries := fakeAutoManager(t, "", 8)
	lease, err := manager.Acquire(runtimeConfig(autoEncoders()))
	if err != nil {
		t.Fatal(err)
	}
	waitReady(t, lease, "domain")
	waitReady(t, lease, "guard")
	got := processes(t, lease)
	if got["domain"] != autoDevice || got["guard"] != autoDevice || lease.members["domain"].group.plan.threads != 0 {
		t.Fatalf("an unresolved auto keeps one process for every auto deployment: %v", got)
	}
	extra := map[string]config.ModelDeployment{"pii": {Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-PII"}}
	if _, err := manager.AcquireDeployments(extra); err != nil {
		t.Fatal(err)
	}
	if data, _ := os.ReadFile(queries); strings.Count(string(data), "devices\n") != 1 {
		t.Fatalf("a failed query is not repeated, got %q", data)
	}
}

func TestQueryAutoDeviceRefusesAnswersThatAreNotADevice(t *testing.T) {
	binary, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	t.Setenv(fakeRuntimeEnv, "1")
	for auto, ok := range map[string]bool{"rocm:0": true, "cpu": true, "auto": false, "ROCm 0": false} {
		t.Setenv(fakeAutoEnv, auto)
		device, err := queryAutoDevice([]string{binary})
		if (err == nil) != ok || (ok && device != auto) {
			t.Fatalf("%q: device %q, error %v", auto, device, err)
		}
	}
	t.Setenv(fakeAutoEnv, "")
	if _, err := queryAutoDevice([]string{binary}); err == nil || !strings.Contains(err.Error(), "no accelerator plugin could be loaded") {
		t.Fatalf("a failed query reports the runtime's last error line, got %v", err)
	}
	if _, err := queryAutoDevice(nil); err == nil {
		t.Fatal("no command, no answer")
	}
}

func TestManagerRefusesAGPUOnlyImplicitDeploymentOnTheCPU(t *testing.T) {
	nine := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-9B", Revision: strings.Repeat("b", 40), Device: autoDevice}
	cpuHost, _ := fakeAutoManager(t, "cpu", 8)
	_, err := cpuHost.AcquireDeployments(map[string]config.ModelDeployment{"@Vela-2.0-9B/auto": nine})
	if err == nil || !strings.Contains(err.Error(), `"@Vela-2.0-9B/auto": vllm-sr/Vela-2.0-9B runs on a GPU only, and the model runtime finds no GPU on this host`) {
		t.Fatalf("a host without a GPU must refuse the 9B, got %v", err)
	}
	onCPU := nine
	onCPU.Device = "cpu"
	if err := refuseGPUOnlyOnCPU(map[string]config.ModelDeployment{"@Vela-2.0-9B": onCPU}, "rocm:0"); err == nil {
		t.Fatal("a GPU-only implicit deployment on cpu is refused on any host")
	}
	if err := refuseGPUOnlyOnCPU(map[string]config.ModelDeployment{"@Vela-2.0-9B/auto": nine}, "rocm:0"); err != nil {
		t.Fatalf("a GPU host serves the 9B: %v", err)
	}
	if err := refuseGPUOnlyOnCPU(map[string]config.ModelDeployment{"vela-9b": onCPU}, "cpu"); err != nil {
		t.Fatalf("a declared deployment is the operator's choice: %v", err)
	}
	eight := config.ModelDeployment{Provider: config.ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.8B", Device: autoDevice}
	if err := refuseGPUOnlyOnCPU(map[string]config.ModelDeployment{"@Vela-2.0-0.8B/auto": eight}, "cpu"); err != nil {
		t.Fatalf("the 0.8B runs on a CPU: %v", err)
	}
}
