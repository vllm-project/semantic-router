package servingtest

import (
	"context"
	"os"
	"os/exec"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// TB is the subset of testing.TB, and of GinkgoT(), that Managed uses.
type TB interface {
	Helper()
	Skipf(format string, args ...any)
	Fatal(args ...any)
	Cleanup(func())
}

// Installed reports whether the managed runtime command
// (VLLM_SRUN_COMMAND, else vllm-srun) is installed.
func Installed() bool {
	_, err := exec.LookPath(runtimeCommand())
	return err == nil
}

// Managed serves real model artifacts from managed model_runtime processes,
// as the router does: each deployment starts on its first binding. The test
// is skipped when the runtime command is not installed.
func Managed(t TB) *serving.Runtime {
	t.Helper()
	if !Installed() {
		t.Skipf("model runtime command %q is not installed", runtimeCommand())
	}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = lease.Close() })
	return serving.New(lease, nil)
}

func runtimeCommand() string {
	if command := strings.Fields(os.Getenv(modelservice.RuntimeCommandEnv)); len(command) > 0 {
		return command[0]
	}
	return "vllm-srun"
}
