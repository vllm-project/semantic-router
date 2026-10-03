//go:build linux

package modelservice

import (
	"os/exec"
	"runtime"

	"golang.org/x/sys/unix"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

func allowedCPUs() []int {
	var set unix.CPUSet
	if err := unix.SchedGetaffinity(0, &set); err != nil {
		return sequentialCPUs(runtime.NumCPU())
	}
	cpus := make([]int, 0, set.Count())
	for cpu := 0; len(cpus) < set.Count(); cpu++ {
		if set.IsSet(cpu) {
			cpus = append(cpus, cpu)
		}
	}
	return cpus
}

// startPinned starts cmd on cpus. A child inherits the affinity of the thread
// that forks it, so the start runs on a locked thread restricted to cpus; the
// thread exits with its goroutine instead of returning to the scheduler.
func startPinned(cmd *exec.Cmd, cpus []int) error {
	if len(cpus) == 0 {
		return cmd.Start()
	}
	started := make(chan error, 1)
	go func() {
		runtime.LockOSThread()
		var set unix.CPUSet
		for _, cpu := range cpus {
			set.Set(cpu)
		}
		if err := unix.SchedSetaffinity(0, &set); err != nil {
			logging.ComponentWarnEvent("model_runtime", "runtime_process_unpinned", map[string]interface{}{"error": err.Error()})
		}
		started <- cmd.Start()
	}()
	return <-started
}
