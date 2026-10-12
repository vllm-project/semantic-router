package modelservice

import (
	"os"
	"runtime"
	"strconv"
)

// CPU workers have stable thread budgets so changing Router consumers does not
// change process identity. Every logical deployment remains an independent
// process; the thread budget is derived only from host capacity and operator
// configuration, never from the current routing mode.
const (
	// CPUThreadsEnv sets each CPU worker's threads, capped to available cores.
	CPUThreadsEnv     = "VLLM_SRUN_CPU_THREADS"
	defaultCPUThreads = 16
)

// cpuCores is the router's CPU budget: GOMAXPROCS, which Go derives from the
// affinity mask and the container's CPU quota.
func cpuCores() int { return runtime.GOMAXPROCS(0) }

// cpuThreads leaves half the cores for a peer worker, with a cap on large
// hosts. Operators can choose another budget for their inference concurrency.
func cpuThreads(cores int) int {
	if cores <= 0 {
		return 0
	}
	threads := max(1, min(defaultCPUThreads, cores/2))
	if value, err := strconv.Atoi(os.Getenv(CPUThreadsEnv)); err == nil && value > 0 {
		threads = value
	}
	return min(cores, threads)
}
