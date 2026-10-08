package modelservice

import (
	"os"
	"runtime"
	"strconv"
)

// CPU workers have stable thread budgets so changing Router consumers does not
// change process identity. Every logical deployment remains an independent
// process; the concurrency budget below sizes threads rather than grouping
// models into a process. It is derived only from host capacity and operator
// configuration, never from the current routing mode.
const (
	// CPUProcessesEnv sets the CPU concurrency budget used to size each
	// worker's threads (default: one slot per minCPUThreads cores).
	CPUProcessesEnv = "VLLM_SRUN_CPU_PROCESSES"
	minCPUThreads   = 2
)

// cpuCores is the router's CPU budget: GOMAXPROCS, which Go derives from the
// affinity mask and the container's CPU quota.
func cpuCores() int { return runtime.GOMAXPROCS(0) }

func maxCPUProcesses(cores int) int {
	limit := max(1, cores/minCPUThreads)
	if value, err := strconv.Atoi(os.Getenv(CPUProcessesEnv)); err == nil && value >= 1 {
		limit = min(limit, value)
	}
	return limit
}

// cpuThreads is each of n CPU processes' share of cores, rounded up.
func cpuThreads(cores, n int) int {
	if cores <= 0 || n <= 0 {
		return 0
	}
	return (cores + n - 1) / n
}
