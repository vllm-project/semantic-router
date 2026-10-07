package modelservice

import (
	"os"
	"runtime"
	"strconv"
)

// A runtime process runs every model's forward on one device thread, so the
// models of one CPU process run one after another. The router therefore
// spreads CPU models over processes and sizes each process's threads to an
// equal share of its cores, rounded up: a request stage's models run in
// parallel, and the cores of a process that is idle serve the busy ones.
// Pinning processes to disjoint cores measured slower, because the share of a
// rarely used model then idles.
const (
	// CPUProcessesEnv caps the processes CPU models are spread over
	// (default: one per model, at most one per minCPUThreads cores).
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
