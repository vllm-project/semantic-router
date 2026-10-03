package modelservice

import (
	"os"
	"runtime"
	"strconv"
)

// A runtime process runs every model's forward on one device thread, so the
// models of one CPU process run one after another. The router therefore
// spreads CPU models over processes and gives each process a disjoint share
// of its cores, pinned, with a matching thread count: the models of a request
// stage run in parallel and never oversubscribe the cores.
const (
	// CPUProcessesEnv caps the processes CPU models are spread over
	// (default: one per model, at most one per minCPUThreads cores).
	CPUProcessesEnv = "VLLM_SR_RUNTIME_CPU_PROCESSES"
	minCPUThreads   = 2
)

// cpuBudget is the CPUs runtime processes may use: the router's affinity,
// limited to GOMAXPROCS, which Go derives from the container's CPU quota.
func cpuBudget() []int {
	cpus := allowedCPUs()
	if limit := runtime.GOMAXPROCS(0); limit > 0 && limit < len(cpus) {
		cpus = cpus[:limit]
	}
	return cpus
}

func maxCPUProcesses(cores int) int {
	limit := max(1, cores/minCPUThreads)
	if value, err := strconv.Atoi(os.Getenv(CPUProcessesEnv)); err == nil && value >= 1 {
		limit = min(limit, value)
	}
	return limit
}

// shareCPUs splits cpus into n contiguous shares whose sizes differ by at
// most one, or returns nil when there are fewer cpus than shares.
func shareCPUs(cpus []int, n int) [][]int {
	if n <= 0 || len(cpus) < n {
		return nil
	}
	shares := make([][]int, n)
	start := 0
	for i := range shares {
		size := len(cpus) / n
		if i < len(cpus)%n {
			size++
		}
		shares[i] = cpus[start : start+size : start+size]
		start += size
	}
	return shares
}

func sequentialCPUs(n int) []int {
	cpus := make([]int, n)
	for i := range cpus {
		cpus[i] = i
	}
	return cpus
}
