package classification

import "runtime"

// maxEmbeddingConcurrency bounds how many classifiers build at once. The
// model runtime batches concurrent calls; the bound keeps a router well inside
// the runtime's per-process request queue.
const maxEmbeddingConcurrency = 8

// classifierBuildParallelism is the number of classifiers built at once.
func classifierBuildParallelism(steps int) int {
	return max(1, min(steps, runtime.NumCPU(), maxEmbeddingConcurrency))
}
