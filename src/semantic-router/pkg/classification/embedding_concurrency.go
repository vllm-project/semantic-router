package classification

import "runtime"

// maxEmbeddingConcurrency bounds how many classifiers build at once and how
// many embedding calls one classifier keeps in flight, whether it prepares its
// candidates or embeds a request's chunks. The model runtime batches
// concurrent calls; the bound keeps a router well inside the runtime's
// per-process request queue.
const maxEmbeddingConcurrency = 8

// embeddingWorkers is the number of concurrent embedding calls for tasks texts.
func embeddingWorkers(tasks int) int {
	return min(tasks, maxEmbeddingConcurrency)
}

// classifierBuildParallelism is the number of classifiers built at once.
func classifierBuildParallelism(steps int) int {
	return max(1, min(steps, runtime.NumCPU(), maxEmbeddingConcurrency))
}
