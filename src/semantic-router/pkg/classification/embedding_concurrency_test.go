package classification

import (
	"runtime"
	"testing"
)

func TestEmbeddingWorkersAreBoundedByTasksAndConcurrency(t *testing.T) {
	for tasks, want := range map[int]int{0: 0, 1: 1, 3: 3, maxEmbeddingConcurrency: maxEmbeddingConcurrency, 10_000: maxEmbeddingConcurrency} {
		if got := embeddingWorkers(tasks); got != want {
			t.Fatalf("embeddingWorkers(%d) = %d, want %d", tasks, got, want)
		}
	}
}

func TestClassifierBuildParallelismIsBounded(t *testing.T) {
	bound := min(runtime.NumCPU(), maxEmbeddingConcurrency)
	for steps, want := range map[int]int{0: 1, 1: 1, 13: bound, 10_000: bound} {
		if got := classifierBuildParallelism(steps); got != want {
			t.Fatalf("classifierBuildParallelism(%d) = %d, want %d", steps, got, want)
		}
	}
}
