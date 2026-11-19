//go:build !windows

package benchmarks

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

var (
	domainOnce sync.Once
	domainTask *signalClassifier
)

// Model quality is asserted by the runtime's golden answers. This
// microbenchmark measures the decision model's Domain question without an
// optional gold dataset.
func BenchmarkClassifyDomain(b *testing.B) {
	domainOnce.Do(func() {
		domainTask = newSignalClassifier(b, "perf-decision", benchmarkInput, presets.Domain)
	})
	if domainTask == nil {
		b.Fatal("prepare the decision model: an earlier benchmark stopped while preparing it")
	}
	recordModelIdentity(b, "decision")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := domainTask.ask(context.Background(), testTexts[i%len(testTexts)]); err != nil {
			b.Fatal(err)
		}
	}
}
