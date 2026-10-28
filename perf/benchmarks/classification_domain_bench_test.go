//go:build !windows

package benchmarks

import (
	"context"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

var (
	domainOnce sync.Once
	domainTask *binding.Resolved[string, tasks.LabelDistribution]
	domainErr  error
)

// Model quality is asserted by the runtime's golden answers. This
// microbenchmark measures Domain inference without an optional gold dataset.
func BenchmarkClassifyDomain(b *testing.B) {
	domainOnce.Do(func() {
		spec := benchmarkModel(b, "domain", "label_distribution.v1")
		domainTask, domainErr = benchmarkRuntime.Sequence(context.Background(), spec)
	})
	if domainErr != nil {
		b.Fatalf("prepare Vela Domain: %v", domainErr)
	}
	recordModelIdentity(b, "domain")
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := domainTask.Call(context.Background(), "perf", testTexts[i%len(testTexts)]); err != nil {
			b.Fatal(err)
		}
	}
}
