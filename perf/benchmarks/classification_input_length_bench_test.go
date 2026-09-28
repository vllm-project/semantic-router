//go:build !windows && cgo

package benchmarks

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

var (
	inputLengthTokens = []int{512, 2048, 8192}
	inputLengthUnit   = strings.Join(testTexts, "\n") + "\n"
	// The shared 512-token truncating deployment would time only a prefix of a
	// long input. Rejecting instead fails a fixture that outgrows the budget.
	inputLengthBudget = config.ModelInputBudget{MaxTokens: inputLengthTokens[len(inputLengthTokens)-1], Overflow: "reject"}

	inputLengthOnce                                  sync.Once
	inputLengthDomain                                *binding.Resolved[string, tasks.LabelDistribution]
	inputLengthErr                                   error
	inputLengthUnitTokens, inputLengthTemplateTokens int
)

func initInputLengthDomain(b *testing.B) {
	b.Helper()
	inputLengthOnce.Do(func() {
		spec := benchmarkModel(b, "domain", "label_distribution.v1")
		spec.Binding.Deployment += "-input-length"
		spec.Deployment.Input = inputLengthBudget
		if inputLengthDomain, inputLengthErr = benchmarkRuntime.Sequence(context.Background(), spec); inputLengthErr != nil {
			return
		}
		one, err := inputTokens(inputLengthUnit)
		if err != nil {
			inputLengthErr = err
			return
		}
		two, err := inputTokens(strings.Repeat(inputLengthUnit, 2))
		if err != nil {
			inputLengthErr = err
			return
		}
		inputLengthUnitTokens, inputLengthTemplateTokens = two-one, 2*one-two
	})
	if inputLengthErr != nil {
		if missingBenchModels(inputLengthErr) {
			b.Skipf("Failed to initialize domain classifier: %v", inputLengthErr)
		}
		b.Fatalf("prepare owned Vela Domain with a %d-token budget: %v", inputLengthBudget.MaxTokens, inputLengthErr)
	}
}

func inputTokens(text string) (int, error) {
	result, err := inputLengthDomain.Call(context.Background(), "perf", text)
	if err != nil {
		return 0, err
	}
	if result.Input == nil || result.Input.Truncated {
		return 0, fmt.Errorf("domain result did not report complete input usage: %+v", result.Input)
	}
	return result.Input.ProcessedTokens, nil
}

func recordInputLengthProtocol(b *testing.B) {
	recordModelIdentity(b, "domain")
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	identity := benchmarkIdentities[b.Name()]
	identity.Protocol = fmt.Sprintf("owned-native-v1;max_tokens=%d;overflow=%s;embedding=full-layer/full-dimension;input=repeated-fixture",
		inputLengthBudget.MaxTokens, inputLengthBudget.Overflow)
	benchmarkIdentities[b.Name()] = identity
}

// Each size repeats the fixture prompts up to, never past, its token count.
func BenchmarkClassifyInputLength(b *testing.B) {
	initInputLengthDomain(b)
	for _, tokens := range inputLengthTokens {
		text := strings.Repeat(inputLengthUnit, (tokens-inputLengthTemplateTokens)/inputLengthUnitTokens)
		b.Run(fmt.Sprintf("tokens_%d", tokens), func(b *testing.B) {
			recordInputLengthProtocol(b)
			processed := 0
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				result, err := inputLengthDomain.Call(context.Background(), "perf", text)
				if err != nil {
					b.Fatalf("classify %d-token input: %v", tokens, err)
				}
				processed = result.Input.ProcessedTokens
			}
			b.StopTimer()
			if processed > tokens || processed <= tokens-inputLengthUnitTokens {
				b.Fatalf("fixture processed %d tokens, want (%d, %d]", processed, tokens-inputLengthUnitTokens, tokens)
			}
			b.ReportMetric(float64(processed), "tokens/op")
		})
	}
}
