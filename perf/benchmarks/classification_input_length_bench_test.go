//go:build !windows

package benchmarks

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
)

var (
	inputLengthTokens = []int{512, 2048, 8192}
	inputLengthUnit   = strings.Join(testTexts, "\n") + "\n"
	// The shared 512-token truncating deployment would time only a prefix of a
	// long input. Rejecting instead fails a fixture that outgrows the budget.
	inputLengthBudget = config.ModelInputBudget{MaxTokens: inputLengthTokens[len(inputLengthTokens)-1], Overflow: "reject"}

	inputLengthOnce                                  sync.Once
	inputLengthDomain                                *signalClassifier
	inputLengthErr                                   error
	inputLengthUnitTokens, inputLengthTemplateTokens int
)

func initInputLengthDomain(b *testing.B) {
	b.Helper()
	inputLengthOnce.Do(func() {
		inputLengthErr = fmt.Errorf("an earlier benchmark stopped while preparing it")
		inputLengthDomain = newSignalClassifier(b, "perf-decision-input-length", inputLengthBudget, presets.Domain)
		// The domain question reads as much of the text as the budget allows.
		inputLengthDomain.questions[0].Truncate = false
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
		inputLengthUnitTokens, inputLengthTemplateTokens, inputLengthErr = two-one, 2*one-two, nil
	})
	if inputLengthErr != nil {
		b.Fatalf("prepare the decision model with a %d-token budget: %v", inputLengthBudget.MaxTokens, inputLengthErr)
	}
}

func inputTokens(text string) (int, error) {
	response, err := inputLengthDomain.ask(context.Background(), text)
	if err != nil {
		return 0, err
	}
	answer := response.Answers[inputLengthDomain.questions[0].ID]
	if answer.Error != "" || answer.InputCoverage != "complete" {
		return 0, fmt.Errorf("domain answer did not read the complete input: %+v", answer)
	}
	return response.InputTokens, nil
}

func recordInputLengthProtocol(b *testing.B) {
	recordModelIdentity(b, "decision")
	benchmarkModelMu.Lock()
	defer benchmarkModelMu.Unlock()
	identity := benchmarkIdentities[b.Name()]
	card := benchmarkCards["decision"]
	protocol, err := modelRuntimeProtocol(card, inputLengthBudget, "full-layer/full-dimension")
	if err != nil {
		b.Fatalf("record input-length execution settings: %v", err)
	}
	identity.Protocol = protocol + ";input=repeated-fixture"
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
				response, err := inputLengthDomain.ask(context.Background(), text)
				if err != nil {
					b.Fatalf("classify %d-token input: %v", tokens, err)
				}
				processed = response.InputTokens
			}
			b.StopTimer()
			if processed > tokens || processed <= tokens-inputLengthUnitTokens {
				b.Fatalf("fixture processed %d tokens, want (%d, %d]", processed, tokens-inputLengthUnitTokens, tokens)
			}
			b.ReportMetric(float64(processed), "tokens/op")
		})
	}
}
