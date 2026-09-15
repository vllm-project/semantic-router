package native

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

func TestTokenWindowsRequireCompleteOriginalCoverageAndValidSpans(t *testing.T) {
	scored := true
	output := tasks.WindowedTokenClassification{ContentTokens: 4, Windows: [][2]int{{0, 3}, {2, 4}}, Result: tasks.TokenClassificationResult{Input: &tasks.InputUsage{OriginalTokens: 6, ProcessedTokens: 6}, ScoresAvailable: &scored, Entities: []tasks.TokenEntity{{EntityType: "PERSON", Text: "猫", Start: 1, End: 4, Confidence: .8}}}}
	input := tasks.TextWindowsRequest{Text: "a猫b", Size: 5, Overlap: 1}
	if err := validateWindowTokens(input, output); err != nil {
		t.Fatal(err)
	}
	for name, change := range map[string]func(*tasks.WindowedTokenClassification){
		"missing tail": func(o *tasks.WindowedTokenClassification) { o.Windows = [][2]int{{0, 3}} },
		"hole":         func(o *tasks.WindowedTokenClassification) { o.Windows = [][2]int{{0, 1}, {2, 4}} },
		"truncated": func(o *tasks.WindowedTokenClassification) {
			o.Result.Input = &tasks.InputUsage{OriginalTokens: 6, ProcessedTokens: 5, Truncated: true}
		},
		"unscored": func(o *tasks.WindowedTokenClassification) { o.Result.ScoresAvailable = nil },
		"offset":   func(o *tasks.WindowedTokenClassification) { o.Result.Entities[0].End = 3 },
		"nan":      func(o *tasks.WindowedTokenClassification) { o.Result.Entities[0].Confidence = float32(math.NaN()) },
	} {
		t.Run(name, func(t *testing.T) {
			c := output
			c.Result.Entities = append([]tasks.TokenEntity(nil), output.Result.Entities...)
			change(&c)
			if validateWindowTokens(input, c) == nil {
				t.Fatal("invalid window result accepted")
			}
		})
	}
}

func TestTokenWindowModeCannotFallThroughToSingleInput(t *testing.T) {
	spec := config.ResolvedModelBinding{Deployment: config.ModelDeployment{Provider: "candle", Input: config.ModelInputBudget{MaxTokens: 32768, Overflow: "window"}}}
	if _, err := New(nil).Tokens(context.Background(), spec); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("window silently ignored: %v", err)
	}
	limits := binding.Capability{Limits: binding.Limits{ModelTokens: 512, TaskTokens: 512}}
	if !errors.Is(validateTokenWindowBudget(spec, limits), binding.ErrCapability) {
		t.Fatal("custom low-capacity model silently promoted")
	}
	spec.Deployment.Input.MaxTokens = 512
	if err := validateTokenWindowBudget(spec, limits); err != nil {
		t.Fatal(err)
	}
}
