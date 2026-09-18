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
	limits.Limits.DocumentTokens = 512
	if err := validateTokenWindowBudget(spec, limits); err != nil {
		t.Fatal(err)
	}
}

func TestWindowCapabilityAdmitsDocumentsOnlyWithNativeScanEvidence(t *testing.T) {
	spec := config.ResolvedModelBinding{Deployment: config.ModelDeployment{Input: config.ModelInputBudget{MaxTokens: 262144, Overflow: "window"}}}
	window := tasks.TextWindowsRequest{Size: 32768, Overlap: 16383}
	capability := binding.Capability{Limits: binding.Limits{ModelTokens: 32768, TaskTokens: 32768, DocumentTokens: 262144}}
	if err := validateWindowCapability(spec, capability, window); err != nil {
		t.Fatal(err)
	}
	if err := validateTokenWindowBudget(spec, capability); err != nil {
		t.Fatal(err)
	}
	for name, mutate := range map[string]func(*binding.Capability, *tasks.TextWindowsRequest){
		"unknown scan":     func(c *binding.Capability, _ *tasks.TextWindowsRequest) { c.Limits.DocumentTokens = 0 },
		"short scan":       func(c *binding.Capability, _ *tasks.TextWindowsRequest) { c.Limits.DocumentTokens = 32768 },
		"task too small":   func(c *binding.Capability, _ *tasks.TextWindowsRequest) { c.Limits.TaskTokens = 512 },
		"window too large": func(_ *binding.Capability, w *tasks.TextWindowsRequest) { w.Size = 32769 },
	} {
		t.Run(name, func(t *testing.T) {
			c, w := capability, window
			mutate(&c, &w)
			if !errors.Is(validateWindowCapability(spec, c, w), binding.ErrCapability) {
				t.Fatal("unsupported full-document scan accepted")
			}
		})
	}
}
