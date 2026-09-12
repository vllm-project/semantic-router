package classification

import (
	"context"
	"errors"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime"
)

func TestRoutingModelFailureStopsStartupButAPIFailureDoesNot(t *testing.T) {
	for _, routing := range []bool{false, true} {
		for _, model := range []struct {
			task   string
			signal string
		}{
			{"classifier.fact_check", config.SignalTypeFactCheck},
			{"classifier.feedback", config.SignalTypeUserFeedback},
		} {
			name := model.task + "/api-only"
			if routing {
				name = model.task + "/routing"
			}
			t.Run(name, func(t *testing.T) {
				cfg := &config.RouterConfig{
					InlineModels: config.InlineModels{
						HallucinationMitigation: config.HallucinationMitigationConfig{
							FactCheckModel: config.FactCheckModelConfig{ModelID: "models/test-fact-check"},
						},
						FeedbackDetector: config.FeedbackDetectorConfig{Enabled: true, ModelID: "models/test-feedback"},
					},
					IntelligentRouting: config.IntelligentRouting{
						Signals: config.Signals{
							FactCheckRules:    []config.FactCheckRule{{Name: "required"}},
							UserFeedbackRules: []config.UserFeedbackRule{{Name: "required"}},
						},
					},
				}
				if routing {
					cfg.Decisions = []config.Decision{{
						Name: "dependent-route", Rules: config.RuleNode{Type: model.signal, Name: "required"},
					}}
				}
				classifier := &Classifier{Config: cfg}
				failure := errors.New("model artifact unavailable")
				for _, task := range classifier.runtimeTasks() {
					if task.Name != model.task {
						continue
					}
					// Execute the real lifecycle policy with a failed initializer;
					// no model download or native fixture is needed for this failure.
					task.Run = func(context.Context) error { return failure }
					err := classifier.executeRuntimeTasks([]modelruntime.Task{task})
					if routing && !errors.Is(err, failure) {
						t.Fatalf("required model failure did not abort startup: %v", err)
					}
					if !routing && err != nil {
						t.Fatalf("API-only failure aborted unrelated routing: %v", err)
					}
					return
				}
				t.Fatalf("runtime task %q was omitted", model.task)
			})
		}
	}
}
