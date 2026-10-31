package extproc

import (
	"context"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

func TestDecisionMediaDoesNotReachResponseStage(t *testing.T) {
	media := map[string]modelservice.DecisionMedia{
		"primary": {
			Images: []string{"data:image/png;base64,iVBORw0KGgo="},
			Videos: []string{"data:video/mp4;base64,AAAAFGZ0eXBpc29t"},
		},
	}
	ctx := &RequestContext{TraceContext: t.Context(), decisionMedia: media}
	for _, tc := range []struct {
		name   string
		parent context.Context
	}{
		{"plain trace control", ctx.TraceContext},
		{"response selection context", selectionRequestContext(ctx)},
		{"response without trace", selectionRequestContext(&RequestContext{decisionMedia: media})},
	} {
		t.Run(tc.name, func(t *testing.T) {
			definition, _ := modelservice.BuiltinTask("jailbreak")
			question := definition.Question
			question.ID = decisionSelectorQuestionID
			plan, err := modelservice.CompileTask(definition, question, modelservice.ModelCard{Surfaces: []string{"decisions"}, QuestionTypes: []string{"choice"}})
			if err != nil {
				t.Fatal(err)
			}
			decider := &recordingDecider{answer: modelservice.Answer{Type: "choice", Choice: "benign", Probabilities: map[string]float64{"benign": 0.8, "jailbreak": 0.2}, InputCoverage: "complete"}}
			var callErr error
			var result modelservice.TaskResult
			runResponseStage(tc.parent, responseStageCheck{run: func(call context.Context) {
				var results map[string]modelservice.TaskResult
				_, results, callErr = modelservice.ExecuteTaskPlans(call, decider, "primary",
					modelservice.Request{State: "assistant output"}, []modelservice.TaskPlan{plan})
				result = results[question.ID]
			}})
			if callErr != nil || result.Answer.Error != "" || len(decider.requests) != 1 {
				t.Fatalf("response judgment failed: result=%+v err=%v calls=%d", result, callErr, len(decider.requests))
			}
			request := decider.requests[0]
			if request.State != "assistant output" || len(request.Images) != 0 || len(request.Videos) != 0 {
				t.Fatalf("response judgment inherited request media: text=%q images=%d videos=%d", request.State, len(request.Images), len(request.Videos))
			}
		})
	}
}
