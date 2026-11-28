package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// This synthetic, in-process test isolates joint-model fusion; it does not
// exercise a network listener or establish actual model numerical behaviour.
func TestPrivateJointFusionDependsOnLogicalQuestionSet(t *testing.T) {
	choice := func(id string) Question {
		return Question{ID: id, Type: "choice", Instructions: "Choose", Choices: []Choice{{Key: "benign"}, {Key: "jailbreak"}}}
	}
	security := choice("preset:jailbreak")
	questions := []Question{choice("preset:domain"), {ID: "preset:pii", Type: "span", Instructions: "Find person", Labels: []Choice{{Key: "PERSON"}}}, security}
	fake := runtimetest.New(runtimetest.Model{ID: "fixture", Joint: true, Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON"}}})
	ask := func(questions []Question) float64 {
		t.Helper()
		calls := make([]*bundleCall, 0, len(questions))
		for _, q := range questions {
			req := Request{Model: "fixture", State: "person text", Questions: []Question{q}}
			body, err := encodeDecisionRequest(context.Background(), req)
			if err != nil {
				t.Fatal(err)
			}
			calls = append(calls, &bundleCall{ctx: context.Background(), task: api.BundleTask{Id: q.ID, Decisions: &body}, decision: &decisionCall{request: req}})
		}
		tasks := fuseDecisions(100, calls)
		if len(tasks) != 1 || len(tasks[0].decision.states) != 1 || len(tasks[0].task.Decisions.Questions) != len(questions) {
			t.Fatalf("unexpected fusion: %+v", tasks)
		}
		data, err := json.Marshal(tasks[0].task.Decisions)
		if err != nil {
			t.Fatal(err)
		}
		recorder := httptest.NewRecorder()
		fake.Handler().ServeHTTP(recorder, httptest.NewRequest(http.MethodPost, "/v1/decisions", bytes.NewReader(data)))
		if recorder.Code != http.StatusOK {
			t.Fatalf("fixture status %d: %s", recorder.Code, recorder.Body.String())
		}
		var response api.DecisionResponse
		if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
			t.Fatal(err)
		}
		answer := response.Answers["preset:jailbreak"]
		if answer.Confidence == nil {
			t.Fatalf("no confidence: %+v", answer)
		}
		return *answer.Confidence
	}
	alone, together := ask([]Question{security}), ask(questions)
	if math.Abs(alone-.7) > 1e-9 || math.Abs(together-.6) > 1e-9 {
		t.Fatalf("joint fixture did not reflect question set: alone=%v together=%v", alone, together)
	}
}
