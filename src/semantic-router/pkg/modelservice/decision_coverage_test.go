package modelservice

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

func TestFullInputProofSurvivesNativeTransport(t *testing.T) {
	for _, tc := range []struct {
		name, kind, body string
	}{
		{"scalar", "noul", `{"answers":{"q":{"type":"noul","noul":0.01,"input_coverage":"complete"}}}`},
		{"set", "set", `{"answers":{},"sets":{"q":{"selected":[],"probabilities":{"email":0.01},"input_coverage":"complete"}}}`},
		{"span", "span", `{"answers":{"q":{"type":"noul","noul":0.01,"input_coverage":"complete"}},"spans":{"q":[]}}`},
	} {
		t.Run(tc.name, func(t *testing.T) {
			for _, proof := range []bool{true, false} {
				body := tc.body
				if !proof {
					body = strings.ReplaceAll(body, `,"input_coverage":"complete"`, "")
				}
				var response api.DecisionResponse
				if err := json.Unmarshal([]byte(body), &response); err != nil {
					t.Fatal(err)
				}
				question := Question{ID: "q", Type: tc.kind, RequireFullInput: true}
				answer := decodeResponse(response, []Question{question}).Answers["q"]
				if proof {
					if answer.Error != "" || answer.InputCoverage != "complete" {
						t.Fatalf("complete evidence lost: %+v", answer)
					}
				} else if answer.Error != "input_coverage_unknown" {
					t.Fatalf("legacy runtime certified full input: %+v", answer)
				}
				question.RequireFullInput = false
				if answer := decodeResponse(response, []Question{question}).Answers["q"]; answer.Error != "" {
					t.Fatalf("permissive native answer rejected: %+v", answer)
				}
			}
		})
	}
}

func TestFullInputRequirementSeparatesCacheAndWireRequests(t *testing.T) {
	for _, question := range []Question{{ID: "q", Type: "noul", Instructions: "Personal data?"}, {ID: "q", Preset: "pii"}} {
		request := Request{State: "Alex", Questions: []Question{question}}
		permissive := decideKey(request)
		request.Questions[0].RequireFullInput = true
		if permissive == decideKey(request) {
			t.Fatal("strict request can reuse a permissive result")
		}
		bounded := boundedRead(request)
		encoded, err := json.Marshal(encodeQuestion(bounded.Questions[0]))
		if err != nil || !strings.Contains(string(encoded), `"require_full_input":true`) {
			t.Fatalf("strict contract lost for a model without scanning: %s %v", encoded, err)
		}
	}
}

func TestFullInputTaskRequiresProofFromEveryComposedLabel(t *testing.T) {
	definition, _ := BuiltinTask("pii_categories")
	definition.Question.Labels = []Choice{{Key: "email"}, {Key: "name"}}
	plan, err := CompileTask(definition, definition.Question, ModelCard{QuestionTypes: []string{"noul"}, Surfaces: []string{"decisions"}})
	if err != nil {
		t.Fatal(err)
	}
	for _, question := range plan.Questions {
		if !question.RequireFullInput {
			t.Fatal("composed label lost the task input requirement")
		}
	}
	for _, coverage := range []string{"", "partial", "unknown", "complete"} {
		response := Response{Answers: map[string]Answer{
			plan.Questions[0].ID: {Type: "noul", Noul: .01, InputCoverage: "complete"},
			plan.Questions[1].ID: {Type: "noul", Noul: .01, InputCoverage: coverage},
		}}
		answer := reduceTaskAnswer(plan, response)
		if coverage == "complete" {
			if answer.Error != "" || answer.InputCoverage != "complete" {
				t.Fatalf("complete set proof lost: %+v", answer)
			}
		} else if answer.Error != "input_coverage_unknown" {
			t.Fatalf("incomplete label certified clean: %+v", answer)
		}
	}
	definition.Question.Truncate = true
	if _, err := CompileTask(definition, definition.Question, ModelCard{}); err == nil {
		t.Fatal("contradictory full-input and truncation requirements accepted")
	}
}
