package systemone

import (
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	setQuestion = `{"type":"set","instructions":"Which apply?","criteria":{"pii":"Personal data","spam":"Unsolicited mail"}}`
	setRequest  = `{"model":"vllm-sr/auto","state":"Mail me at a@example.com","questions":{"tags":` + setQuestion + `}}`
	// One state's part of a response. pii is selected at 0.9 and spam is not
	// at 0.2, so the margins are 0.8 and 0.6.
	setAnswered = `"answers":{"tags.pii":{"type":"noul","noul":0.9},"tags.spam":{"type":"noul","noul":0.2}},"sets":{"tags":{"selected":["pii"],"probabilities":{"pii":0.9,"spam":0.2}}},"thresholds":{"tags":0.5}`
	// No label reaches the threshold: the selection is empty.
	noneAnswered = `"answers":{"tags.pii":{"type":"noul","noul":0.1},"tags.spam":{"type":"noul","noul":0.2}},"sets":{"tags":{"selected":[],"probabilities":{"pii":0.1,"spam":0.2}}},"thresholds":{"tags":0.5}`
	setResponse  = `{"model":"set-actual",` + setAnswered + `,"usage":{"input_tokens":12,"output_tokens":0}}`
)

// setMargin gates every set answer on its probability margin.
func setMargin(field string, threshold float64) *config.EscalateAccept {
	return &config.EscalateAccept{Native: []config.NativeAcceptanceRule{{QuestionType: "set", Field: field, NumericPredicate: config.NumericPredicate{GTE: &threshold}}}}
}

// Auto admits a set question by the labels it names, in criteria or in the
// vllm-sr/4.0 labels list, and needs a model that answers sets.
func TestAutoAdmitsASetQuestionByTheLabelsItNames(t *testing.T) {
	labels := map[string]bool{"pii": true, "spam": true}
	for name, question := range map[string]string{
		"criteria":    setQuestion,
		"labels list": `{"type":"set","instructions":"Which apply?","labels":[{"key":"pii","description":"Personal data"},{"key":"spam"}]}`,
	} {
		request, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","state":"hi","questions":{"tags":` + question + `}}`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if q := request.Questions[0]; q.Type != "set" || !reflect.DeepEqual(q.Labels, labels) {
			t.Fatalf("%s: question %+v", name, q)
		}
		if !reflect.DeepEqual(request.Requirements.QuestionTypes, []string{"set"}) {
			t.Fatalf("%s: requirements %+v", name, request.Requirements)
		}
	}
	for name, questions := range map[string]string{
		"no labels":                     `{"tags":{"type":"set","instructions":"Which apply?"}}`,
		"empty criteria":                `{"tags":{"type":"set","instructions":"Which apply?","criteria":{}}}`,
		"criteria as a list":            `{"tags":{"type":"set","instructions":"Which apply?","criteria":["pii","spam"]}}`,
		"a label answer's name is used": `{"tags":` + setQuestion + `,"tags.pii":{"type":"noul","instructions":"Is it personal?"}}`,
	} {
		if _, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","state":"hi","questions":` + questions + `}`)); err == nil {
			t.Fatalf("%s accepted: %s", name, questions)
		}
	}
}

// A set answer is valid when its own entry and its label Nouls both cover
// every declared label and agree. An empty selection is an answer; a missing
// or malformed part is not.
func TestASetAnswerNeedsBothFormsForEveryLabelAndTheirAgreement(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(setRequest))
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name, old, new string
		valid          bool
	}{
		{"as the runtime writes it", "", "", true},
		{"no label selected", setAnswered, noneAnswered, true},
		{"forms that differ within the tolerance", `"tags.pii":{"type":"noul","noul":0.9}`, `"tags.pii":{"type":"noul","noul":0.90001}`, true},
		{"no sets", `,"sets":{"tags":{"selected":["pii"],"probabilities":{"pii":0.9,"spam":0.2}}}`, ``, false},
		{"a null entry", `{"selected":["pii"],"probabilities":{"pii":0.9,"spam":0.2}}`, `null`, false},
		{"no selection", `"selected":["pii"],`, ``, false},
		{"a null selection", `"selected":["pii"]`, `"selected":null`, false},
		{"an undeclared label selected", `"selected":["pii"]`, `"selected":["pii","other"]`, false},
		{"null probabilities", `"probabilities":{"pii":0.9,"spam":0.2}`, `"probabilities":null`, false},
		{"a label without a probability", `{"pii":0.9,"spam":0.2}`, `{"pii":0.9}`, false},
		{"an undeclared label's probability", `{"pii":0.9,"spam":0.2}`, `{"pii":0.9,"spam":0.2,"other":0.5}`, false},
		{"a null probability", `"spam":0.2}}`, `"spam":null}}`, false},
		{"a probability out of range", `{"pii":0.9,"spam":0.2}`, `{"pii":1.9,"spam":0.2}`, false},
		{"a label without its Noul", `,"tags.spam":{"type":"noul","noul":0.2}`, ``, false},
		{"a label answered as another type", `"tags.spam":{"type":"noul","noul":0.2}`, `"tags.spam":{"type":"score","score":0.2}`, false},
		{"a failed label", `"tags.spam":{"type":"noul","noul":0.2}`, `"tags.spam":{"type":"noul","error":"unavailable"}`, false},
		{"forms that disagree", `"tags.pii":{"type":"noul","noul":0.9}`, `"tags.pii":{"type":"noul","noul":0.8}`, false},
		{"a failed set", `"answers":{`, `"answers":{"tags":{"type":"set","error":"unavailable"},`, false},
	} {
		body := strings.Replace(setResponse, tc.old, tc.new, 1)
		if tc.old != "" && body == setResponse {
			t.Fatalf("%s: the response was not changed", tc.name)
		}
		observation := request.Observe(json.RawMessage(body))[0]
		if observation.Valid != tc.valid {
			t.Errorf("%s: valid %v, want %v: %s", tc.name, observation.Valid, tc.valid, body)
		}
		if !tc.valid && observation.LabelProbabilities != nil {
			t.Errorf("%s: an invalid set kept probabilities %v", tc.name, observation.LabelProbabilities)
		}
	}
	if got := request.Observe(json.RawMessage(setResponse))[0].LabelProbabilities; !reflect.DeepEqual(got, map[string]float64{"pii": 0.9, "spam": 0.2}) {
		t.Fatalf("label probabilities %v", got)
	}

	// A set that requires its full input needs the coverage its entry proves.
	full, err := ParseNativeRequest(json.RawMessage(strings.Replace(setRequest, `"type":"set"`, `"type":"set","require_full_input":true`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if full.Observe(json.RawMessage(setResponse))[0].Valid {
		t.Fatal("a set without input coverage answered a question that requires full input")
	}
	covered := strings.Replace(setResponse, `"selected":["pii"]`, `"input_coverage":"complete","selected":["pii"]`, 1)
	if !full.Observe(json.RawMessage(covered))[0].Valid {
		t.Fatal("a set with complete input coverage was refused")
	}
}

// Each batch entry's set is observed from that entry. An empty selection is
// an answer and a missing entry is not; a malformed set fails its own
// question and no other.
func TestABatchObservesTheSetOfEachEntry(t *testing.T) {
	const (
		entry = `{"state":"hi","questions":{"tags":` + setQuestion + `,"risk":{"type":"noul","instructions":"Unsafe?"}}}`
		risk  = `"answers":{"risk":{"type":"noul","noul":0.01}`
	)
	request, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","states":{"found":` + entry + `,"none":` + entry + `,"missing":` + entry + `,"malformed":` + entry + `}}`))
	if err != nil {
		t.Fatal(err)
	}
	answered := func(part string) string { return "{" + strings.Replace(part, `"answers":{`, risk+",", 1) + "}" }
	response := `{"model":"m","states":{` +
		`"found":` + answered(setAnswered) + `,` +
		`"none":` + answered(noneAnswered) + `,` +
		`"missing":{` + risk + `}},` +
		`"malformed":` + answered(strings.Replace(setAnswered, `{"pii":0.9,"spam":0.2}`, `{"pii":0.9}`, 1)) + `}}`
	got := map[string]bool{}
	for _, observation := range request.Observe(json.RawMessage(response)) {
		got[observation.State+"."+observation.Name] = observation.Valid
	}
	want := map[string]bool{
		"found.tags": true, "found.risk": true,
		"none.tags": true, "none.risk": true,
		"missing.tags": false, "missing.risk": true,
		"malformed.tags": false, "malformed.risk": true,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("valid answers %v, want %v: %s", got, want, response)
	}
}

// A set is as sure as its least sure label: its probability margin is the
// smallest of its labels', and it has no top probability and no confidence.
func TestASetIsGatedByItsLeastSureLabel(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(setRequest))
	if err != nil {
		t.Fatal(err)
	}
	observations := request.Observe(json.RawMessage(setResponse))
	if !acceptNative(observations, setMargin("probability_margin", 0.5), true) {
		t.Fatal("a set whose labels all pass the margin was rejected")
	}
	if acceptNative(observations, setMargin("probability_margin", 0.7), true) {
		t.Fatal("a set passed on its surest label")
	}
	for _, field := range []string{"top_probability", "confidence"} {
		if acceptNative(observations, setMargin(field, 0), true) {
			t.Fatalf("a set answer was given a %s", field)
		}
	}
	none := request.Observe(json.RawMessage(`{"model":"m",` + noneAnswered + `}`))
	if !acceptNative(none, setMargin("probability_margin", 0.5), true) {
		t.Fatal("a set that surely selects nothing was rejected")
	}
}

// A set request moves on while a label is unsure, and the accepted model's
// response is returned as it wrote it, for a standard request and a batch.
func TestASetRequestEscalatesUntilEveryLabelIsSure(t *testing.T) {
	const entry = `{"state":"hi","questions":{"tags":` + setQuestion + `}}`
	unsure := strings.NewReplacer(`"noul":0.2`, `"noul":0.45`, `"spam":0.2`, `"spam":0.45`).Replace(setAnswered)
	for name, shape := range map[string]struct {
		request string
		respond func(part string) string
	}{
		"standard": {setRequest, func(part string) string { return `{"model":"m",` + part + `}` }},
		"batch": {`{"model":"vllm-sr/auto","states":{"a":` + entry + `,"b":` + entry + `}}`, func(part string) string {
			return `{"model":"m","states":{"a":{` + setAnswered + `},"b":{` + part + `}}}`
		}},
	} {
		for fast, want := range map[string][]string{setAnswered: {"kai"}, unsure: {"kai", "nox"}} {
			request, err := ParseNativeRequest(json.RawMessage(shape.request))
			if err != nil {
				t.Fatal(err)
			}
			executor, err := NewExecutor(&config.AlgorithmConfig{
				Type:   config.DecisionAlgorithmEscalate,
				Budget: &config.AlgorithmBudget{Deadline: "3s", MaxCalls: 3},
				Escalate: &config.EscalateAlgorithmConfig{
					Quality: &config.EscalateQuality{Type: "uncalibrated", Accept: setMargin("probability_margin", 0)},
					Stages: []config.EscalateStage{
						{Name: "fast", Kind: "native", Model: "kai", Accept: setMargin("probability_margin", 0.5)},
						{Name: "strong", Kind: "native", Model: "nox"},
					},
				},
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			var calls []string
			answers := map[string]string{"kai": shape.respond(fast), "nox": shape.respond(setAnswered)}
			result, err := executor.Execute(context.Background(), request, func(_ context.Context, model string, body json.RawMessage) (int, []byte, error) {
				calls = append(calls, model)
				if string(body) != shape.request {
					t.Fatal("native task rewritten")
				}
				return 200, []byte(answers[model]), nil
			})
			if err != nil || !reflect.DeepEqual(calls, want) {
				t.Fatalf("%s: calls %v, want %v, error %v", name, calls, want, err)
			}
			if accepted := want[len(want)-1]; result.Model != accepted || string(result.Body) != answers[accepted] {
				t.Fatalf("%s: result %+v, want the response of %s unchanged", name, result, accepted)
			}
		}
	}
}

// A set question goes only to the member that lists sets, and to none when
// no member does.
func TestASetQuestionGoesOnlyToMembersThatAnswerSets(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(setRequest))
	if err != nil {
		t.Fatal(err)
	}
	executor, err := NewExecutor(poolPlan(), nil)
	if err != nil {
		t.Fatal(err)
	}
	var calls []string
	invoke := func(_ context.Context, model string, _ json.RawMessage) (int, []byte, error) {
		calls = append(calls, model)
		return 200, []byte(setResponse), nil
	}
	sets := textOnly("listing")
	sets.QuestionTypes = append(sets.QuestionTypes, "set")
	ctx := withMembers(context.Background(), members(map[string]Capabilities{"kai": textOnly("card"), "vega": sets}))
	result, err := executor.Execute(ctx, request, invoke)
	if err != nil || result.Model != "vega" || !reflect.DeepEqual(calls, []string{"vega"}) {
		t.Fatalf("result %+v, calls %v, error %v", result, calls, err)
	}

	calls = nil
	ctx = withMembers(context.Background(), members(map[string]Capabilities{"kai": textOnly("card"), "vega": textOnly("listing")}))
	_, err = executor.Execute(ctx, request, invoke)
	var refusal *IncapableError
	if !errors.As(err, &refusal) || !reflect.DeepEqual(refusal.Missing, []string{"set"}) || len(calls) != 0 {
		t.Fatalf("error %v, calls %v", err, calls)
	}
}

// A judge is shown each candidate's whole response, its sets among it.
func TestAJudgeIsShownTheSetsOfEachCandidate(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(setRequest))
	if err != nil {
		t.Fatal(err)
	}
	stage := config.EscalateStage{Name: "review", Kind: "judge", Model: "reviewer", Generation: &config.NativeGenerationConfig{MaxOutputTokens: 128}}
	prompt, err := JudgeRequest(request, stage, []Candidate{{Stage: "fast", Body: json.RawMessage(setResponse)}})
	if err != nil {
		t.Fatal(err)
	}
	var chat struct {
		Messages []struct {
			Content string `json:"content"`
		} `json:"messages"`
	}
	var shown struct {
		Request    json.RawMessage            `json:"request"`
		Candidates map[string]json.RawMessage `json:"candidates"`
	}
	if json.Unmarshal(prompt, &chat) != nil || len(chat.Messages) != 2 || json.Unmarshal([]byte(chat.Messages[1].Content), &shown) != nil {
		t.Fatalf("judge request %s", prompt)
	}
	if string(shown.Request) != setRequest || string(shown.Candidates["fast"]) != setResponse {
		t.Fatalf("the judge was shown %s", chat.Messages[1].Content)
	}
}
