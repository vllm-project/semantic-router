package systemonecodec

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

func TestTheVendoredSpecsAreTheFilesTheTablesWereGeneratedFrom(t *testing.T) {
	for _, spec := range VendoredSpecs() {
		data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", spec.Path))
		if err != nil {
			t.Fatal(err)
		}
		sum := sha256.Sum256(data)
		if hex.EncodeToString(sum[:]) != spec.SHA256 {
			t.Fatalf("%s changed since the tables were generated; run make systemone-schemas-generate", spec.Path)
		}
	}
}

type fakeCodec struct {
	SystemOne02Codec
	id        SchemaID
	stateful  bool
	wrongHash bool
}

func (c fakeCodec) ID() SchemaID    { return c.id }
func (c fakeCodec) Stateless() bool { return !c.stateful }
func (c fakeCodec) Spec() SpecRef {
	spec, _ := VendoredSpec(c.id)
	if c.wrongHash {
		spec.SHA256 = strings.Repeat("0", 64)
	}
	return spec
}

type fakeDeviation struct {
	StateTextDeviation
	id     string
	schema SchemaID
}

func (d fakeDeviation) ID() string                     { return d.id }
func (d fakeDeviation) AppliesTo(schema SchemaID) bool { return schema == d.schema }

func TestTheRegistryValidatesWhatItHolds(t *testing.T) {
	for name, tc := range map[string]struct {
		codecs     []SchemaCodec
		deviations []Deviation
		want       string
	}{
		"nil codec":         {codecs: []SchemaCodec{nil}, want: "is nil"},
		"bad id":            {codecs: []SchemaCodec{fakeCodec{id: "SystemOne"}}, want: "invalid ID"},
		"stateful":          {codecs: []SchemaCodec{fakeCodec{id: SchemaSystemOne02, stateful: true}}, want: "not stateless"},
		"duplicate":         {codecs: []SchemaCodec{SystemOne02Codec{}, SystemOne02Codec{}}, want: "more than once"},
		"no vendored spec":  {codecs: []SchemaCodec{fakeCodec{id: "systemone/0.9"}}, want: "no vendored specification"},
		"wrong sha256":      {codecs: []SchemaCodec{fakeCodec{id: SchemaSystemOne02, wrongHash: true}}, want: "not the vendored"},
		"empty":             {want: "empty"},
		"bad deviation id":  {codecs: []SchemaCodec{SystemOne02Codec{}}, deviations: []Deviation{fakeDeviation{id: "State-Text", schema: SchemaSystemOne02}}, want: "invalid ID"},
		"deviation applies": {codecs: []SchemaCodec{VLLMSR50Codec{}}, deviations: []Deviation{StateTextDeviation{}}, want: "applies to no registered schema"},
	} {
		t.Run(name, func(t *testing.T) {
			_, err := NewRegistry(tc.codecs, tc.deviations)
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err %v, want %q", err, tc.want)
			}
		})
	}
}

func TestResolveNamesWhatIsSupported(t *testing.T) {
	registry := Builtin()
	codec, deviations, err := registry.Resolve("", nil)
	if err != nil || codec.ID() != DefaultSchema || len(deviations) != 0 {
		t.Fatalf("default: %v %v", codec, err)
	}
	if _, _, err := registry.Resolve("systemone/0.3", nil); err == nil || !strings.Contains(err.Error(), "systemone/0.2, vllm-sr/4.0, vllm-sr/5.0") {
		t.Fatalf("unknown schema: %v", err)
	}
	if _, _, err := registry.Resolve(SchemaSystemOne02, []string{"state_json"}); err == nil || !strings.Contains(err.Error(), "the deviations are state_text") {
		t.Fatalf("unknown deviation: %v", err)
	}
	if _, _, err := registry.Resolve(SchemaSystemOne02, []string{DeviationStateText, DeviationStateText}); err == nil || !strings.Contains(err.Error(), "twice") {
		t.Fatalf("repeated deviation: %v", err)
	}
	if _, _, err := registry.Resolve(SchemaVLLMSR50, []string{DeviationStateText}); err == nil || !strings.Contains(err.Error(), "does not apply") {
		t.Fatalf("inapplicable deviation: %v", err)
	}
}

func question(t *testing.T, text string) systemoneprotocol.Question {
	t.Helper()
	value, err := systemoneprotocol.Parse([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	q, err := systemoneprotocol.ParseQuestion(value)
	if err != nil {
		t.Fatal(err)
	}
	return q
}

// problemOf is err as the Problem it must be.
func problemOf(t *testing.T, err error) *systemoneprotocol.Problem {
	t.Helper()
	var problem *systemoneprotocol.Problem
	if !errors.As(err, &problem) {
		t.Fatalf("not a Problem: %v", err)
	}
	return problem
}

func state(t *testing.T, text string) systemoneprotocol.Value {
	t.Helper()
	value, err := systemoneprotocol.Parse([]byte(text))
	if err != nil {
		t.Fatal(err)
	}
	return value
}

func TestTheStandardCodecSplitsAndStrips(t *testing.T) {
	deadline, _ := systemoneprotocol.Number("950")
	batch := systemoneprotocol.Batch{Model: "d3", Options: systemoneprotocol.Options{DeadlineMs: deadline}, Items: []systemoneprotocol.NamedRequest{
		{Name: "1", Request: systemoneprotocol.Request{
			State: state(t, `"Our export fails with a 500 again."`),
			Questions: []systemoneprotocol.NamedQuestion{{Name: "systemone:team", Question: question(t,
				`{"type": "choice", "instructions": "Which team?", "criteria": {"billing": "Payments", "eng": null}, "overflow": "truncate"}`)}},
		}},
		{Name: "2", Request: systemoneprotocol.Request{
			State:     state(t, `{"history": "How do I export?", "request": "Again, the export."}`),
			Roles:     systemoneprotocol.Roles{Mode: systemoneprotocol.RolesExplicit, Map: []systemoneprotocol.Member{{Key: "request", Value: systemoneprotocol.String("user")}}},
			Questions: []systemoneprotocol.NamedQuestion{{Name: "preset:turn_relation", Question: question(t, `{"type": "noul", "instructions": "Is it a re-ask?", "preset": "turn_relation"}`)}},
		}},
	}}
	plan, diagnostics, err := SystemOne02Codec{}.Plan(batch, Target{Model: "d3"}, DefaultPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.Calls) != 2 {
		t.Fatalf("%d calls, want one per request", len(plan.Calls))
	}
	first, _ := json.Marshal(plan.Calls[0].Body)
	if string(first) != `{"model":"d3","state":"Our export fails with a 500 again.","questions":{"systemone:team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","eng":null}}}}` {
		t.Fatalf("first call %s", first)
	}
	second, _ := json.Marshal(plan.Calls[1].Body)
	if string(second) != `{"model":"d3","state":{"history":"How do I export?","request":"Again, the export."},"questions":{"preset:turn_relation":{"type":"noul","instructions":"Is it a re-ask?"}}}` {
		t.Fatalf("second call %s", second)
	}
	stripped := map[string]bool{}
	for _, diagnostic := range diagnostics {
		stripped[diagnostic.Field+":"+string(diagnostic.Action)] = true
	}
	for _, want := range []string{"batch:split", "options:stripped", "question.overflow:stripped", "roles:stripped", "question.preset:stripped"} {
		if !stripped[want] {
			t.Fatalf("no %s diagnostic in %+v", want, diagnostics)
		}
	}
}

func TestTheStandardCodecRefusesWhatItCannotCarry(t *testing.T) {
	for name, text := range map[string]string{
		"span":               `{"type": "span", "instructions": "Find names.", "criteria": {"PERSON": null}}`,
		"head":               `{"type": "noul", "instructions": "x", "head": "router"}`,
		"over":               `{"type": "noul", "instructions": "x", "over": "request"}`,
		"require_full_input": `{"type": "noul", "instructions": "x", "require_full_input": true}`,
		"unknown":            `{"type": "noul", "instructions": "x", "colour": "red"}`,
		"bare preset":        `{"type": "span", "preset": "pii"}`,
	} {
		t.Run(name, func(t *testing.T) {
			batch := systemoneprotocol.Batch{Model: "d3", Items: []systemoneprotocol.NamedRequest{{Name: "1", Request: systemoneprotocol.Request{
				State: state(t, `"text"`), Questions: []systemoneprotocol.NamedQuestion{{Name: "q", Question: question(t, text)}},
			}}}}
			if _, _, err := (SystemOne02Codec{}).Plan(batch, Target{}, DefaultPolicy); err == nil {
				t.Fatal("planned a question systemone/0.2 cannot carry")
			}
		})
	}
}

func TestTheStandardCodecSendsImagesAsAssets(t *testing.T) {
	batch := systemoneprotocol.Batch{Model: "d3", Items: []systemoneprotocol.NamedRequest{{Name: "1", Request: systemoneprotocol.Request{
		State:      state(t, `"What is in the picture?"`),
		Questions:  []systemoneprotocol.NamedQuestion{{Name: "q", Question: question(t, `{"type": "noul", "instructions": "Is it a cat?"}`)}},
		Extensions: []systemoneprotocol.Member{{Key: "images", Value: systemoneprotocol.Array(systemoneprotocol.String("data:image/png;base64,iVBORw0KGgo="))}},
	}}}}
	plan, _, err := SystemOne02Codec{}.Plan(batch, Target{}, DefaultPolicy)
	if err != nil {
		t.Fatal(err)
	}
	body, _ := json.Marshal(plan.Calls[0].Body)
	if !strings.Contains(string(body), `"state":["⟦vsr-image-1⟧","What is in the picture?"]`) ||
		!strings.Contains(string(body), `"assets":{"⟦vsr-image-1⟧":{"type":"image","format":"png","data_base64":"iVBORw0KGgo="}}`) {
		t.Fatalf("body %s", body)
	}
	if plan.Calls[0].Header.Get(headerImages) != "true" {
		t.Fatal("a request with assets sends the images header")
	}
}

func TestTheStandardCodecReadsItsReplies(t *testing.T) {
	codec := SystemOne02Codec{}
	plan := Plan{Names: []string{"1"}}
	call := WireCall{Items: []int{0}}
	results, _ := codec.DecodeReply(plan, call, WireReply{Status: 200, Body: []byte(`{"model":"d3","answers":{"q":{"type":"noul","noul":0.9}},"usage":{"input_tokens":12,"output_tokens":0}}`)})
	if results[0].Err != nil || results[0].Response.Usage.InputTokens != 12 || results[0].Response.Answers[0].Key != "q" {
		t.Fatalf("200: %+v", results[0])
	}
	results, _ = codec.DecodeReply(plan, call, WireReply{Status: 422, Body: []byte(`{"detail":[{"loc":["body","questions","q","noul","criteria"],"msg":"bad criteria","type":"value_error"}]}`)})
	problem := problemOf(t, results[0].Err)
	if problem.Question != "q" || problem.QuestionError != "invalid_question" || problem.Kind != systemoneprotocol.ProblemRejected {
		t.Fatalf("422: %+v", problem)
	}
	results, _ = codec.DecodeReply(plan, call, WireReply{Status: 400, Body: []byte(`{"detail":"state must be a string"}`)})
	if problem := problemOf(t, results[0].Err); problem.Detail != "state must be a string" || problem.Item != "1" {
		t.Fatalf("400: %+v", problem)
	}
	results, _ = codec.DecodeReply(plan, call, WireReply{Status: 429, Header: http.Header{"Retry-After": {"2"}}, Body: []byte(`{"detail":"slow down"}`)})
	if problem := problemOf(t, results[0].Err); problem.Kind != systemoneprotocol.ProblemOverloaded || problem.RetryAfter != 2*time.Second {
		t.Fatalf("429: %+v", problem)
	}
}

func TestStateTextSendsTheCanonicalText(t *testing.T) {
	call := WireCall{Body: systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String("d3")},
		systemoneprotocol.Member{Key: "state", Value: state(t, `{"previous_turns": "user: 导出\nassistant: <ok>", "request": "again \u2028"}`)},
	)}
	if _, err := (StateTextDeviation{}).RewriteRequest(&call); err != nil {
		t.Fatal(err)
	}
	text, _ := mustGet(call.Body, "state").Text()
	if text != "{\"previous_turns\": \"user: 导出\\nassistant: <ok>\", \"request\": \"again \u2028\"}" {
		t.Fatalf("state text %q", text)
	}
	plain := WireCall{Body: systemoneprotocol.Object(systemoneprotocol.Member{Key: "state", Value: systemoneprotocol.String("as is")})}
	if diagnostics, _ := (StateTextDeviation{}).RewriteRequest(&plain); diagnostics != nil {
		t.Fatal("a text state is left as it is")
	}
}

func TestTheInternalCodecReadsAndWritesTheNativeForms(t *testing.T) {
	codec := InternalCodec{}
	body := []byte(`{"model":"vllm-sr/d3","states":{"a":{"state":"one","questions":{"q":{"type":"choice","choices":[{"key":"x","description":"X"},{"key":"y"}]}}},"b":{"state":"two","questions":{"s":{"type":"score","levels":["low","high"]}}}},"options":{"deadline_ms":500,"long_context":true}}`)
	batch, err := codec.DecodeRequest(body)
	if err != nil || batch.Form != systemoneprotocol.FormStates || len(batch.Items) != 2 || !batch.Options.LongContext {
		t.Fatalf("decode: %+v %v", batch, err)
	}
	if deadline, _ := batch.Options.Deadline(); deadline != 500 {
		t.Fatalf("deadline %v", deadline)
	}
	choice, err := batch.Items[0].Request.Questions[0].Question.Standard()
	if err != nil || systemoneprotocol.Canonical(choice.Criteria) != `{"x": "X", "y": null}` {
		t.Fatalf("choices become %s (%v)", systemoneprotocol.Canonical(choice.Criteria), err)
	}
	score, _ := batch.Items[1].Request.Questions[0].Question.Standard()
	if got := systemoneprotocol.Canonical(score.Criteria); got != `["low", "high"]` {
		t.Fatalf("levels become %s", got)
	}
	encoded, err := codec.EncodeRequest(batch)
	if err != nil || string(encoded) != string(body) {
		t.Fatalf("the internal form does not write back as it came:\n%s\n%s (%v)", encoded, body, err)
	}
	status, answer, err := codec.EncodeResults([]ItemResult{
		{Response: systemoneprotocol.Response{Model: "m", Answers: []systemoneprotocol.Member{{Key: "q", Value: state(t, `{"type":"choice","choice":"x"}`)}}, Usage: systemoneprotocol.Usage{InputTokens: 3}}},
		{Response: systemoneprotocol.Response{Model: "m", Answers: []systemoneprotocol.Member{{Key: "s", Value: state(t, `{"type":"score","score":1}`)}}, Usage: systemoneprotocol.Usage{InputTokens: 4}}},
	}, []string{"a", "b"}, batch.Form, "vllm-sr/d3")
	if err != nil || status != 200 || string(answer) != `{"model":"vllm-sr/d3","states":{"a":{"model":"m","answers":{"q":{"type":"choice","choice":"x"}},"usage":{"input_tokens":3,"output_tokens":0}},"b":{"model":"m","answers":{"s":{"type":"score","score":1}},"usage":{"input_tokens":4,"output_tokens":0}}},"usage":{"input_tokens":7,"output_tokens":0}}` {
		t.Fatalf("encode %d %s %v", status, answer, err)
	}
	status, answer = NativeErrorBody(&systemoneprotocol.Problem{Status: 422, Kind: systemoneprotocol.ProblemRejected, Question: "q", Detail: "bad"}, "a")
	if status != 422 || !strings.Contains(string(answer), `"code":"question_failed"`) || !strings.Contains(string(answer), `"state":"a"`) {
		t.Fatalf("question failure %d %s", status, answer)
	}
}
