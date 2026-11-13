package systemonecodec

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// canonicalBytes is a JSON document as Value writes it, the form a lossless
// codec must give back.
func canonicalBytes(t testing.TB, body []byte) []byte {
	t.Helper()
	value, err := systemoneprotocol.Parse(body)
	if err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(value)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func internalCorpus(t testing.TB) []struct{ Name, Body string } {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("testdata", "internal_corpus.json"))
	if err != nil {
		t.Fatal(err)
	}
	var corpus struct {
		Cases []struct{ Name, Body string } `json:"cases"`
	}
	if err := json.Unmarshal(data, &corpus); err != nil {
		t.Fatal(err)
	}
	if len(corpus.Cases) < 8 {
		t.Fatalf("the corpus holds %d cases", len(corpus.Cases))
	}
	return corpus.Cases
}

// The internal form is the IR's encoding: every golden request, the families'
// self-test requests among them, reads into the IR and writes back as it came.
func TestTheInternalFormRoundTripsThroughTheIR(t *testing.T) {
	for _, item := range internalCorpus(t) {
		t.Run(item.Name, func(t *testing.T) {
			batch, err := InternalCodec{}.DecodeRequest([]byte(item.Body))
			if err != nil {
				t.Fatalf("decode: %v", err)
			}
			encoded, err := InternalCodec{}.EncodeRequest(batch)
			if err != nil {
				t.Fatal(err)
			}
			if want := canonicalBytes(t, []byte(item.Body)); string(encoded) != string(want) {
				t.Fatalf("round trip changed the request:\n got %s\nwant %s", encoded, want)
			}
		})
	}
}

func FuzzInternalRoundTrip(f *testing.F) {
	for _, item := range internalCorpus(f) {
		f.Add([]byte(item.Body))
	}
	f.Fuzz(func(t *testing.T, body []byte) {
		batch, err := InternalCodec{}.DecodeRequest(body)
		if err != nil {
			return
		}
		encoded, err := InternalCodec{}.EncodeRequest(batch)
		if err != nil {
			t.Fatalf("a request the codec read does not write: %v", err)
		}
		value, _ := systemoneprotocol.Parse(body)
		want, _ := json.Marshal(value)
		if string(encoded) != string(want) {
			t.Fatalf("round trip changed\n%s\ninto\n%s", want, encoded)
		}
	})
}

func question50(t *testing.T, text string) systemoneprotocol.Question {
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

func TestTheFiveCodecSendsStandardRequestsWithFlatExtensions(t *testing.T) {
	presets := func(name string) (systemoneprotocol.Question, bool) {
		if name != "pii" {
			return systemoneprotocol.Question{}, false
		}
		return systemoneprotocol.Question{
			Type: "span", Instructions: systemoneprotocol.String("Find personal data."),
			Criteria: systemoneprotocol.Object(systemoneprotocol.Member{Key: "EMAIL_ADDRESS", Value: systemoneprotocol.String("An email address")}),
		}, true
	}
	deadline, _ := systemoneprotocol.Number("1500")
	batch := systemoneprotocol.Batch{Model: "d3", Options: systemoneprotocol.Options{DeadlineMs: deadline, LongContext: true}, Items: []systemoneprotocol.NamedRequest{{Name: "1", Request: systemoneprotocol.Request{
		State: state(t, `{"request": "Mail me at a@example.com", "previous_answer": "Sure."}`),
		Questions: []systemoneprotocol.NamedQuestion{
			{Name: "team", Question: question50(t, `{"type": "choice", "instructions": "Which team?", "choices": [{"key": "billing", "description": "Payments"}, {"key": "eng"}]}`)},
			{Name: "pii", Question: question50(t, `{"preset": "pii", "threshold": 0.4}`)},
		},
		Extensions: []systemoneprotocol.Member{{Key: "images", Value: systemoneprotocol.Array(systemoneprotocol.String("data:image/png;base64,iVBORw0KGgo="))}},
	}}}}
	plan, _, err := VLLMSR50Codec{}.Plan(batch, Target{}, Policy{StripUnsupported: true, Presets: presets})
	if err != nil {
		t.Fatal(err)
	}
	call := plan.Calls[0]
	body, _ := json.Marshal(call.Body)
	want := `{"model":"d3","state":["⟦vsr-image-1⟧",{"request":"Mail me at a@example.com","previous_answer":"Sure."}],` +
		`"questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"Payments","eng":null}},` +
		`"pii":{"type":"span","instructions":"Find personal data.","criteria":{"EMAIL_ADDRESS":"An email address"},"preset":"pii","threshold":0.4}},` +
		`"assets":{"⟦vsr-image-1⟧":{"type":"image","format":"png","data_base64":"iVBORw0KGgo="}},` +
		`"roles":{"request":"user","previous_answer":"context"},"options":{"deadline_ms":1500}}`
	if string(body) != want || call.Path != "" || call.Header.Get(HeaderLongContext) != "true" {
		t.Fatalf("request %s path %q header %v\nwant %s", body, call.Path, call.Header, want)
	}
	two := batch
	two.Items = append(two.Items, systemoneprotocol.NamedRequest{Name: "2", Request: systemoneprotocol.Request{
		State: state(t, `"plain"`), Questions: []systemoneprotocol.NamedQuestion{{Name: "q", Question: question50(t, `{"type": "noul", "instructions": "Ok?"}`)}},
	}})
	plan, _, err = VLLMSR50Codec{}.Plan(two, Target{}, Policy{StripUnsupported: true, Presets: presets})
	if err != nil || len(plan.Calls) != 1 || !plan.Calls[0].Batch || plan.Calls[0].Path != BatchPath {
		t.Fatalf("a batch plans %+v (%v)", plan, err)
	}
	if requests, ok := plan.Calls[0].Body.Get("requests"); !ok || len(requests.Members()) != 2 {
		t.Fatalf("batch body %v", plan.Calls[0].Body)
	}
	_, _, err = (VLLMSR50Codec{}).Plan(systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: systemoneprotocol.Request{
		State: state(t, `"x"`), Questions: []systemoneprotocol.NamedQuestion{{Name: "q", Question: question50(t, `{"preset": "unknown"}`)}},
	}}}}, Target{}, Policy{Presets: presets})
	var problem *systemoneprotocol.Problem
	if !errors.As(err, &problem) || problem.Question != "q" || problem.Reason != "preset_unknown" {
		t.Fatalf("a preset the Router has no question for fails that question, not the request: %v", err)
	}
	if status, body := NativeErrorBody(problem, ""); status != http.StatusUnprocessableEntity || !strings.Contains(string(body), `"reason":"preset_unknown"`) {
		t.Fatalf("an unknown preset answers %d %s", status, body)
	}
}

// The roles the 5.0 codec writes for a 4.0 request are the ones the model
// runtime's implicit roles give its keys, whatever their case: the cases
// config/schemas/systemone/implicit-roles.json shares with the runtime's tests.
func TestImplicitRolesMatchTheRuntimesForEveryCase(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "schemas", "systemone", "implicit-roles.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			State json.RawMessage `json:"state"`
			Roles json.RawMessage `json:"roles"`
		} `json:"cases"`
	}
	if err := json.Unmarshal(data, &fixture); err != nil || len(fixture.Cases) == 0 {
		t.Fatalf("fixture: %v", err)
	}
	for _, tc := range fixture.Cases {
		object := state(t, string(tc.State))
		want := canonicalBytes(t, tc.Roles)
		if got, _ := json.Marshal(systemoneprotocol.Object(systemoneprotocol.ImplicitRoles(object)...)); string(got) != string(want) {
			t.Fatalf("implicit roles of %s: %s, want %s", tc.State, got, want)
		}
		roles, err := systemoneprotocol.Parse(tc.Roles)
		if err != nil {
			t.Fatal(err)
		}
		if !(systemoneprotocol.Roles{Mode: systemoneprotocol.RolesExplicit, Map: roles.Members()}).SameAsImplicit(object) {
			t.Fatalf("roles %s are not the implicit ones of %s", tc.Roles, tc.State)
		}
		plan, _, err := VLLMSR50Codec{}.Plan(systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: systemoneprotocol.Request{
			State: object, Questions: []systemoneprotocol.NamedQuestion{{Name: "q", Question: question50(t, `{"type": "noul", "instructions": "Ok?"}`)}},
		}}}}, Target{}, Policy{})
		if err != nil {
			t.Fatal(err)
		}
		sent, _ := plan.Calls[0].Body.Get("roles")
		if got, _ := json.Marshal(sent); string(got) != string(want) {
			t.Fatalf("the 5.0 codec sent roles %s for %s, want %s", got, tc.State, want)
		}
	}
}

// A question the server fails by name is asked again without it and answers
// with its error and reason, as a runtime's /v1/decisions answers it.
func TestAFailedQuestionAnswersWithItsReason(t *testing.T) {
	calls := 0
	exchange := Exchange{Codec: VLLMSR50Codec{}, Send: func(_ context.Context, _ WireCall, body []byte) (WireReply, error) {
		calls++
		if strings.Contains(string(body), `"bad"`) {
			return WireReply{Status: http.StatusUnprocessableEntity, Body: []byte(`{"detail":[{"loc":["body","questions","bad"],"msg":"no question is valid","type":"invalid_question"}],"code":"question_failed","question":"bad","reason":"question_malformed"}`)}, nil
		}
		return WireReply{Status: http.StatusOK, Body: []byte(`{"model":"d3","answers":{"ok":{"type":"noul","noul":0.25}},"usage":{"input_tokens":3,"output_tokens":0}}`)}, nil
	}}
	request := systemoneprotocol.Request{State: state(t, `"x"`), Questions: []systemoneprotocol.NamedQuestion{
		{Name: "ok", Question: question50(t, `{"type": "noul", "instructions": "Ok?"}`)},
		{Name: "bad", Question: question50(t, `{"type": "choice", "instructions": "Which?", "criteria": {"only": "one"}}`)},
	}}
	results := exchange.Ask(context.Background(), systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: request}}}, nil)
	if results[0].Err != nil || calls != 2 {
		t.Fatalf("calls %d, %v", calls, results[0].Err)
	}
	answers, _ := json.Marshal(systemoneprotocol.Object(results[0].Response.Answers...))
	if want := `{"ok":{"type":"noul","noul":0.25},"bad":{"type":"choice","error":"invalid_question","reason":"question_malformed"}}`; string(answers) != want {
		t.Fatalf("answers %s\nwant %s", answers, want)
	}
	calls = 0
	request.Questions = []systemoneprotocol.NamedQuestion{
		{Name: "ok", Question: question50(t, `{"type": "noul", "instructions": "Ok?"}`)},
		{Name: "p", Question: question50(t, `{"preset": "no_such_preset"}`)},
	}
	results = exchange.Ask(context.Background(), systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: request}}}, nil)
	answers, _ = json.Marshal(systemoneprotocol.Object(results[0].Response.Answers...))
	if want := `{"ok":{"type":"noul","noul":0.25},"p":{"type":null,"error":"invalid_question","reason":"preset_unknown"}}`; results[0].Err != nil || calls != 1 || string(answers) != want {
		t.Fatalf("calls %d, %v, answers %s\nwant %s", calls, results[0].Err, answers, want)
	}
}

// A request none of whose questions is valid is refused as the runtime
// refuses it on /v1/decisions: 400 invalid_request naming its first question,
// with every question's reason. A batch is refused so only when none of its
// entries has one; otherwise each entry answers, its questions failed.
func TestARequestWithNoValidQuestionIsRefusedAsTheRuntimeRefusesIt(t *testing.T) {
	exchange := Exchange{Codec: VLLMSR50Codec{}, Send: func(_ context.Context, _ WireCall, body []byte) (WireReply, error) {
		for _, bad := range []string{"bad1", "bad2"} {
			if strings.Contains(string(body), `"`+bad+`"`) {
				return WireReply{Status: http.StatusUnprocessableEntity, Body: []byte(`{"detail":[{"loc":["body","questions","` + bad + `"],"msg":"bad","type":"invalid_question"}],"code":"question_failed","question":"` + bad + `","reason":"question_malformed"}`)}, nil
			}
		}
		return WireReply{Status: http.StatusOK, Body: []byte(`{"model":"d3","answers":{"ok":{"type":"noul","noul":0.25}},"usage":{"input_tokens":3,"output_tokens":0}}`)}, nil
	}}
	invalid := systemoneprotocol.Request{State: state(t, `"x"`), Questions: []systemoneprotocol.NamedQuestion{
		{Name: "bad1", Question: question50(t, `{"type": "choice", "instructions": "Which?", "criteria": {"only": "one"}}`)},
		{Name: "bad2", Question: question50(t, `{"type": "choice", "instructions": "Which?", "criteria": {"only": "one"}}`)},
	}}
	valid := systemoneprotocol.Request{State: state(t, `"y"`), Questions: []systemoneprotocol.NamedQuestion{
		{Name: "ok", Question: question50(t, `{"type": "noul", "instructions": "Ok?"}`)},
	}}
	ask := func(form systemoneprotocol.BatchForm, items ...systemoneprotocol.NamedRequest) (int, string) {
		t.Helper()
		names := make([]string, len(items))
		for index, item := range items {
			names[index] = item.Name
		}
		results := exchange.Ask(context.Background(), systemoneprotocol.Batch{Form: form, Items: items}, nil)
		status, body, err := InternalCodec{}.EncodeResults(results, names, form, "vllm-sr/test")
		if err != nil {
			t.Fatal(err)
		}
		return status, string(body)
	}
	status, body := ask(systemoneprotocol.FormAlone, systemoneprotocol.NamedRequest{Request: invalid})
	if want := `{"error":{"code":"invalid_request","message":"no question is valid (bad1: question_malformed; bad2: question_malformed)","question":"bad1","reason":"question_malformed"}}`; status != 400 || body != want {
		t.Fatalf("a request without a valid question => %d %s\nwant 400 %s", status, body, want)
	}
	status, body = ask(systemoneprotocol.FormStates, systemoneprotocol.NamedRequest{Name: "a", Request: invalid}, systemoneprotocol.NamedRequest{Name: "b", Request: valid})
	if want := `"a":{"model":"vllm-sr/test","answers":{"bad1":{"type":"choice","error":"invalid_question","reason":"question_malformed"},"bad2":{"type":"choice","error":"invalid_question","reason":"question_malformed"}}`; status != 200 || !strings.Contains(body, want) || !strings.Contains(body, `"b":{"model":"d3","answers":{"ok"`) {
		t.Fatalf("a batch with a valid entry => %d %s", status, body)
	}
	status, body = ask(systemoneprotocol.FormStates, systemoneprotocol.NamedRequest{Name: "a", Request: invalid}, systemoneprotocol.NamedRequest{Name: "b", Request: invalid})
	if want := `{"error":{"code":"invalid_request","message":"no question is valid (states.a.questions.bad1: question_malformed; states.a.questions.bad2: question_malformed; states.b.questions.bad1: question_malformed; states.b.questions.bad2: question_malformed)","question":"bad1","reason":"question_malformed","state":"a"}}`; status != 400 || body != want {
		t.Fatalf("a batch without a valid question => %d %s\nwant 400 %s", status, body, want)
	}
}

// A medium a codec cannot carry is refused before any call, as the runtime
// refuses it: invalid_request with the medium's format reason.
func TestMediaACodecCannotCarryAreRefusedByFormat(t *testing.T) {
	called := false
	exchange := Exchange{Codec: VLLMSR50Codec{}, Send: func(context.Context, WireCall, []byte) (WireReply, error) {
		called = true
		return WireReply{}, nil
	}}
	request := systemoneprotocol.Request{
		State:      state(t, `"Here is a picture."`),
		Questions:  []systemoneprotocol.NamedQuestion{{Name: "q", Question: question50(t, `{"type": "noul", "instructions": "Ok?"}`)}},
		Extensions: []systemoneprotocol.Member{{Key: "images", Value: systemoneprotocol.Array(systemoneprotocol.String("iVBORw0KGgo="))}},
	}
	results := exchange.Ask(context.Background(), systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: request}}}, nil)
	var problem *systemoneprotocol.Problem
	if called || !errors.As(results[0].Err, &problem) || problem.Code != "invalid_request" || problem.Reason != "image_format" {
		t.Fatalf("an image that is no data URL: called %v, %v", called, results[0].Err)
	}
	if status, body := NativeErrorBody(problem, ""); status != http.StatusBadRequest || !strings.Contains(string(body), `"reason":"image_format"`) {
		t.Fatalf("answers %d %s", status, body)
	}
}

func TestTheFiveCodecReadsItsReplies(t *testing.T) {
	codec := VLLMSR50Codec{}
	plan := Plan{Names: []string{"a", "b"}}
	call := WireCall{Items: []int{0, 1}, Batch: true}
	results, err := codec.DecodeReply(plan, call, WireReply{Status: 200, Body: []byte(`{"model":"d3","responses":{"a":{"model":"d3","answers":{"q":{"type":"noul","noul":0.9}},"usage":{"input_tokens":3,"output_tokens":0}},"b":{"answers":{"q":{"type":"noul","noul":0.1}},"usage":{"input_tokens":4,"output_tokens":0}}},"usage":{"input_tokens":7,"output_tokens":0}}`)})
	if err != nil || results[1].Err != nil || results[1].Response.Model != "d3" || results[0].Response.Usage.InputTokens != 3 {
		t.Fatalf("batch reply %+v %v", results, err)
	}
	results, _ = codec.DecodeReply(plan, call, WireReply{Status: 400, Body: []byte(`{"detail":"Question 'pii' exceeds the scan budget.","code":"question_failed","question":"pii","state":"b","reason":"scan_budget"}`)})
	problem := problemOf(t, results[0].Err)
	if problem.Question != "pii" || problem.Item != "b" || problem.QuestionError != "scan_budget_exceeded" || problem.Code != "question_failed" {
		t.Fatalf("flat error %+v", problem)
	}
	results, _ = codec.DecodeReply(Plan{Names: []string{"1"}}, WireCall{Items: []int{0}}, WireReply{Status: 529, Header: http.Header{"Retry-After": {"1"}}, Body: []byte(`{"detail":"busy","code":"overloaded"}`)})
	if problem := problemOf(t, results[0].Err); problem.Kind != systemoneprotocol.ProblemOverloaded || problem.RetryAfter == 0 {
		t.Fatalf("529 %+v", problem)
	}
}

func readFive(t *testing.T, path, body string, header http.Header) ClientRequest {
	t.Helper()
	read, err := Listener50{}.Read(path, []byte(body), header)
	if err != nil {
		t.Fatalf("%s %s: %v", path, body, err)
	}
	return read
}

func TestAFiveListenerReadsStrictlyAndTranslates(t *testing.T) {
	read := readFive(t, PathSystemOne, `{"model":"d3","state":{"context":"Refunds within 30 days.","answer":"You have 90 days."},"questions":{"q":{"type":"noul","instructions":"Supported?"}}}`, http.Header{})
	if read.Plan != PlanTranslate || string(read.Internal) != `{"model":"d3","state":{"context":"Refunds within 30 days.","answer":"You have 90 days."},"questions":{"q":{"type":"noul","instructions":"Supported?"}},"roles":null}` {
		t.Fatalf("an object state without roles: %s (%s)", read.Internal, read.Plan)
	}
	longContext := http.Header{}
	longContext.Set(HeaderLongContext, "true")
	read = readFive(t, PathSystemOne, `{"model":"d3","state":["img","What is it?"],"questions":{"q":{"type":"noul","instructions":"A cat?"}},"assets":{"img":{"type":"image","format":"jpeg","data_base64":"/9j/"}}}`, longContext)
	if string(read.Internal) != `{"model":"d3","state":"What is it?","questions":{"q":{"type":"noul","instructions":"A cat?"}},"images":["data:image/jpeg;base64,/9j/"],"options":{"long_context":true}}` {
		t.Fatalf("media in front of the state: %s", read.Internal)
	}
	read = readFive(t, PathBatch, `{"model":"d3","requests":{"a":{"state":"one","questions":{"q":{"type":"noul"}}},"b":{"model":"d3","state":{"request":"two"},"roles":{"request":"user"},"questions":{"q":{"type":"noul"}}}},"options":{"deadline_ms":800}}`, http.Header{})
	if string(read.Internal) != `{"model":"d3","states":{"a":{"state":"one","questions":{"q":{"type":"noul"}}},"b":{"state":{"request":"two"},"questions":{"q":{"type":"noul"}},"roles":{"request":"user"}}},"options":{"deadline_ms":800}}` {
		t.Fatalf("batch: %s", read.Internal)
	}
	for body, want := range map[string]string{
		`{"model":"d3","state":"x","questions":{"q":{"type":"choice","choices":[{"key":"a"}]}}}`:               "choices is not part of vllm-sr/5.0; use criteria",
		`{"model":"d3","state":"x","images":["data:image/png;base64,AA=="],"questions":{"q":{"type":"noul"}}}`: "images is not part of vllm-sr/5.0; use assets",
		`{"model":"d3","states":{"a":{"state":"x","questions":{"q":{"type":"noul"}}}}}`:                        "states is not part of vllm-sr/5.0; use POST /v1/systemone/batch",
		`{"model":"d3","state":"x","questions":{"q":{"type":"noul"}},"options":{"long_context":true}}`:         "send the request header TypeSafe-Danger-Zone-Long-Context: true",
		`{"model":"d3","state":"x","questions":{"q":{"type":"noul","colour":"red"}}}`:                          "colour is not a field of a vllm-sr/5.0 question",
		`{"model":"d3","state":"x","questions":{"q":{"preset":"pii"}}}`:                                        "type is required",
		`{"model":"d3","state":"x","roles":{"x":"user"},"questions":{"q":{"type":"noul"}}}`:                    "roles name the keys of an object state",
	} {
		if _, err := (Listener50{}).Read(PathSystemOne, []byte(body), http.Header{}); err == nil || !strings.Contains(err.Error(), want) {
			t.Errorf("%s: %v, want %q", body, err, want)
		}
	}
	native := readFive(t, PathDecisions, `{"model":"d3","state":{"request":"x"},"questions":{"q":{"type":"choice","choices":[{"key":"a"},{"key":"b"}]}}}`, http.Header{})
	if native.Plan != PlanPassThrough || strings.Join(native.Legacy, ",") != "choices,implicit_roles" {
		t.Fatalf("/v1/decisions is native: %+v", native)
	}
}

// A vllm-sr/4.0 listener keeps 4.0's implicit roles where a 5.0 listener
// gives none: the same request reaches the runtime with no roles member and
// with roles: null, which a role-aware family reads differently.
func TestImplicitRolesDifferBetweenTheListeners(t *testing.T) {
	body := `{"model":"vela","state":{"context":"Refunds within 30 days.","answer":"You have 90 days."},"questions":{"q":{"type":"noul","instructions":"Supported?"}}}`
	four, err := Listener40{}.Read(PathSystemOne, []byte(body), http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	five := readFive(t, PathSystemOne, body, http.Header{})
	if strings.Contains(string(four.Internal), `"roles"`) || four.Plan != PlanPassThrough || strings.Join(four.Legacy, ",") != "implicit_roles" {
		t.Fatalf("4.0 listener: %s %+v", four.Internal, four)
	}
	if !strings.HasSuffix(string(five.Internal), `"roles":null}`) {
		t.Fatalf("5.0 listener: %s", five.Internal)
	}
}

// Media placed in front of a state is the other valid 5.0 request a 4.0
// listener reads otherwise: it keeps the array state and its assets, and
// counts the form, where a 5.0 listener reads the item with the image in
// front of it.
func TestMediaPlacedInFrontDifferBetweenTheListeners(t *testing.T) {
	body := `{"model":"vela","state":["logo","Where is the logo?"],"assets":{"logo":{"type":"image","format":"png","data_base64":"iVBORw0KGgo="}},"questions":{"q":{"type":"noul","instructions":"Is there a logo?"}}}`
	four, err := Listener40{}.Read(PathSystemOne, []byte(body), http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(four.Internal), `"state":["logo","Where is the logo?"]`) || strings.Join(four.Legacy, ",") != "placed_assets" {
		t.Fatalf("4.0 listener: %s %+v", four.Internal, four.Legacy)
	}
	five := readFive(t, PathSystemOne, body, http.Header{})
	if !strings.Contains(string(five.Internal), `"state":"Where is the logo?"`) || !strings.Contains(string(five.Internal), `"images":["data:image/png;base64,iVBORw0KGgo="]`) {
		t.Fatalf("5.0 listener: %s", five.Internal)
	}
}

func TestAFourListenerPassesLegacyRequestsThroughAsAPlan(t *testing.T) {
	body := `{"model": "d3", "state": "x", "questions": {"q": {"type": "choice", "choices": [{"key": "a"}, {"key": "b"}]}}, "images": []}`
	read, err := Listener40{}.Read(PathSystemOne, []byte(body), http.Header{})
	if err != nil {
		t.Fatal(err)
	}
	if read.Plan != PlanPassThrough || string(read.Internal) != body || strings.Join(read.Legacy, ",") != "images,choices" {
		t.Fatalf("legacy request: %+v", read)
	}
	actions := map[string]bool{}
	for _, diagnostic := range read.Diagnostics {
		actions[diagnostic.Field+":"+string(diagnostic.Action)] = true
	}
	if !actions["images:legacy_form"] || !actions["choices:legacy_form"] || !actions["request:passed_through"] {
		t.Fatalf("diagnostics %+v", read.Diagnostics)
	}
	standard, err := Listener40{}.Read(PathSystemOne, []byte(`{"model": "d3", "state": "x", "questions": {"q": {"type": "noul", "instructions": "Ok?"}}}`), http.Header{})
	if err != nil || standard.Plan != PlanPassThrough || len(standard.Legacy) != 0 || string(standard.Internal) != `{"model": "d3", "state": "x", "questions": {"q": {"type": "noul", "instructions": "Ok?"}}}` {
		t.Fatalf("standard request: %+v %v", standard, err)
	}
	longContext := http.Header{}
	longContext.Set(HeaderLongContext, "true")
	header, err := Listener40{}.Read(PathSystemOne, []byte(`{"model": "d3", "state": "x", "questions": {"q": {"type": "noul"}}}`), longContext)
	if err != nil || header.Plan != PlanTranslate || !strings.Contains(string(header.Internal), `"options":{"long_context":true}`) {
		t.Fatalf("long-context header: %+v %v", header, err)
	}
	if _, readErr := (Listener40{}).Read(PathSystemOne, []byte(`{"model":"d3","state":"x","questions":{"q":{"type":"noul"}},"colour":"red"}`), http.Header{}); readErr == nil {
		t.Fatal("a 4.0 listener checks the request before it passes it on")
	}
	batch, err := Listener40{}.Read(PathBatch, []byte(`{"model":"d3","requests":{"a":{"state":{"request":"x"},"questions":{"q":{"type":"noul"}}}}}`), http.Header{})
	if err != nil || strings.Contains(string(batch.Internal), "roles") || batch.Batch.Form != systemoneprotocol.FormRequests {
		t.Fatalf("a 4.0 listener's batch keeps implicit roles: %s %v", batch.Internal, err)
	}
}

func TestListenersWriteAnswersInTheirClientsSchema(t *testing.T) {
	native := []byte(`{"error":{"code":"question_failed","type":"systemone_error","message":"question \"pii\" cannot be answered: the input is longer than its scan budget","reason":"scan_budget","question":"pii","state":"b","question_error":"scan_budget_exceeded"}}`)
	status, body := Listener40{}.AnswerError(nil, PathSystemOne, 400, native)
	var four map[string]any
	if json.Unmarshal(body, &four) != nil || status != 400 || four["error"] == nil || four["detail"] == nil || four["code"] != "question_failed" || four["question"] != "pii" {
		t.Fatalf("4.0 error %d %s", status, body)
	}
	status, body = Listener50{}.AnswerError(nil, PathSystemOne, 400, native)
	if status != 400 || string(body) != `{"detail":"question \"pii\" cannot be answered: the input is longer than its scan budget","code":"question_failed","question":"pii","state":"b","reason":"scan_budget"}` {
		t.Fatalf("5.0 error %d %s", status, body)
	}
	status, body = Listener50{}.AnswerError(nil, PathSystemOne, 422, []byte(`{"error":{"code":"question_failed","message":"bad","question":"q","question_error":"invalid_question","reason":"question_malformed"}}`))
	if status != 422 || !strings.HasPrefix(string(body), `{"detail":[{"loc":["body","questions","q"],"msg":"bad","type":"invalid_question"}]`) {
		t.Fatalf("5.0 422 %d %s", status, body)
	}
	if status, _ = (Listener50{}).AnswerError(nil, PathSystemOne, 429, []byte(`{"error":{"code":"overloaded","message":"busy"}}`)); status != StatusOverloaded {
		t.Fatalf("an overloaded 5.0 server answers %d", status)
	}
	read := readFive(t, PathBatch, `{"model":"d3","requests":{"a":{"state":"one","questions":{"q":{"type":"noul"}}}}}`, http.Header{})
	answer, err := Listener50{}.AnswerSuccess(&read, []byte(`{"model":"d3","states":{"a":{"model":"d3","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":2,"output_tokens":0}}},"usage":{"input_tokens":2,"output_tokens":0}}`))
	if err != nil || string(answer) != `{"model":"d3","responses":{"a":{"model":"d3","answers":{"q":{"type":"noul","noul":0.5}},"usage":{"input_tokens":2,"output_tokens":0}}},"usage":{"input_tokens":2,"output_tokens":0}}` {
		t.Fatalf("batch answer %s %v", answer, err)
	}
	listing, err := Listener50{}.AnswerListing([]byte(`{"object":"list","data":[{"id":"d3","object":"model","api":"systemone","routing":false,"question_types":["choice","noul"]}]}`))
	if err != nil || string(listing) != `{"models":[{"name":"d3","description":"System One decision model","release_date":"","version":"5.0","features":["choice","noul"],"card":{"id":"d3","object":"model","api":"systemone","routing":false,"question_types":["choice","noul"]}}]}` {
		t.Fatalf("listing %s %v", listing, err)
	}
}

func TestMediaLeadingTheStateRoundTrips(t *testing.T) {
	state, assets, err := placeMedia(systemoneprotocol.String("Describe it."), []string{"data:image/png;base64,iVBORw0KGgo="}, []string{"data:video/mp4;base64,AAAA"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	unplaced, images, videos, rest, ok, err := UnplaceMedia(state, assets)
	if err != nil || !ok || len(rest) != 0 || len(images) != 1 || len(videos) != 1 {
		t.Fatalf("unplace %v %v %v %v %v", unplaced, images, videos, rest, err)
	}
	if text, _ := unplaced.Text(); text != "Describe it." || images[0] != "data:image/png;base64,iVBORw0KGgo=" || videos[0] != "data:video/mp4;base64,AAAA" {
		t.Fatalf("unplaced %v %v %v", unplaced, images, videos)
	}
	inline := systemoneprotocol.Array(systemoneprotocol.String("look"), systemoneprotocol.String("⟦vsr-image-1⟧"))
	if _, _, _, _, ok, _ := UnplaceMedia(inline, assets[:1]); ok {
		t.Fatal("an asset inside the state stays where it is")
	}
}

func TestQuestionErrorsFollowTheirReasons(t *testing.T) {
	for reason, want := range map[string]string{
		"scan_budget": "scan_budget_exceeded", "question_tokens": "max_length_exceeded", "input_malformed": "invalid_input",
		"model_output": "invalid_model_output", "deadline": "deadline_exceeded", "model_unavailable": "unavailable", "preset_unknown": "invalid_question",
	} {
		if got := QuestionError(reason, 400); got != want {
			t.Errorf("%s: %s, want %s", reason, got, want)
		}
	}
}

func TestTheLegacyCodecRefusesRolesItCannotCarry(t *testing.T) {
	request := systemoneprotocol.Request{State: state(t, `{"request": "x", "history": "y"}`), Questions: []systemoneprotocol.NamedQuestion{{Name: "q", Question: question50(t, `{"type": "noul"}`)}}}
	request.Roles = systemoneprotocol.Roles{Mode: systemoneprotocol.RolesExplicit, Map: systemoneprotocol.ImplicitRoles(request.State)}
	plan, _, err := VLLMSR40Codec{}.Plan(systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: request}}}, Target{Model: "m"}, DefaultPolicy)
	if err != nil {
		t.Fatal(err)
	}
	if body, _ := json.Marshal(plan.Calls[0].Body); strings.Contains(string(body), "roles") {
		t.Fatalf("roles equal to the implicit ones are left out: %s", body)
	}
	request.Roles = systemoneprotocol.Roles{Mode: systemoneprotocol.RolesNone}
	if _, _, err := (VLLMSR40Codec{}).Plan(systemoneprotocol.Batch{Items: []systemoneprotocol.NamedRequest{{Request: request}}}, Target{Model: "m"}, DefaultPolicy); err == nil {
		t.Fatal("no roles cannot go to vllm-sr/4.0")
	}
}
