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
	spanQuestion = `{"type":"span","instructions":"Find personal data.","criteria":{"EMAIL":"An email address","NAME":"A person's name"}}`
	// é is one code point and two bytes: the address starts at code point 5
	// and at byte 6, and the text is 17 code points long.
	spanState   = `"café tom.b@ex.com"`
	spanRequest = `{"model":"vllm-sr/auto","state":` + spanState + `,"questions":{"pii":` + spanQuestion + `}}`
	// One state's part of a response: the address is found, almost surely.
	spanAnswered = `"answers":{"pii":{"type":"noul","noul":0.97}},"spans":{"pii":[{"label":"EMAIL","start":5,"end":17,"text":"tom.b@ex.com","probability":0.97}]},"thresholds":{"pii":0.5}`
	// No word reaches the threshold: the list of spans is empty.
	noSpanAnswered = `"answers":{"pii":{"type":"noul","noul":0.02}},"spans":{"pii":[]},"thresholds":{"pii":0.5}`
	spanResponse   = `{"model":"span-actual",` + spanAnswered + `,"usage":{"input_tokens":12,"output_tokens":0}}`
)

// spanGate gates every span answer on a statistic of its Noul.
func spanGate(field string, threshold float64) *config.EscalateAccept {
	return &config.EscalateAccept{Native: []config.NativeAcceptanceRule{{QuestionType: "span", Field: field, NumericPredicate: config.NumericPredicate{GTE: &threshold}}}}
}

// Auto admits a span question by the labels it names and the text it reads: a
// text state, or the text part of an object state that its over names. It
// refuses a request that leaves the part to the model, a text state that a
// model may read as an object, and a label without a name, and needs a model
// that answers spans.
func TestAutoAdmitsASpanQuestionByItsLabelsAndTheTextItReads(t *testing.T) {
	const (
		parts = `{"request":"Who wrote it?","answer":"Tom did."}`
		over  = `{"type":"span","over":"answer","criteria":{"NAME":"A person's name"}}`
	)
	for name, tc := range map[string]struct{ state, question, target string }{
		"criteria":                  {spanState, spanQuestion, "café tom.b@ex.com"},
		"labels list":               {spanState, `{"type":"span","labels":[{"key":"EMAIL","description":"An email address"},{"key":"NAME"}]}`, "café tom.b@ex.com"},
		"a text state with over":    {spanState, strings.Replace(spanQuestion, `"type":"span"`, `"type":"span","over":"request"`, 1), "café tom.b@ex.com"},
		"a part of an object":       {parts, over, "Tom did."},
		"a part without a role":     {`{"note":"Ask Tom.","answer":"Tom did."}`, strings.Replace(over, `"over":"answer"`, `"over":"note"`, 1), "Ask Tom."},
		"a text that opens a brace": {`"{Tom} did."`, over, "{Tom} did."},
		"a text that holds a list":  {`"[\"Tom did.\"]"`, over, `["Tom did."]`},
	} {
		request, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","state":` + tc.state + `,"questions":{"pii":` + tc.question + `}}`))
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if q := request.Questions[0]; q.Type != "span" || !q.Labels["NAME"] || q.Target != tc.target {
			t.Fatalf("%s: question %+v", name, q)
		}
		if !reflect.DeepEqual(request.Requirements.QuestionTypes, []string{"span"}) {
			t.Fatalf("%s: requirements %+v", name, request.Requirements)
		}
	}
	for name, tc := range map[string]struct{ state, question string }{
		"no labels":                 {spanState, `{"type":"span","instructions":"Find personal data."}`},
		"empty criteria":            {spanState, `{"type":"span","criteria":{}}`},
		"criteria as a list":        {spanState, `{"type":"span","criteria":["EMAIL","NAME"]}`},
		"an object without over":    {parts, `{"type":"span","criteria":{"NAME":"A person's name"}}`},
		"a null over":               {parts, strings.Replace(over, `"over":"answer"`, `"over":null`, 1)},
		"over naming a role":        {parts, strings.Replace(over, `"over":"answer"`, `"over":"user"`, 1)},
		"over naming several parts": {parts, strings.Replace(over, `"over":"answer"`, `"over":["answer"]`, 1)},
		"over naming a null part":   {`{"request":"Who wrote it?","answer":null}`, over},
		"over naming an object":     {`{"request":"Who wrote it?","answer":{"text":"Tom did."}}`, over},
		"an array state":            {`["Tom did."]`, over},
		// Vela 2.0 reads these texts as the object they hold, and its spans
		// are then ranges of the answer part.
		"a text that holds an object":                {`"{\"answer\":\"Tom did.\"}"`, over},
		"such a text without over":                   {`"{\"answer\":\"Tom did.\"}"`, `{"type":"span","criteria":{"NAME":"A person's name"}}`},
		"such a text between white space":            {`" \n{\"answer\":\"Tom did.\"}\t"`, over},
		"such a text after a separator Python trims": {`"\u001c{\"answer\":\"Tom did.\"}"`, over},
		"a label without a name":                     {spanState, `{"type":"span","criteria":{"":"Anything","NAME":"A person's name"}}`},
		"a label named by white space":               {spanState, `{"type":"span","criteria":{" ":"Anything","NAME":"A person's name"}}`},
		"a listed label without a key":               {spanState, `{"type":"span","labels":[{"description":"Anything"},{"key":"NAME"}]}`},
	} {
		if _, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","state":` + tc.state + `,"questions":{"pii":` + tc.question + `}}`)); err == nil {
			t.Fatalf("%s accepted: %s over %s", name, tc.question, tc.state)
		}
	}
}

// A span answer is valid when its Noul is and each of its spans names a
// declared label and a range of the text the question read, counted in code
// points, with that range's text. An empty list is an answer; a missing or
// malformed part is not.
func TestASpanAnswerNeedsItsNoulAndSpansOfTheTextItsQuestionRead(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(spanRequest))
	if err != nil {
		t.Fatal(err)
	}
	const found = `{"label":"EMAIL","start":5,"end":17,"text":"tom.b@ex.com","probability":0.97}`
	for _, tc := range []struct {
		name, old, new string
		valid          bool
	}{
		{"as the runtime writes it", "", "", true},
		{"no span found", spanAnswered, noSpanAnswered, true},
		{"two spans", found, found + `,{"label":"NAME","start":0,"end":4,"text":"café","probability":0.6}`, true},
		{"a span of the whole text", found, `{"label":"NAME","start":0,"end":17,"text":"café tom.b@ex.com","probability":0.6}`, true},
		{"no spans", `,"spans":{"pii":[` + found + `]}`, ``, false},
		{"no list for the question", `"spans":{"pii":[` + found + `]}`, `"spans":{}`, false},
		{"a null list", `[` + found + `]`, `null`, false},
		{"a list that is an object", `[` + found + `]`, found, false},
		{"a null span", found, `null`, false},
		{"an undeclared label", `"label":"EMAIL"`, `"label":"PHONE"`, false},
		{"no label", `"label":"EMAIL",`, ``, false},
		{"a null label", `"label":"EMAIL"`, `"label":null`, false},
		{"a blank label", `"label":"EMAIL"`, `"label":""`, false},
		{"no probability", `,"probability":0.97}]`, `}]`, false},
		{"a null probability", `"probability":0.97}]`, `"probability":null}]`, false},
		{"a probability out of range", `"probability":0.97}]`, `"probability":1.97}]`, false},
		{"no start", `"start":5,`, ``, false},
		{"no end", `"end":17,`, ``, false},
		{"an offset that is not whole", `"start":5,`, `"start":5.5,`, false},
		{"a negative start", `"start":5,`, `"start":-1,`, false},
		{"an empty range", `"start":5,`, `"start":17,`, false},
		{"a reversed range", `"start":5,"end":17`, `"start":17,"end":5`, false},
		{"an end past the text", `"end":17`, `"end":18`, false},
		{"byte offsets", `"start":5,"end":17`, `"start":6,"end":18`, false},
		{"a range of other text", `"start":5,"end":17`, `"start":4,"end":16`, false},
		{"text that is not the range's", `"text":"tom.b@ex.com"`, `"text":"tom.b@ex.org"`, false},
		{"no text", `"text":"tom.b@ex.com",`, ``, false},
		{"no Noul", `"answers":{"pii":{"type":"noul","noul":0.97}}`, `"answers":{}`, false},
		{"answered as a span", `{"type":"noul","noul":0.97}`, `{"type":"span","noul":0.97}`, false},
		{"a failed question", `{"type":"noul","noul":0.97}`, `{"type":"noul","error":"unavailable"}`, false},
		{"a Noul out of range", `"noul":0.97}}`, `"noul":1.97}}`, false},
	} {
		body := strings.Replace(spanResponse, tc.old, tc.new, 1)
		if tc.old != "" && body == spanResponse {
			t.Fatalf("%s: the response was not changed", tc.name)
		}
		observation := request.Observe(json.RawMessage(body))[0]
		if observation.Valid != tc.valid {
			t.Errorf("%s: valid %v, want %v: %s", tc.name, observation.Valid, tc.valid, body)
		}
		if !tc.valid && (observation.Probability != nil || observation.Distribution != nil) {
			t.Errorf("%s: an invalid span answer kept its statistics %+v", tc.name, observation)
		}
	}

	// A span carries its label: one without it names no label, whatever the
	// question declares.
	unnamed := *request
	unnamed.Questions = []Question{request.Questions[0]}
	unnamed.Questions[0].Labels = map[string]bool{"": true, "EMAIL": true}
	if unnamed.Observe(json.RawMessage(strings.Replace(spanResponse, `"label":"EMAIL",`, ``, 1)))[0].Valid {
		t.Fatal("a span without a label was accepted")
	}

	// The spans of a part are ranges of that part, not of the whole state.
	part, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","state":{"request":"Who is Tom?","answer":` + spanState + `},"questions":{"pii":` +
		strings.Replace(spanQuestion, `"type":"span"`, `"type":"span","over":"answer"`, 1) + `}}`))
	if err != nil {
		t.Fatal(err)
	}
	if !part.Observe(json.RawMessage(spanResponse))[0].Valid {
		t.Fatal("a span of the part its question names was refused")
	}
	other := strings.Replace(spanResponse, found, `{"label":"NAME","start":7,"end":10,"text":"Tom","probability":0.9}`, 1)
	if part.Observe(json.RawMessage(other))[0].Valid {
		t.Fatal("a span of a part the question did not read was accepted")
	}

	// A span question that requires its full input needs the coverage its
	// Noul proves.
	full, err := ParseNativeRequest(json.RawMessage(strings.Replace(spanRequest, `"type":"span"`, `"type":"span","require_full_input":true`, 1)))
	if err != nil {
		t.Fatal(err)
	}
	if full.Observe(json.RawMessage(spanResponse))[0].Valid {
		t.Fatal("a span answer without input coverage answered a question that requires full input")
	}
	covered := strings.Replace(spanResponse, `"noul":0.97}}`, `"noul":0.97,"input_coverage":"complete"}}`, 1)
	if !full.Observe(json.RawMessage(covered))[0].Valid {
		t.Fatal("a span answer with complete input coverage was refused")
	}
}

// Each batch entry's spans are observed from that entry and checked against
// that entry's text. An empty list is an answer and a missing one is not; a
// malformed span fails its own question and no other.
func TestABatchObservesTheSpansOfEachEntry(t *testing.T) {
	const (
		questions = `"questions":{"pii":` + spanQuestion + `,"risk":{"type":"noul","instructions":"Unsafe?"}}`
		entry     = `{"state":` + spanState + `,` + questions + `}`
		risk      = `"answers":{"risk":{"type":"noul","noul":0.01}`
		// Another text, whose address starts at code point 9.
		letter         = `{"state":"Write to ann@ex.org",` + questions + `}`
		letterAnswered = `"answers":{"pii":{"type":"noul","noul":0.9}},"spans":{"pii":[{"label":"EMAIL","start":9,"end":19,"text":"ann@ex.org","probability":0.9}]}`
	)
	request, err := ParseNativeRequest(json.RawMessage(`{"model":"vllm-sr/auto","states":{"found":` + entry + `,"letter":` + letter + `,"none":` + entry +
		`,"missing":` + entry + `,"malformed":` + entry + `,"swapped":` + letter + `}}`))
	if err != nil {
		t.Fatal(err)
	}
	answered := func(part string) string { return "{" + strings.Replace(part, `"answers":{`, risk+",", 1) + "}" }
	response := `{"model":"m","states":{` +
		`"found":` + answered(spanAnswered) + `,` +
		`"letter":` + answered(letterAnswered) + `,` +
		`"none":` + answered(noSpanAnswered) + `,` +
		`"missing":{` + risk + `,"pii":{"type":"noul","noul":0.97}}},` +
		`"malformed":` + answered(strings.Replace(spanAnswered, `"end":17`, `"end":"17"`, 1)) + `,` +
		// The spans of the found entry, answered for the letter.
		`"swapped":` + answered(spanAnswered) + `}}`
	got := map[string]bool{}
	for _, observation := range request.Observe(json.RawMessage(response)) {
		got[observation.State+"."+observation.Name] = observation.Valid
	}
	want := map[string]bool{
		"found.pii": true, "found.risk": true,
		"letter.pii": true, "letter.risk": true,
		"none.pii": true, "none.risk": true,
		"missing.pii": false, "missing.risk": true,
		"malformed.pii": false, "malformed.risk": true,
		"swapped.pii": false, "swapped.risk": true,
	}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("valid answers %v, want %v: %s", got, want, response)
	}
}

// A span answer is as sure as its Noul: its probability margin and its top
// probability are the Noul's, and it has no confidence.
func TestASpanIsGatedByItsNoul(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(spanRequest))
	if err != nil {
		t.Fatal(err)
	}
	// The Noul is 0.97: a margin of 0.94 and a top probability of 0.97.
	observations := request.Observe(json.RawMessage(spanResponse))
	for field, passes := range map[string]float64{"probability_margin": 0.9, "top_probability": 0.95} {
		if !acceptNative(observations, spanGate(field, passes), true) {
			t.Fatalf("a span answer whose Noul passes %s was rejected", field)
		}
		if acceptNative(observations, spanGate(field, 0.98), true) {
			t.Fatalf("a span answer passed a %s above its Noul's", field)
		}
	}
	confident := request.Observe(json.RawMessage(strings.Replace(spanResponse, `"noul":0.97}}`, `"noul":0.97,"confidence":0.99}}`, 1)))
	if !confident[0].Valid || acceptNative(confident, spanGate("confidence", 0), true) {
		t.Fatalf("a span answer was given a confidence: %+v", confident[0])
	}
	// Finding nothing, surely, is as sure an answer: a Noul of 0.02.
	none := request.Observe(json.RawMessage(`{"model":"m",` + noSpanAnswered + `}`))
	if !acceptNative(none, spanGate("probability_margin", 0.9), true) {
		t.Fatal("a span answer that surely finds nothing was rejected")
	}
}

// A span request moves on while its Noul is unsure or a span is malformed, and
// the accepted model's response is returned as it wrote it, for a standard
// request and a batch.
func TestASpanRequestEscalatesUntilItsAnswerIsSureAndWellFormed(t *testing.T) {
	const entry = `{"state":` + spanState + `,"questions":{"pii":` + spanQuestion + `}}`
	unsure := strings.NewReplacer(`"noul":0.97`, `"noul":0.6`).Replace(spanAnswered)
	malformed := strings.Replace(spanAnswered, `"start":5,"end":17`, `"start":6,"end":18`, 1)
	for name, shape := range map[string]struct {
		request string
		respond func(part string) string
	}{
		"standard": {spanRequest, func(part string) string { return `{"model":"m",` + part + `}` }},
		"batch": {`{"model":"vllm-sr/auto","states":{"a":` + entry + `,"b":` + entry + `}}`, func(part string) string {
			return `{"model":"m","states":{"a":{` + spanAnswered + `},"b":{` + part + `}}}`
		}},
	} {
		for fast, want := range map[string][]string{spanAnswered: {"kai"}, unsure: {"kai", "nox"}, malformed: {"kai", "nox"}} {
			request, err := ParseNativeRequest(json.RawMessage(shape.request))
			if err != nil {
				t.Fatal(err)
			}
			executor, err := NewExecutor(&config.AlgorithmConfig{
				Type:   config.DecisionAlgorithmEscalate,
				Budget: &config.AlgorithmBudget{Deadline: "3s", MaxCalls: 3},
				Escalate: &config.EscalateAlgorithmConfig{
					Quality: &config.EscalateQuality{Type: "uncalibrated", Accept: spanGate("probability_margin", 0)},
					Stages: []config.EscalateStage{
						{Name: "fast", Kind: "native", Model: "kai", Accept: spanGate("probability_margin", 0.5)},
						{Name: "strong", Kind: "native", Model: "nox"},
					},
				},
			}, nil)
			if err != nil {
				t.Fatal(err)
			}
			var calls []string
			answers := map[string]string{"kai": shape.respond(fast), "nox": shape.respond(spanAnswered)}
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

// A span question goes only to the member that lists spans, and to none when
// no member does.
func TestASpanQuestionGoesOnlyToMembersThatAnswerSpans(t *testing.T) {
	request, err := ParseNativeRequest(json.RawMessage(spanRequest))
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
		return 200, []byte(spanResponse), nil
	}
	spans := textOnly("listing")
	spans.QuestionTypes = append(spans.QuestionTypes, "span")
	ctx := withMembers(context.Background(), members(map[string]Capabilities{"kai": textOnly("card"), "vega": spans}))
	result, err := executor.Execute(ctx, request, invoke)
	if err != nil || result.Model != "vega" || !reflect.DeepEqual(calls, []string{"vega"}) {
		t.Fatalf("result %+v, calls %v, error %v", result, calls, err)
	}

	calls = nil
	ctx = withMembers(context.Background(), members(map[string]Capabilities{"kai": textOnly("card"), "vega": textOnly("listing")}))
	_, err = executor.Execute(ctx, request, invoke)
	var refusal *IncapableError
	if !errors.As(err, &refusal) || !reflect.DeepEqual(refusal.Missing, []string{"span"}) || len(calls) != 0 {
		t.Fatalf("error %v, calls %v", err, calls)
	}
}
