package systemone

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	batchRequest  = `{"model":"vllm-sr/auto","states":{"first":{"state":{"request":"hi"},"questions":{"task":{"type":"choice","instructions":"Type?","criteria":{"code":"Code","chat":"Chat"}},"risk":{"type":"noul","instructions":"Unsafe?","require_full_input":true}}},"second":{"state":"hello","questions":{"difficulty":{"type":"score","instructions":"Difficulty?","criteria":["easy","hard"]}}}}}`
	batchResponse = `{"model":"small","states":{"first":{"answers":{"task":{"type":"choice","choice":"chat","confidence":0.9,"probabilities":{"code":0.01,"chat":0.99}},"risk":{"type":"noul","noul":0.01,"input_coverage":"complete"}}},"second":{"answers":{"difficulty":{"type":"score","score":0.1,"confidence":0.8,"probabilities":{"0":0.9,"1":0.1}}}}}}`
)

func TestNativeObservationKeepsAllStatesAndConfidentFalse(t *testing.T) {
	r, err := ParseNativeRequest(json.RawMessage(batchRequest))
	if err != nil {
		t.Fatal(err)
	}
	if string(r.Body) != batchRequest || len(r.Questions) != 3 || !strings.Contains(r.SignalText, "second") {
		t.Fatal("request or bundle lost")
	}
	o := r.Observe(json.RawMessage(batchResponse))
	if !complete(o) {
		t.Fatal("complete multi-state response marked invalid")
	}
	threshold := 0.75
	acceptance := &config.EscalateAccept{Native: []config.NativeAcceptanceRule{
		{QuestionType: "choice", Field: "confidence", NumericPredicate: config.NumericPredicate{GTE: &threshold}},
		{QuestionType: "score", Field: "confidence", NumericPredicate: config.NumericPredicate{GTE: &threshold}},
		{QuestionType: "noul", Field: "probability_margin", NumericPredicate: config.NumericPredicate{GTE: &threshold}},
	}}
	if !acceptNative(o, acceptance) {
		t.Fatal("confident false should satisfy the probability margin")
	}
}

func TestNativeRequestIsAStandardRequestOrABatch(t *testing.T) {
	question := `{"q":{"type":"noul","instructions":"Is it?"}}`
	entry := `{"state":"hello","questions":` + question + `}`
	for name, body := range map[string]string{
		"state beside states":     `{"state":"hi","states":{"a":` + entry + `}}`,
		"questions beside states": `{"questions":` + question + `,"states":{"a":` + entry + `}}`,
		"images beside states":    `{"images":[],"states":{"a":` + entry + `}}`,
		"an empty batch":          `{"states":{}}`,
		"an entry without state":  `{"states":{"a":{"questions":` + question + `}}}`,
	} {
		if _, err := ParseNativeRequest(json.RawMessage(body)); err == nil {
			t.Fatalf("%s accepted: %s", name, body)
		}
	}
	r, err := ParseNativeRequest(json.RawMessage(`{"states":{"a":` + entry + `,"b":` + entry + `}}`))
	if err != nil || len(r.Questions) != 2 || r.Questions[0].State != "a" || r.Questions[1].State != "b" {
		t.Fatalf("a batch's questions by entry: %+v, %v", r, err)
	}
}

func TestNativeQualityCannotDropMissingOrIncompleteAnswers(t *testing.T) {
	r, _ := ParseNativeRequest(json.RawMessage(batchRequest))
	for _, body := range []string{
		`{}`, strings.Replace(batchResponse, `"input_coverage":"complete"`, `"input_coverage":"partial"`, 1),
		strings.Replace(batchResponse, `"type":"score"`, `"type":"score","error":"unavailable"`, 1),
	} {
		o := r.Observe(json.RawMessage(body))
		if len(o) != 3 || complete(o) {
			t.Fatalf("missing failure in %s", body)
		}
	}
}

func TestMalformedNativeEnvelopeCannotAcceptPartiallyDecodedAnswers(t *testing.T) {
	request, _ := ParseNativeRequest(json.RawMessage(singleRequest))
	body := strings.TrimSuffix(certainResponse, "}") + `,"states":"invalid"}`
	if observations := request.Observe(json.RawMessage(body)); observations[0].Valid {
		t.Fatal("malformed native envelope accepted its otherwise valid answer")
	}
}

func TestNativeGateRequiresCoverageAndKnownStatistics(t *testing.T) {
	r, _ := ParseNativeRequest(json.RawMessage(batchRequest))
	o := r.Observe(json.RawMessage(batchResponse))
	zero := 0.0
	rules := &config.EscalateAccept{}
	for _, question := range []string{"task", "risk", "difficulty"} {
		rules.Native = append(rules.Native, config.NativeAcceptanceRule{Question: question, Field: "top_probability", NumericPredicate: config.NumericPredicate{GTE: &zero}})
	}
	if !acceptNative(o, rules) {
		t.Fatal("covered answers rejected")
	}
	if acceptNative(o, &config.EscalateAccept{Native: rules.Native[:1]}) {
		t.Fatal("uncovered answers accepted")
	}
	rules.Native[1].Field = "confidence"
	if acceptNative(o, rules) {
		t.Fatal("absent Noul confidence was imputed as zero")
	}
	rules.Native[1].Field = "top_probability"
	bad := strings.Replace(batchResponse, `"0":0.9,"1":0.1`, `"10":0.9,"11":0.1`, 1)
	if acceptNative(r.Observe(json.RawMessage(bad)), rules) {
		t.Fatal("invalid score level probabilities accepted")
	}
}

func TestNativeProvenanceRequestPreservesTaskAndClientOptions(t *testing.T) {
	for _, options := range []string{"", `,"options":null`, `,"options":{}`, `,"options":{"return_meta":false}`, `,"options":{"return_meta":true}`, `,"options":{"deadline_ms":500,"max_tokens":8192}`} {
		t.Run(options, func(t *testing.T) {
			body := strings.TrimSuffix(batchRequest, "}") + options + "}"
			request, err := ParseNativeRequest(json.RawMessage(body))
			if err != nil {
				t.Fatal(err)
			}
			if string(request.Body) != body || request.ReturnMeta != strings.Contains(options, `"return_meta":true`) {
				t.Fatal("changed client request or visibility")
			}
			var original, internal map[string]json.RawMessage
			_ = json.Unmarshal([]byte(body), &original)
			_ = json.Unmarshal(request.InferenceBody, &internal)
			for _, field := range []string{"model", "state", "questions", "states"} {
				if string(original[field]) != string(internal[field]) {
					t.Fatalf("internal metadata changed %s", field)
				}
			}
			var inferenceOptions map[string]json.RawMessage
			_ = json.Unmarshal(internal["options"], &inferenceOptions)
			if string(inferenceOptions["return_meta"]) != "true" {
				t.Fatal("internal provenance was not requested")
			}
			if strings.Contains(options, "max_tokens") && (string(inferenceOptions["deadline_ms"]) != "500" || string(inferenceOptions["max_tokens"]) != "8192") {
				t.Fatal("inference options changed")
			}
			if strings.Contains(request.SignalText, "return_meta") || strings.Contains(request.SignalText, "max_tokens") {
				t.Fatal("transport options became task evidence")
			}
		})
	}
	for _, options := range []string{`true`, `[]`, `{"return_meta":null}`, `{"return_meta":"true"}`} {
		body := strings.TrimSuffix(singleRequest, "}") + `,"options":` + options + "}"
		if _, err := ParseNativeRequest(json.RawMessage(body)); err == nil {
			t.Fatalf("invalid options accepted: %s", options)
		}
	}
}
