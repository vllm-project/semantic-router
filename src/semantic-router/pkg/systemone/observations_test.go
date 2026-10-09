package systemone

import (
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	mixedRequest  = `{"model":"vllm-sr/auto","state":{"request":"hi"},"questions":{"task":{"type":"choice","instructions":"Type?","criteria":{"code":"Code","chat":"Chat"}},"risk":{"type":"noul","instructions":"Unsafe?","require_full_input":true}},"states":{"second":{"state":"hello","questions":{"difficulty":{"type":"score","instructions":"Difficulty?","criteria":["easy","hard"]}}}}}`
	mixedResponse = `{"model":"small","answers":{"task":{"type":"choice","choice":"chat","confidence":0.9,"probabilities":{"code":0.01,"chat":0.99}},"risk":{"type":"noul","noul":0.01,"input_coverage":"complete"}},"states":{"second":{"answers":{"difficulty":{"type":"score","score":0.1,"confidence":0.8,"probabilities":{"0":0.9,"1":0.1}}}}}}`
)

func TestNativeObservationKeepsAllStatesAndConfidentFalse(t *testing.T) {
	r, err := ParseNativeRequest(json.RawMessage(mixedRequest))
	if err != nil {
		t.Fatal(err)
	}
	if string(r.Body) != mixedRequest || len(r.Questions) != 3 || !strings.Contains(r.SignalText, "second") {
		t.Fatal("request or bundle lost")
	}
	o := r.Observe(json.RawMessage(mixedResponse))
	f := r.Features(o)
	if f[10] != 0 || math.Abs(f[1]-0.9) > 1e-9 || math.Abs(f[4]-0.8) > 1e-9 {
		t.Fatalf("features=%v", f)
	}
	threshold := 0.75
	acceptance := &config.NativeAcceptance{Rules: []config.NativeAcceptanceRule{
		{QuestionType: "choice", Field: "confidence", Predicate: config.NumericPredicate{GTE: &threshold}},
		{QuestionType: "score", Field: "confidence", Predicate: config.NumericPredicate{GTE: &threshold}},
		{QuestionType: "noul", Field: "probability_margin", Predicate: config.NumericPredicate{GTE: &threshold}},
	}}
	if !acceptNative(o, acceptance, true) {
		t.Fatal("confident false should satisfy the probability margin")
	}
}

func TestNativeQualityCannotDropMissingOrIncompleteAnswers(t *testing.T) {
	r, _ := ParseNativeRequest(json.RawMessage(mixedRequest))
	for _, body := range []string{
		`{}`, strings.Replace(mixedResponse, `"input_coverage":"complete"`, `"input_coverage":"partial"`, 1),
		strings.Replace(mixedResponse, `"type":"score"`, `"type":"score","error":"unavailable"`, 1),
	} {
		o := r.Observe(json.RawMessage(body))
		if len(o) != 3 || r.Features(o)[10] != 1 {
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
	r, _ := ParseNativeRequest(json.RawMessage(mixedRequest))
	o := r.Observe(json.RawMessage(mixedResponse))
	zero := 0.0
	partial := &config.NativeAcceptance{Rules: []config.NativeAcceptanceRule{{Question: "task", Field: "confidence", Predicate: config.NumericPredicate{GTE: &zero}}}}
	if acceptNative(o, partial, true) {
		t.Fatal("uncovered answers accepted")
	}
	if !acceptNative(o, partial, false) {
		t.Fatal("additional stage predicate failed")
	}
	partial.Rules[0].Question = "risk"
	if acceptNative(o, partial, false) {
		t.Fatal("absent Noul confidence was imputed as zero")
	}
	bad := strings.Replace(mixedResponse, `"0":0.9,"1":0.1`, `"10":0.9,"11":0.1`, 1)
	if r.Features(r.Observe(json.RawMessage(bad)))[10] != 1 {
		t.Fatal("invalid score level probabilities accepted")
	}
}

func TestNativeProvenanceRequestPreservesTaskAndClientOptions(t *testing.T) {
	for _, options := range []string{"", `,"options":null`, `,"options":{}`, `,"options":{"return_meta":false}`, `,"options":{"return_meta":true}`, `,"options":{"profile":"exact","max_tokens":8192}`} {
		t.Run(options, func(t *testing.T) {
			body := strings.TrimSuffix(mixedRequest, "}") + options + "}"
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
			if strings.Contains(options, "max_tokens") && (string(inferenceOptions["profile"]) != `"exact"` || string(inferenceOptions["max_tokens"]) != "8192") {
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
