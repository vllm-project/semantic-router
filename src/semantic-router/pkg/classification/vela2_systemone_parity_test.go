package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"slices"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// Vela2SystemOneEndpointEnv names a running runtime that serves one Vela 2.0
// model; TestVela2RouterMatchesSystemOne skips without it. Its report goes to
// the file Vela2SystemOneReportEnv names, when set.
const (
	Vela2SystemOneEndpointEnv = "VLLM_SRUN_VELA2_ENDPOINT"
	Vela2SystemOneReportEnv   = "VLLM_SRUN_VELA2_REPORT"
)

const vela2ParityConfig = `
version: v0.3
providers:
  defaults:
    model: m
  models:
    - name: m
      backend_refs:
        - name: b
          endpoint: localhost:8000
          protocol: http
global:
  model_catalog:
    deployments:
      vela2:
        provider: model_runtime
        endpoint: %s
        served_name: %s
    bindings:
      pii_classifier:
        deployment: vela2
        contract: token_spans.v1
      hallucination_detector:
        deployment: vela2
        contract: token_spans.v1
    modules:
      hallucination_mitigation:
        detector:
          min_span_length: 1
          min_span_confidence: 0
routing:
  signals:
    decision:
      - name: topics
        deployment: vela2
        timeout_ms: 60000
        question:
          type: set
          instructions: Which topics does the request mention?
          labels:
            - {key: billing, description: "payments, invoices, charges or refunds"}
            - {key: shipping, description: "deliveries, parcels, tracking or returns"}
            - {key: account, description: "logins, passwords or account settings"}
            - {key: medication, description: drugs or doses}
      - name: entities
        deployment: vela2
        timeout_ms: 60000
        question:
          type: span
          instructions: Which spans name a place or an organization?
          labels:
            - {key: location, description: "a city, country or address"}
            - {key: organization, description: a company or institution}
      - name: urgent
        deployment: vela2
        timeout_ms: 60000
        question:
          type: noul
          instructions: Does the request need an urgent answer?
    pii:
      - name: any_pii
        threshold: 0.0001
  decisions:
    - name: every-signal
      priority: 10
      rules:
        operator: OR
        conditions:
          - {type: decision, name: topics, label: billing}
          - {type: decision, name: topics, label: shipping}
          - {type: decision, name: topics, label: account}
          - {type: decision, name: topics, label: medication}
          - {type: decision, name: entities, label: location}
          - {type: decision, name: entities, label: organization}
          - {type: decision, name: urgent}
          - {type: pii, name: any_pii}
      modelRefs:
        - model: m
`

// The questions the configuration above asks, written out as a client of
// /v1/systemone would, independently of the Router's encoder.
var vela2ParityQuestions = map[string]interface{}{
	"topics": map[string]interface{}{"type": "set", "instructions": "Which topics does the request mention?", "criteria": orderedObject{
		{"billing", "payments, invoices, charges or refunds"},
		{"shipping", "deliveries, parcels, tracking or returns"},
		{"account", "logins, passwords or account settings"},
		{"medication", "drugs or doses"},
	}},
	"entities": map[string]interface{}{"type": "span", "instructions": "Which spans name a place or an organization?", "criteria": orderedObject{
		{"location", "a city, country or address"}, {"organization", "a company or institution"},
	}},
	"urgent":             map[string]interface{}{"type": "noul", "instructions": "Does the request need an urgent answer?"},
	"pii_classifier:pii": map[string]interface{}{"preset": "pii"},
}

var vela2ParityTexts = []string{
	"Hi, I'm Tom Baker (tom.baker@example.com). My card was charged twice and the parcel never arrived in Berlin.",
	"Please reset the password for my account; I can no longer log in from the Lisbon office of Acme Corp.",
	"What is the maximum daily dose of paracetamol for an adult? Call me at +1 415 555 0134.",
	"My order 4471 was shipped to 221B Baker Street, London, but the invoice lists the wrong VAT number.",
	"Ich heiße Anna Müller, wohne in München und möchte mein Abonnement bei der Deutschen Bahn kündigen.",
	"Write a haiku about autumn leaves.",
	"Our team at the University of Toronto needs the refund for the conference fee by Friday.",
	"请帮我查询一下发往上海的包裹，收件人是李华，电话 13800138000。",
}

var vela2ParityHallucinations = []struct{ Context, Question, Answer string }{
	{"For adults, the maximum dose of paracetamol is 4 grams in 24 hours.", "What is the maximum daily dose of paracetamol for an adult?", "Adults can take up to 6 grams of paracetamol in 24 hours."},
	{"The Eiffel Tower is 330 metres tall and stands in Paris.", "How tall is the Eiffel Tower?", "The Eiffel Tower in Paris is 330 metres tall."},
	{"Acme's refund policy allows returns within 30 days of delivery with a receipt.", "Can I return my order?", "Yes, Acme accepts returns within 90 days, and no receipt is needed."},
}

// orderedObject is a JSON object whose keys keep their order.
type orderedObject [][2]string

func (o orderedObject) MarshalJSON() ([]byte, error) {
	var buffer bytes.Buffer
	buffer.WriteByte('{')
	for index, pair := range o {
		if index > 0 {
			buffer.WriteByte(',')
		}
		key, _ := json.Marshal(pair[0])
		value, _ := json.Marshal(pair[1])
		buffer.Write(key)
		buffer.WriteByte(':')
		buffer.Write(value)
	}
	buffer.WriteByte('}')
	return buffer.Bytes(), nil
}

type systemOneAnswer struct {
	Model   string `json:"model"`
	Answers map[string]struct {
		Noul  *float64 `json:"noul"`
		Error string   `json:"error"`
	} `json:"answers"`
	Sets map[string]struct {
		Selected      []string           `json:"selected"`
		Probabilities map[string]float64 `json:"probabilities"`
	} `json:"sets"`
	Spans map[string][]modelservice.Span `json:"spans"`
}

// servedModel is the one model a runtime serves.
func servedModel(t *testing.T, endpoint string) string {
	t.Helper()
	response, err := http.Get(strings.TrimRight(endpoint, "/") + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var models struct {
		Data []struct {
			ID string `json:"id"`
		} `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&models); err != nil || len(models.Data) != 1 {
		t.Fatalf("%s must serve one model: %+v %v", endpoint, models, err)
	}
	return models.Data[0].ID
}

func askSystemOne(t *testing.T, endpoint string, state interface{}, questions map[string]interface{}) systemOneAnswer {
	t.Helper()
	body, err := json.Marshal(map[string]interface{}{"state": state, "questions": questions})
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.Post(strings.TrimRight(endpoint, "/")+"/v1/systemone", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("/v1/systemone answered %d: %s", response.StatusCode, raw)
	}
	var answer systemOneAnswer
	if err := json.Unmarshal(raw, &answer); err != nil {
		t.Fatal(err)
	}
	return answer
}

func spanLabels(spans []modelservice.Span) []string {
	var labels []string
	for _, span := range spans {
		if !slices.Contains(labels, span.Label) {
			labels = append(labels, span.Label)
		}
	}
	sort.Strings(labels)
	return labels
}

// TestVela2RouterMatchesSystemOne routes on a live Vela 2.0 model through the
// Router's own signal path (configuration, classifier, lease, request bundle
// and client) and checks every Set, Span, PII and hallucination answer
// against /v1/systemone asked the same questions about the same text.
func TestVela2RouterMatchesSystemOne(t *testing.T) {
	endpoint := os.Getenv(Vela2SystemOneEndpointEnv)
	if os.Getenv("VLLM_SR_REQUIRE_MODEL_TESTS") == "1" {
		// The published-model runner serves the pinned 0.3B there.
		endpoint, _ = requireVela2Runtime(t, Vela2SystemOneEndpointEnv, config.Vela2SignalModel)
	}
	if endpoint == "" {
		t.Skipf("set %s to a runtime serving one Vela 2.0 model", Vela2SystemOneEndpointEnv)
	}
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(vela2ParityConfig, endpoint, servedModel(t, endpoint))))
	if err != nil {
		t.Fatal(err)
	}
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.Acquire(cfg)
	if err != nil {
		t.Fatal(err)
	}
	classifier, err := buildClassifierWithAdmission(cfg, nil, nil, nil, nil, RecipeRuntimeOptions{Runtime: serving.New(lease, nil)})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = classifier.Close() })
	if err = classifier.InitializeRuntime(); err != nil {
		t.Fatal(err)
	}
	card, err := lease.Card(context.Background(), "vela2")
	if err != nil {
		t.Fatal(err)
	}
	report := map[string]interface{}{"model": card.ID, "revision": card.Revision, "model_sha256": card.ModelSHA256, "device": card.Device, "profile": card.Profile}
	var rows []map[string]interface{}
	for _, text := range vela2ParityTexts {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		results := classifier.evaluateAllSignalsWithContext(SignalEvaluationInput{Text: text, RequestFacts: RequestFacts{Context: ctx}}, cfg.Decisions, true)
		cancel()
		if len(results.SignalErrors) > 0 {
			t.Fatalf("%q: signal errors %v", text, results.SignalErrors)
		}
		want := askSystemOne(t, endpoint, text, vela2ParityQuestions)
		row := map[string]interface{}{"text": text}
		var selected []string
		for label, probability := range want.Sets["topics"].Probabilities {
			if got := results.SignalValues["decision:topics:"+label]; got != probability {
				t.Fatalf("%q: set label %s = %v, /v1/systemone %v", text, label, got, probability)
			}
		}
		for _, name := range results.MatchedDecisionRules {
			if label, ok := strings.CutPrefix(name, "topics:"); ok {
				selected = append(selected, label)
			}
		}
		sort.Strings(selected)
		wantSelected := slices.Clone(want.Sets["topics"].Selected)
		sort.Strings(wantSelected)
		if !slices.Equal(selected, wantSelected) {
			t.Fatalf("%q: set labels matched %v, /v1/systemone selected %v", text, selected, wantSelected)
		}
		var spanned []string
		for _, name := range results.MatchedDecisionRules {
			if label, ok := strings.CutPrefix(name, "entities:"); ok {
				spanned = append(spanned, label)
			}
		}
		sort.Strings(spanned)
		if wantSpans := spanLabels(want.Spans["entities"]); !slices.Equal(spanned, wantSpans) {
			t.Fatalf("%q: span labels matched %v, /v1/systemone found %v", text, spanned, wantSpans)
		}
		for _, span := range want.Spans["entities"] {
			if got := results.SignalValues["decision:entities:"+span.Label]; got < span.Probability {
				t.Fatalf("%q: span label %s = %v, below a /v1/systemone span at %v", text, span.Label, got, span.Probability)
			}
		}
		if got, noul := results.SignalValues["decision:entities"], want.Answers["entities"].Noul; noul == nil || got != *noul {
			t.Fatalf("%q: span question value %v, /v1/systemone %v", text, got, noul)
		}
		if got, noul := results.SignalValues["decision:urgent"], want.Answers["urgent"].Noul; noul == nil || got != *noul {
			t.Fatalf("%q: noul %v, /v1/systemone %v", text, got, noul)
		}
		entities := slices.Clone(results.PIIEntities)
		sort.Strings(entities)
		if wantPII := spanLabels(want.Spans["pii_classifier:pii"]); !slices.Equal(entities, wantPII) {
			t.Fatalf("%q: PII entities %v, /v1/systemone %v", text, entities, wantPII)
		}
		row["set_selected"], row["span_labels"], row["pii_entities"] = selected, spanned, entities
		row["set_probabilities"], row["span_question_noul"] = want.Sets["topics"].Probabilities, results.SignalValues["decision:entities"]
		rows = append(rows, row)
	}
	report["requests"] = rows
	report["hallucination"] = checkVela2Hallucination(t, endpoint, classifier)
	if path := os.Getenv(Vela2SystemOneReportEnv); path != "" {
		encoded, _ := json.MarshalIndent(report, "", "  ")
		if err := os.WriteFile(path, append(encoded, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
	}
}

func checkVela2Hallucination(t *testing.T, endpoint string, classifier *Classifier) []map[string]interface{} {
	t.Helper()
	detector, err := NewHallucinationDetector(&classifier.models.cfg.HallucinationMitigation.HallucinationModel, classifier.models)
	if err != nil {
		t.Fatal(err)
	}
	if err := detector.Initialize(); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = detector.Close() })
	var rows []map[string]interface{}
	for _, item := range vela2ParityHallucinations {
		got, err := detector.Detect(context.Background(), item.Context, item.Question, item.Answer)
		if err != nil {
			t.Fatal(err)
		}
		want := askSystemOne(t, endpoint, map[string]string{"request": item.Question, "context": item.Context, "answer": item.Answer},
			map[string]interface{}{"hallucination_detector:halu": map[string]interface{}{"preset": "halu"}})
		spans := want.Spans["hallucination_detector:halu"]
		if len(got.Spans) != len(spans) {
			t.Fatalf("%q: %d unsupported spans, /v1/systemone %d", item.Answer, len(got.Spans), len(spans))
		}
		var texts []string
		for index, span := range spans {
			if got.Spans[index].Text != span.Text || math.Abs(float64(got.Spans[index].Confidence)-span.Probability) > 1e-6 {
				t.Fatalf("%q: span %d = %+v, /v1/systemone %+v", item.Answer, index, got.Spans[index], span)
			}
			texts = append(texts, span.Text)
		}
		rows = append(rows, map[string]interface{}{"answer": item.Answer, "unsupported": texts, "detected": got.HallucinationDetected})
	}
	return rows
}
