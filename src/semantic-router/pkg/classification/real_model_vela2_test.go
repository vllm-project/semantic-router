package classification

import (
	"bytes"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"net/http/httptest"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/tidwall/sjson"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// The published-model runner serves each Vela 2.0 size it provisions from a
// runtime of its own, on CPU at the profile the Router runs that size with,
// and names the endpoint in one of these variables (the 0.3B shares
// Vela2SystemOneEndpointEnv with the System One parity test).
const vela2Endpoint08BEnv = "VLLM_SRUN_VELA2_08B_ENDPOINT"

// vela2AnswersPath holds the answers each pinned size gave the corpus on CPU.
// With vela2RecordEnv=1 the tests rewrite its entry for the size they serve,
// which make record-vela2-answers does for every size.
const (
	vela2AnswersPath = "testdata/vela2_published_answers.json"
	vela2RecordEnv   = "VLLM_SR_VELA2_RECORD"
	vela2RecordHint  = "record the answers again with make record-vela2-answers"
)

const (
	// vela2CPUTolerance is how far an answer may move from its recorded value:
	// the runtime's golden tolerance on CPU, where AVX2 and AVX-512 kernels
	// round differently.
	vela2CPUTolerance = 1e-3
	// vela2StateTolerance bounds a state's answers in a call with several
	// states against the same questions asked about that state alone: on CPU,
	// max_speed packs the states' sequences into one forward and rounds the
	// last bits differently; exact runs them as separate requests.
	vela2StateTolerance = 1e-6
	// vela2Margin keeps a recorded decision (a Choice's top option, a Set
	// label's selection, a span) this far from its boundary, so a move within
	// the tolerance cannot flip it.
	vela2Margin = 0.01
)

// vela2Card is the card of the one model a runtime serves.
type vela2Card struct {
	ID          string `json:"id"`
	Family      string `json:"family"`
	Repo        string `json:"repo"`
	Revision    string `json:"revision"`
	ModelSHA256 string `json:"model_sha256"`
	Profile     string `json:"profile"`
	Device      string `json:"device"`
}

func servedCard(t *testing.T, endpoint string) vela2Card {
	t.Helper()
	response, err := http.Get(strings.TrimRight(endpoint, "/") + "/v1/models")
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	var models struct {
		Data []vela2Card `json:"data"`
	}
	if err := json.NewDecoder(response.Body).Decode(&models); err != nil || len(models.Data) != 1 {
		t.Fatalf("%s must serve one model: %+v %v", endpoint, models, err)
	}
	return models.Data[0]
}

// requireVela2Runtime returns the endpoint and card of the runtime that serves
// a pinned Vela 2.0 size. Like requireRealModel it needs an explicit endpoint,
// whose absence fails a run that requires the model, and it fails when the
// runtime serves another revision than the Router's registry pins, or not on
// CPU at the profile an implicit CPU deployment of the model runs.
func requireVela2Runtime(t *testing.T, env, model string) (string, vela2Card) {
	t.Helper()
	spec := config.GetModelByPath(model)
	if spec == nil || len(spec.Revision) != 40 {
		t.Fatalf("real-model default %q must have an immutable registry revision", model)
	}
	endpoint := os.Getenv(env)
	if endpoint == "" {
		if os.Getenv("VLLM_SR_REQUIRE_MODEL_TESTS") == "1" {
			t.Fatalf("required real model %s needs a runtime serving it at %s", spec.RepoID, env)
		}
		t.Skipf("optional real model %s requires a runtime serving it at %s", spec.RepoID, env)
	}
	deployment, err := config.ImplicitModelRuntimeDeployment(model, true)
	if err != nil {
		t.Fatal(err)
	}
	card := servedCard(t, endpoint)
	if card.Repo != spec.RepoID || card.Revision != spec.Revision {
		t.Fatalf("real model %s registers revision %s, but %s serves %s at revision %s", spec.RepoID, spec.Revision, endpoint, card.Repo, card.Revision)
	}
	if card.Device != "cpu" || card.Profile != deployment.Profile {
		t.Fatalf("%s serves %s on %s at profile %s; the Router runs it on cpu at %s", endpoint, card.ID, card.Device, card.Profile, deployment.Profile)
	}
	t.Logf("real model=%s registered_revision=%s model_sha256=%s profile=%s endpoint=%s", spec.RepoID, spec.Revision, card.ModelSHA256, card.Profile, endpoint)
	return endpoint, card
}

// vela2Exchange is one POST the Router sent through the recording proxy.
type vela2Exchange struct {
	path              string
	request, response []byte
	status            int
}

// vela2Recorder serves a proxy to a runtime and keeps every POST it forwards.
type vela2Recorder struct {
	target    string
	mu        sync.Mutex
	exchanges []vela2Exchange
}

func recordVela2Calls(t *testing.T, endpoint string) (string, *vela2Recorder) {
	t.Helper()
	recorder := &vela2Recorder{target: strings.TrimRight(endpoint, "/")}
	server := httptest.NewServer(http.HandlerFunc(recorder.forward))
	t.Cleanup(server.Close)
	return server.URL, recorder
}

func (r *vela2Recorder) forward(w http.ResponseWriter, req *http.Request) {
	body, err := io.ReadAll(req.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	forwarded, err := http.NewRequestWithContext(req.Context(), req.Method, r.target+req.URL.RequestURI(), bytes.NewReader(body))
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	forwarded.Header = req.Header.Clone()
	response, err := http.DefaultClient.Do(forwarded)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	defer response.Body.Close()
	answer, err := io.ReadAll(response.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadGateway)
		return
	}
	for key, values := range response.Header {
		w.Header()[key] = values
	}
	w.WriteHeader(response.StatusCode)
	_, _ = w.Write(answer)
	if req.Method == http.MethodPost {
		r.mu.Lock()
		r.exchanges = append(r.exchanges, vela2Exchange{path: req.URL.Path, request: body, response: answer, status: response.StatusCode})
		r.mu.Unlock()
	}
}

// take returns the exchanges recorded since the last take.
func (r *vela2Recorder) take() []vela2Exchange {
	r.mu.Lock()
	defer r.mu.Unlock()
	taken := r.exchanges
	r.exchanges = nil
	return taken
}

// vela2Call is a decisions call as the runtime got it, with its answer. The
// questions stay raw: a Set or Span question's label order is part of what
// the model reads.
type vela2Call struct {
	Model     *string                    `json:"model,omitempty"`
	State     json.RawMessage            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
	States    map[string]vela2CallState  `json:"states,omitempty"`
	Options   map[string]json.RawMessage `json:"options,omitempty"`
	answer    map[string]interface{}
}

type vela2CallState struct {
	State     json.RawMessage            `json:"state"`
	Questions map[string]json.RawMessage `json:"questions"`
}

// decisionCalls reads every decisions call out of recorded exchanges: the
// tasks of /v1/bundle calls and direct /v1/decisions calls.
func decisionCalls(t *testing.T, exchanges []vela2Exchange) []vela2Call {
	t.Helper()
	var calls []vela2Call
	for _, exchange := range exchanges {
		if exchange.status != http.StatusOK {
			t.Fatalf("%s answered %d: %s", exchange.path, exchange.status, exchange.response)
		}
		switch exchange.path {
		case "/v1/bundle":
			var request struct {
				Tasks []struct {
					ID        string     `json:"id"`
					Decisions *vela2Call `json:"decisions"`
				} `json:"tasks"`
			}
			var response struct {
				Results []struct {
					ID        string                 `json:"id"`
					Status    int                    `json:"status"`
					Decisions map[string]interface{} `json:"decisions"`
				} `json:"results"`
			}
			if json.Unmarshal(exchange.request, &request) != nil || json.Unmarshal(exchange.response, &response) != nil || len(request.Tasks) != len(response.Results) {
				t.Fatalf("unreadable bundle exchange: %s -> %s", exchange.request, exchange.response)
			}
			for index, task := range request.Tasks {
				if task.Decisions == nil {
					continue
				}
				if result := response.Results[index]; result.Status != http.StatusOK || result.Decisions == nil {
					t.Fatalf("decisions task %s answered %d", task.ID, result.Status)
				}
				task.Decisions.answer = response.Results[index].Decisions
				calls = append(calls, *task.Decisions)
			}
		case "/v1/decisions":
			var call vela2Call
			if json.Unmarshal(exchange.request, &call) != nil || json.Unmarshal(exchange.response, &call.answer) != nil {
				t.Fatalf("unreadable decisions exchange: %s -> %s", exchange.request, exchange.response)
			}
			calls = append(calls, call)
		}
	}
	return calls
}

// stateRequests are the requests that ask each state of a call alone: its own
// state first, then its further states by name, each with the call's model
// and options but no deadline.
func (c vela2Call) stateRequests() (names []string, requests []vela2Call) {
	options := make(map[string]json.RawMessage, len(c.Options))
	for key, value := range c.Options {
		if key != "deadline_ms" {
			options[key] = value
		}
	}
	names = append(names, "")
	requests = append(requests, vela2Call{Model: c.Model, State: c.State, Questions: c.Questions, Options: options})
	for _, name := range sortedKeys(c.States) {
		names = append(names, name)
		requests = append(requests, vela2Call{Model: c.Model, State: c.States[name].State, Questions: c.States[name].Questions, Options: options})
	}
	return names, requests
}

// withoutDeadline is the call itself, all of its states, without its deadline.
func (c vela2Call) withoutDeadline() vela2Call {
	_, requests := c.stateRequests()
	repeat := requests[0]
	repeat.States = c.States
	return repeat
}

// stateAnswer is the answer about one state of the call: its own ("") or a
// further state's.
func (c vela2Call) stateAnswer(t *testing.T, name string) map[string]interface{} {
	t.Helper()
	if name == "" {
		return c.answer
	}
	states, _ := c.answer["states"].(map[string]interface{})
	answer, ok := states[name].(map[string]interface{})
	if !ok {
		t.Fatalf("the call's answer has no state %q", name)
	}
	return answer
}

// askVela2 sends one decisions request directly to the runtime.
func askVela2(t *testing.T, endpoint string, request vela2Call) map[string]interface{} {
	t.Helper()
	body, err := json.Marshal(request)
	if err != nil {
		t.Fatal(err)
	}
	response, err := http.Post(strings.TrimRight(endpoint, "/")+"/v1/decisions", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatal(err)
	}
	defer response.Body.Close()
	raw, _ := io.ReadAll(response.Body)
	if response.StatusCode != http.StatusOK {
		t.Fatalf("/v1/decisions answered %d: %s", response.StatusCode, raw)
	}
	var answer map[string]interface{}
	if err := json.Unmarshal(raw, &answer); err != nil {
		t.Fatal(err)
	}
	return answer
}

// questionsDigest identifies the model-facing questions. Full-input admission
// never enters the model's prompt; its request flag and response proof are
// checked separately. Keep every other raw field and criteria ordering intact.
func questionsDigest(questions map[string]json.RawMessage) string {
	var digest bytes.Buffer
	for _, id := range sortedKeys(questions) {
		var compact bytes.Buffer
		question, err := sjson.DeleteBytes(questions[id], "require_full_input")
		if err != nil {
			question = questions[id]
		}
		_ = json.Compact(&compact, question)
		fmt.Fprintf(&digest, "%s=%s\n", id, compact.Bytes())
	}
	return sha256Hex(digest.Bytes())
}

// vela2Answer is one state's answer by path: every number, and every other
// value (a choice, a label, a span's text) as text.
type vela2Answer struct {
	Numbers map[string]float64
	Labels  map[string]string
}

// flattenAnswer reduces a decisions response (or one entry of its states) to
// its answers; the served model, meta and further states are not answers.
func flattenAnswer(answer map[string]interface{}) vela2Answer {
	flat := vela2Answer{Numbers: map[string]float64{}, Labels: map[string]string{}}
	var walk func(path string, value interface{})
	walk = func(path string, value interface{}) {
		switch v := value.(type) {
		case map[string]interface{}:
			for key, child := range v {
				walk(path+"/"+key, child)
			}
		case []interface{}:
			for index, child := range v {
				walk(path+"/"+strconv.Itoa(index), child)
			}
		case float64:
			flat.Numbers[path] = v
		case string:
			flat.Labels[path] = v
		case bool:
			flat.Labels[path] = strconv.FormatBool(v)
		case nil:
			flat.Labels[path] = "null"
		}
	}
	for key, value := range answer {
		if key != "model" && key != "meta" && key != "states" {
			walk(key, value)
		}
	}
	return flat
}

// recordedAnswer is an answer as the record keeps it: the response without
// the served model, meta and further states, its numbers to seven decimals
// (far below the tolerance, and stable to read).
func recordedAnswer(answer map[string]interface{}) map[string]interface{} {
	var round func(value interface{}) interface{}
	round = func(value interface{}) interface{} {
		switch v := value.(type) {
		case map[string]interface{}:
			out := make(map[string]interface{}, len(v))
			for key, child := range v {
				out[key] = round(child)
			}
			return out
		case []interface{}:
			out := make([]interface{}, len(v))
			for index, child := range v {
				out[index] = round(child)
			}
			return out
		case float64:
			return math.Round(v*1e7) / 1e7
		}
		return value
	}
	out := make(map[string]interface{}, len(answer))
	for key, value := range answer {
		if key != "model" && key != "meta" && key != "states" {
			out[key] = round(value)
		}
	}
	// Coverage is transport evidence, checked before recording. Remove only
	// the typed envelope field, never an identically named model label.
	for _, section := range []string{"answers", "sets"} {
		entries, _ := out[section].(map[string]interface{})
		for _, value := range entries {
			if entry, ok := value.(map[string]interface{}); ok {
				delete(entry, "input_coverage")
			}
		}
	}
	return out
}

// diff lists how got departs from want: a missing or extra path, a label that
// differs, or a number further than tolerance.
func (a vela2Answer) diff(want vela2Answer, tolerance float64) []string {
	var problems []string
	for _, path := range sortedKeys(want.Labels) {
		if got, ok := a.Labels[path]; !ok || got != want.Labels[path] {
			problems = append(problems, fmt.Sprintf("%s = %q, want %q", path, got, want.Labels[path]))
		}
	}
	for _, path := range sortedKeys(a.Labels) {
		if _, ok := want.Labels[path]; !ok {
			problems = append(problems, fmt.Sprintf("%s = %q is not expected", path, a.Labels[path]))
		}
	}
	for _, path := range sortedKeys(want.Numbers) {
		got, ok := a.Numbers[path]
		if !ok || math.IsNaN(got) || math.Abs(got-want.Numbers[path]) > tolerance {
			problems = append(problems, fmt.Sprintf("%s = %v, want %v ± %g", path, got, want.Numbers[path], tolerance))
		}
	}
	for _, path := range sortedKeys(a.Numbers) {
		if _, ok := want.Numbers[path]; !ok {
			problems = append(problems, fmt.Sprintf("%s = %v is not expected", path, a.Numbers[path]))
		}
	}
	return problems
}

// boundaries lists the decisions of an answer that sit within margin of
// their boundary: a Choice whose two most likely options are that close, a
// Set label or a span that close to its question's threshold.
func boundaries(answer map[string]interface{}, margin float64) []string {
	var near []string
	thresholds, _ := answer["thresholds"].(map[string]interface{})
	answers, _ := answer["answers"].(map[string]interface{})
	for _, id := range sortedKeys(answers) {
		entry, _ := answers[id].(map[string]interface{})
		if entry["type"] != "choice" {
			continue
		}
		probabilities, _ := entry["probabilities"].(map[string]interface{})
		var values []float64
		for _, p := range probabilities {
			if value, ok := p.(float64); ok {
				values = append(values, value)
			}
		}
		sort.Sort(sort.Reverse(sort.Float64Slice(values)))
		if len(values) > 1 && values[0]-values[1] < margin {
			near = append(near, fmt.Sprintf("choice %s: top options %.4f and %.4f", id, values[0], values[1]))
		}
	}
	sets, _ := answer["sets"].(map[string]interface{})
	for _, id := range sortedKeys(sets) {
		threshold, ok := thresholds[id].(float64)
		set, _ := sets[id].(map[string]interface{})
		probabilities, _ := set["probabilities"].(map[string]interface{})
		for _, label := range sortedKeys(probabilities) {
			if p, isNumber := probabilities[label].(float64); ok && isNumber && math.Abs(p-threshold) < margin {
				near = append(near, fmt.Sprintf("set %s.%s: %.4f at threshold %.4f", id, label, p, threshold))
			}
		}
	}
	spans, _ := answer["spans"].(map[string]interface{})
	for _, id := range sortedKeys(spans) {
		threshold, ok := thresholds[id].(float64)
		list, _ := spans[id].([]interface{})
		for _, item := range list {
			span, _ := item.(map[string]interface{})
			if p, isNumber := span["probability"].(float64); ok && isNumber && p-threshold < margin {
				near = append(near, fmt.Sprintf("span %s %v: %.4f at threshold %.4f", id, span["label"], p, threshold))
			}
		}
	}
	return near
}

// vela2Answers is the committed record: per size, the model it was recorded
// on and the answers of every item's states ("request", then "history:N" for
// the earlier messages the history-aware rules read), plus the hallucination
// items' answers.
type vela2Answers struct {
	Tolerance float64                       `json:"tolerance"`
	Models    map[string]*vela2ModelAnswers `json:"models"`
}

type vela2ModelAnswers struct {
	RepoID        string                      `json:"repo_id"`
	Revision      string                      `json:"revision"`
	ModelSHA256   string                      `json:"model_sha256"`
	Profile       string                      `json:"profile"`
	Items         map[string]vela2ItemAnswers `json:"items"`
	Hallucination map[string]vela2StateAnswer `json:"hallucination,omitempty"`
}

type vela2ItemAnswers struct {
	States map[string]vela2StateAnswer `json:"states"`
}

type vela2StateAnswer struct {
	QuestionsSHA256 string                 `json:"questions_sha256"`
	Answer          map[string]interface{} `json:"answer"`
}

func readVela2Answers(t *testing.T) *vela2Answers {
	t.Helper()
	recorded := &vela2Answers{Tolerance: vela2CPUTolerance, Models: map[string]*vela2ModelAnswers{}}
	raw, err := os.ReadFile(vela2AnswersPath)
	if os.IsNotExist(err) && os.Getenv(vela2RecordEnv) == "1" {
		return recorded
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(raw, recorded); err != nil {
		t.Fatalf("%s: %v", vela2AnswersPath, err)
	}
	return recorded
}

func writeVela2Answers(t *testing.T, recorded *vela2Answers) {
	t.Helper()
	encoded, err := json.MarshalIndent(recorded, "", "  ")
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(vela2AnswersPath, append(encoded, '\n'), 0o644); err != nil {
		t.Fatal(err)
	}
}

func sha256Hex(data []byte) string {
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:])
}

func sortedKeys[V any](values map[string]V) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
