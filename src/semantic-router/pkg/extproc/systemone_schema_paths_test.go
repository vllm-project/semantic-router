package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"maps"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// A vllm-sr/5.0 listener's requests carry roles, null when the client names
// none, and a vllm-sr/4.0 deployment reads none: they reach it through its
// codec, which strips roles the state's keys imply and refuses others before
// any call, with a 400 naming the schema.
func TestAFiveListenerSpeaksToAFourDeploymentThroughItsCodec(t *testing.T) {
	var mu sync.Mutex
	var sent []map[string]json.RawMessage
	runtime := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		data, _ := io.ReadAll(r.Body)
		var body map[string]json.RawMessage
		if err := json.Unmarshal(data, &body); err != nil {
			t.Error(err)
		}
		mu.Lock()
		sent = append(sent, body)
		mu.Unlock()
		w.Header().Set("Content-Type", "application/json")
		if _, roles := body["roles"]; roles {
			w.WriteHeader(http.StatusBadRequest)
			fmt.Fprint(w, `{"error":{"code":"invalid_request","message":"unknown member: roles"}}`)
			return
		}
		fmt.Fprint(w, `{"model":"vllm-sr/d3-ultra","answers":{"tool":{"type":"noul","noul":0.25}},"usage":{"input_tokens":5,"output_tokens":0}}`)
	}))
	defer runtime.Close()
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners:
  - name: native
    port: 8801
    apis: {chat_completions: {}, responses: {}, messages: {}, images: {}, systemone: {schema: vllm-sr/5.0, models: [legacy]}}
providers:
  models:
    - name: legacy
      api_format: systemone
      deployment: legacy-4
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
recipes:
  - name: default
global:
  model_catalog:
    deployments:
      legacy-4:
        provider: systemone
        schema: vllm-sr/4.0
        endpoint: %s/v1/systemone
        model: vllm-sr/d3-ultra
`, runtime.URL)))
	if err != nil {
		t.Fatal(err)
	}
	remote, err := modelservice.NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = remote.Close() }()
	models := &FrontendModels{remote: remote}
	local := func(ctx context.Context, deployment string, body json.RawMessage) (int, []byte, error) {
		result, err := models.SystemOne(ctx, deployment, body)
		return result.Status, result.Body, err
	}
	listener := &cfg.Listeners[0]
	handler := systemone.Handler(cfg, listener, systemone.ServingInvoker(cfg, listener.Name, nil, local, nil, models))
	const question = `"questions":{"tool":{"type":"noul","instructions":"Does the request need a tool?"}}`
	for _, tc := range []struct {
		name, request, state string
		status               int
		answer               []string
	}{
		{
			name:    "an object state without roles",
			request: `{"model":"legacy","state":{"request":"Book a flight to Lisbon","tools":"[search_flights]"},` + question + `}`,
			status:  http.StatusBadRequest, answer: []string{`"code":"invalid_request"`, `"reason":"roles_unsupported"`, "speaks vllm-sr/4.0"},
		},
		{
			name:    "roles the state's keys imply",
			request: `{"model":"legacy","state":{"request":"Book a flight to Lisbon","tools":"[search_flights]"},"roles":{"request":"user","tools":"context"},` + question + `}`,
			state:   `{"request":"Book a flight to Lisbon","tools":"[search_flights]"}`,
			status:  http.StatusOK, answer: []string{`"noul":0.25`},
		},
		{
			name:    "a string state",
			request: `{"model":"legacy","state":"Book a flight to Lisbon",` + question + `}`,
			state:   `"Book a flight to Lisbon"`,
			status:  http.StatusOK, answer: []string{`"noul":0.25`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			mu.Lock()
			sent = nil
			mu.Unlock()
			w := httptest.NewRecorder()
			handler(w, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(tc.request)))
			if w.Code != tc.status {
				t.Fatalf("status %d, body %s", w.Code, w.Body.String())
			}
			for _, want := range tc.answer {
				if !strings.Contains(w.Body.String(), want) {
					t.Fatalf("body %s lacks %s", w.Body.String(), want)
				}
			}
			mu.Lock()
			defer mu.Unlock()
			if tc.state == "" {
				if len(sent) != 0 {
					t.Fatalf("a refused request reached the runtime: %v", sent)
				}
				return
			}
			if len(sent) != 1 || string(sent[0]["state"]) != tc.state || string(sent[0]["model"]) != `"vllm-sr/d3-ultra"` {
				t.Fatalf("sent %v, want one call with state %s", sent, tc.state)
			}
			if _, roles := sent[0]["roles"]; roles {
				t.Fatalf("roles reached a vllm-sr/4.0 runtime: %s", sent[0]["roles"])
			}
		})
	}
}

// fiveUpstream answers as a vllm-sr/5.0 runtime's /v1/systemone and
// /v1/systemone/batch do: a request with a choice question without criteria
// fails as a whole, naming the first such question (and its entry); every
// other question answers.
func fiveUpstream(t *testing.T) *httptest.Server {
	t.Helper()
	type request struct {
		Model     string                                         `json:"model"`
		Questions map[string]struct{ Criteria *json.RawMessage } `json:"questions"`
	}
	invalid := func(questions map[string]struct{ Criteria *json.RawMessage }) string {
		names := slices.Sorted(maps.Keys(questions))
		for _, name := range names {
			if criteria := questions[name].Criteria; criteria != nil && string(*criteria) == "{}" {
				return name
			}
		}
		return ""
	}
	answers := func(questions map[string]struct{ Criteria *json.RawMessage }) map[string]any {
		answered := map[string]any{}
		for name := range questions {
			answered[name] = map[string]any{"type": "noul", "noul": 0.5}
		}
		return answered
	}
	fail := func(w http.ResponseWriter, question, state string) {
		w.WriteHeader(http.StatusUnprocessableEntity)
		failure := map[string]any{
			"detail":   []map[string]any{{"loc": []string{"body", "questions", question}, "msg": "criteria must name 2 to 255 options", "type": "invalid_question"}},
			"code":     "question_failed",
			"question": question, "reason": "question_malformed",
		}
		if state != "" {
			failure["state"] = state
		}
		_ = json.NewEncoder(w).Encode(failure)
	}
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		switch r.URL.Path {
		case "/v1/systemone":
			var body request
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			if question := invalid(body.Questions); question != "" {
				fail(w, question, "")
				return
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"model": body.Model, "answers": answers(body.Questions), "usage": map[string]int{"input_tokens": 3, "output_tokens": 0}})
		case "/v1/systemone/batch":
			var body struct {
				Model    string             `json:"model"`
				Requests map[string]request `json:"requests"`
			}
			if err := json.NewDecoder(r.Body).Decode(&body); err != nil {
				t.Error(err)
			}
			responses := map[string]any{}
			for _, name := range slices.Sorted(maps.Keys(body.Requests)) {
				if question := invalid(body.Requests[name].Questions); question != "" {
					fail(w, question, name)
					return
				}
				responses[name] = map[string]any{"model": body.Model, "answers": answers(body.Requests[name].Questions), "usage": map[string]int{"input_tokens": 3, "output_tokens": 0}}
			}
			_ = json.NewEncoder(w).Encode(map[string]any{"model": body.Model, "responses": responses, "usage": map[string]int{"input_tokens": 3 * len(responses), "output_tokens": 0}})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	t.Cleanup(server.Close)
	return server
}

// The requests with no valid question the model runtime's tests share
// (config/schemas/systemone/no-valid-question.json) answer from the Router,
// through a vllm-sr/5.0 upstream it asks question by question, as the runtime
// answers them on /v1/decisions.
func TestARequestWithoutAValidQuestionAnswersAsTheRuntimeAnswersIt(t *testing.T) {
	data, err := os.ReadFile(filepath.Join("..", "..", "..", "..", "config", "schemas", "systemone", "no-valid-question.json"))
	if err != nil {
		t.Fatal(err)
	}
	var fixture struct {
		Cases []struct {
			Name    string                       `json:"name"`
			Request map[string]json.RawMessage   `json:"request"`
			Status  int                          `json:"status"`
			Error   map[string]string            `json:"error"`
			Failed  map[string]map[string]string `json:"failed"`
		} `json:"cases"`
	}
	if err = json.Unmarshal(data, &fixture); err != nil || len(fixture.Cases) == 0 {
		t.Fatalf("the fixture: %v", err)
	}
	upstreamServer := fiveUpstream(t)
	cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners:
  - name: native
    port: 8801
    apis: {chat_completions: {}, responses: {}, messages: {}, images: {}, systemone: {schema: vllm-sr/4.0, models: [five]}}
providers:
  models:
    - name: five
      api_format: systemone
      schema: vllm-sr/5.0
      provider_model_id: five-native
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
recipes:
  - name: default
`, upstreamServer.URL)))
	if err != nil {
		t.Fatal(err)
	}
	remote, err := upstream.Build(cfg, upstream.Options{})
	if err != nil {
		t.Fatal(err)
	}
	defer remote.Close(context.Background())
	listener := &cfg.Listeners[0]
	handler := systemone.Handler(cfg, listener, systemone.ServingInvoker(cfg, listener.Name, nil, nil, remote, nil))
	for _, tc := range fixture.Cases {
		t.Run(tc.Name, func(t *testing.T) {
			tc.Request["model"] = json.RawMessage(`"five"`)
			body, _ := json.Marshal(tc.Request)
			w := httptest.NewRecorder()
			handler(w, httptest.NewRequest(http.MethodPost, "/v1/decisions", strings.NewReader(string(body))))
			if w.Code != tc.Status {
				t.Fatalf("status %d, want %d: %s", w.Code, tc.Status, w.Body.String())
			}
			var answer struct {
				Error  map[string]any `json:"error"`
				States map[string]struct {
					Answers map[string]map[string]any `json:"answers"`
				} `json:"states"`
			}
			if err := json.Unmarshal(w.Body.Bytes(), &answer); err != nil {
				t.Fatal(err)
			}
			for key, want := range tc.Error {
				if answer.Error[key] != want {
					t.Fatalf("error %s is %v, want %q: %s", key, answer.Error[key], want, w.Body.String())
				}
			}
			for state, failed := range tc.Failed {
				for question, reason := range failed {
					if got := answer.States[state].Answers[question]; got["error"] != "invalid_question" || got["reason"] != reason {
						t.Fatalf("states.%s.answers.%s is %v, want invalid_question %s", state, question, got, reason)
					}
				}
			}
		})
	}
}
