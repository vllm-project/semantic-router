package extproc

import (
	"context"
	"encoding/json"
	"net/http"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// remoteCardConfig asks hosted-d3, a systemone deployment that is the
// decision model, what RECIPE's signals and decisions need. CAPABILITIES is
// the line of what the deployment declares, or empty.
const remoteCardConfig = `
version: v0.5
providers:
  defaults:
    model: fast
  models:
    - name: fast
      backend_refs: [{name: fast, endpoint: "BACKEND", protocol: http, weight: 1}]
    - name: deep
      backend_refs: [{name: deep, endpoint: "BACKEND", protocol: http, weight: 1}]
recipes:
  - name: default
RECIPE
global:
  model_catalog:
    deployments:
      hosted-d3:
        provider: systemone
        endpoint: API/v1/systemone
        model: vllm-sr/d3-ultra
        auth: {api_key_env: REMOTE_DECISION_TEST_KEY}
CAPABILITIES
    system:
      decision_model: {deployment: hosted-d3}
`

// remoteCardAPI is hosted-d3's System One API. It answers GET
// /v1/systemone/models with listing (not found when empty), the pii preset
// question with one email address, and every other question with a noul of
// 0.9; it keeps each question it is asked.
type remoteCardAPI struct {
	listing   string
	mu        sync.Mutex
	questions []string
}

func (a *remoteCardAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		if a.listing == "" {
			http.NotFound(w, r)
			return
		}
		_, _ = w.Write([]byte(a.listing))
		return
	}
	var body struct {
		Model     string                     `json:"model"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Questions == nil {
		http.Error(w, `{"error":{"code":"invalid_request"}}`, http.StatusBadRequest)
		return
	}
	answers, spans := map[string]any{}, map[string]any{}
	for id, question := range body.Questions {
		a.mu.Lock()
		a.questions = append(a.questions, string(question))
		a.mu.Unlock()
		answers[id] = map[string]any{"type": "noul", "noul": 0.9, "input_coverage": "complete"}
		if strings.Contains(string(question), `"preset":"pii"`) {
			spans[id] = []map[string]any{{"label": "EMAIL_ADDRESS", "start": 12, "end": 30, "text": "alice@example.test", "probability": 0.99}}
		}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"model": body.Model, "answers": answers, "spans": spans, "usage": map[string]int{"input_tokens": 12}})
}

func (a *remoteCardAPI) asked() []string {
	a.mu.Lock()
	defer a.mu.Unlock()
	return slices.Clone(a.questions)
}

func remoteCardYAML(recipe, capabilities string) string {
	if capabilities != "" {
		capabilities = "        capabilities: " + capabilities
	}
	return strings.NewReplacer("RECIPE", recipe, "CAPABILITIES", capabilities).Replace(remoteCardConfig)
}

// A remote card that lists pii, as the deployment declares it or as its API
// lists it, answers a PII rule with the model's own pii question.
func TestARemoteDeploymentAnswersPIIWithThePresetItsCardLists(t *testing.T) {
	recipe := `    signals: {pii: [{name: personal}]}
    decisions:
      - {name: private, priority: 200, rules: {operator: AND, conditions: [{type: pii, name: personal}]}, modelRefs: [{model: deep}]}
      - {name: everything-else, priority: 10, rules: {operator: AND, conditions: []}, modelRefs: [{model: fast}]}`
	for name, tc := range map[string]struct{ capabilities, listing string }{
		"declared": {capabilities: "{question_types: [choice, noul, score], presets: [pii, halu]}"},
		"listed":   {listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["choice","noul","score"],"presets":["pii","halu"]}]}`},
	} {
		t.Run(name, func(t *testing.T) {
			api := &remoteCardAPI{listing: tc.listing}
			router := newLooperRouter(t, moduleSystemOneRouterConfig(t, remoteCardYAML(recipe, tc.capabilities), api))
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			header := routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: "/v1/chat/completions"},
				{Name: ":authority", Value: "router"},
				{Name: ":scheme", Value: "http"},
				{Name: "content-type", Value: "application/json"},
				{Name: "x-request-id", Value: "remote-card-" + name},
			}
			plan, err := routing.NewEngine(router, routing.DefaultOptions).Plan(ctx, &routing.Request{Header: header, Body: []byte(
				`{"model":"vllm-sr/auto","messages":[{"role":"user","content":"reach me at alice@example.test"}]}`)})
			if err != nil {
				t.Fatal(err)
			}
			defer plan.Finish(nil)
			asked := api.asked()
			if len(asked) != 1 || !strings.Contains(asked[0], `"preset":"pii"`) {
				t.Fatalf("hosted-d3 was asked %v, want its pii preset", asked)
			}
			if plan.Call == nil || plan.Call.Route != "deep" {
				t.Fatalf("personal data routes to deep, got %+v", plan.Call)
			}
		})
	}
}

// A remote deployment that declares nothing is checked against its API's
// listing when the generation prepares: a question the listed model does not
// answer fails the generation, as a declared mismatch fails the load.
func TestAGenerationFailsOnAQuestionItsRemoteListingDoesNotAnswer(t *testing.T) {
	for name, tc := range map[string]struct{ recipe, listed, want string }{
		"a systemone rule": {
			recipe: `    signals: {systemone: [{name: needs_tools, question: {type: noul, instructions: "Does the request need a tool?"}}]}
    decisions:
      - {name: tools, priority: 200, rules: {operator: AND, conditions: [{type: systemone, name: needs_tools}]}, modelRefs: [{model: deep}]}`,
			listed: "choice",
			want:   `question_type_unsupported: resolves to deployment "hosted-d3", which answers choice questions and cannot answer noul questions`,
		},
		"a systemone selector": {
			recipe: `    decisions:
      - name: choose
        priority: 100
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: fast}, {model: deep}]
        algorithm: {type: systemone, systemone: {deployment: hosted-d3, instructions: "Which model should answer?", candidates: {fast: Quick answers, deep: Careful reasoning}}}`,
			listed: "noul",
			want:   "prepare decision selector choose: model runtime rejected the request: model vllm-sr/d3-ultra does not support choice",
		},
	} {
		t.Run(name, func(t *testing.T) {
			api := &remoteCardAPI{listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["` + tc.listed + `"]}]}`}
			cfg := moduleSystemOneRouterConfig(t, remoteCardYAML(tc.recipe, ""), api)
			router, err := buildOpenAIRouterFromConfig(cfg)
			if err == nil {
				_ = router.Close()
				t.Fatalf("the generation prepared, want %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("error %v, want %q", err, tc.want)
			}
		})
	}
}

// A remote deployment reads the input modalities its API lists for the model,
// with or without question types, else those it declares, else text, and
// never audio: a rule reading a medium the deployment does not read fails the
// generation as a declared mismatch fails the load, and one it reads prepares.
func TestARemoteDeploymentReadsOnlyTheModalitiesItsCardLists(t *testing.T) {
	image := `    signals: {systemone: [{name: chart, input: {image: true}, question: {type: noul, instructions: "Is the image a chart?"}}]}
    decisions:
      - {name: charts, priority: 200, rules: {operator: AND, conditions: [{type: systemone, name: chart}]}, modelRefs: [{model: deep}]}`
	audio := `    signals: {systemone: [{name: voice, input: {audio: true}, question: {type: noul, instructions: "Is the speaker angry?"}}]}
    decisions:
      - {name: angry, priority: 200, rules: {operator: AND, conditions: [{type: systemone, name: voice}]}, modelRefs: [{model: deep}]}`
	const refusedImage = `input_modality_unsupported: input.image resolves to deployment "hosted-d3", which does not read image input`
	const refusedAudio = `input_modality_unsupported: input.audio resolves to deployment "hosted-d3", which does not read audio input`
	for name, tc := range map[string]struct{ recipe, capabilities, listing, want string }{
		"an image rule, listed text only": {
			recipe: image, listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["noul"],"modalities":["text"]}]}`, want: refusedImage,
		},
		"an image rule, listed text only without question types over declared image": {
			recipe: image, capabilities: "{modalities: [text, image]}", listing: `{"data":[{"id":"vllm-sr/d3-ultra","modalities":["text"]}]}`, want: refusedImage,
		},
		"an image rule, unlisted and declaring nothing":  {recipe: image, want: refusedImage},
		"an image rule, unlisted and declaring no image": {recipe: image, capabilities: "{question_types: [noul]}", want: refusedImage},
		"an audio rule, unlisted and declaring nothing":  {recipe: audio, want: refusedAudio},
		"an audio rule, listed with audio": {
			recipe: audio, listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["noul"],"modalities":["text","audio"]}]}`, want: refusedAudio,
		},
		"an image rule, listed with image": {
			recipe: image, listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["noul"],"modalities":["text","image"]}]}`,
		},
		"an image rule, unlisted and declaring image": {recipe: image, capabilities: "{modalities: [text, image]}"},
		"an image rule, listed without modalities over declared image": {
			recipe: image, capabilities: "{modalities: [text, image]}", listing: `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["noul"]}]}`,
		},
	} {
		t.Run(name, func(t *testing.T) {
			api := &remoteCardAPI{listing: tc.listing}
			router, err := buildOpenAIRouterFromConfig(moduleSystemOneRouterConfig(t, remoteCardYAML(tc.recipe, tc.capabilities), api))
			if err == nil {
				_ = router.Close()
			}
			switch {
			case tc.want == "" && err != nil:
				t.Fatalf("the generation failed with %v, want it prepared", err)
			case tc.want != "" && (err == nil || !strings.Contains(err.Error(), tc.want)):
				t.Fatalf("error %v, want %q", err, tc.want)
			}
		})
	}
}
