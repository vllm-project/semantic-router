package extproc

import (
	"cmp"
	"context"
	"encoding/json"
	"io"
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
        schema: vllm-sr/4.0
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

// imageReadingAPI answers like a model that sees the image: 0.95 when the
// call carries one, 0.05 when it does not. It keeps each call's body.
type imageReadingAPI struct {
	mu    sync.Mutex
	calls []string
}

func (a *imageReadingAPI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		http.NotFound(w, r)
		return
	}
	raw, _ := io.ReadAll(r.Body)
	a.mu.Lock()
	a.calls = append(a.calls, string(raw))
	a.mu.Unlock()
	var body struct {
		Model     string                     `json:"model"`
		Questions map[string]json.RawMessage `json:"questions"`
	}
	_ = json.Unmarshal(raw, &body)
	noul := 0.05
	if strings.Contains(string(raw), `"images":[`) {
		noul = 0.95
	}
	answers := map[string]any{}
	for id := range body.Questions {
		answers[id] = map[string]any{"type": "noul", "noul": noul, "input_coverage": "complete"}
	}
	_ = json.NewEncoder(w).Encode(map[string]any{"model": body.Model, "answers": answers, "usage": map[string]int{"input_tokens": 3}})
}

// An image rule reads every image the backend will get, earlier turns and
// tool results included. An image no decision model decodes, a remote URL or
// a GIF, is left out: a coverage: full rule resolves it as its on_unscanned
// says, and a truncating rule reads the images it can, as it does with tool
// results it cannot read. The backend still gets the request as sent.
func TestAnImageRuleResolvesAnImageItCannotReadAsOnUnscannedSays(t *testing.T) {
	recipe := `    signals: {systemone: [{name: unsafe_image, input: {image: true}RULE, question: {type: noul, instructions: "Does the image show prohibited content?"}}]}
    decisions:
      - {name: block-unsafe-image, priority: 200, rules: {operator: AND, conditions: [{type: systemone, name: unsafe_image ON_ERROR}]}, modelRefs: [{model: deep}]}
      - {name: everything-else, priority: 10, rules: {operator: AND, conditions: []}, modelRefs: [{model: fast}]}`
	pngData := "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
	png := "data:image/png;base64," + pngData
	remote := "https://images.example.test/photo.png"
	current := func(url string) string {
		return `{"model":"vllm-sr/auto","messages":[{"role":"user","content":[{"type":"text","text":"describe this"},{"type":"image_url","image_url":{"url":"` + url + `"}}]}]}`
	}
	earlier := func(url string) string {
		return `{"model":"vllm-sr/auto","messages":[{"role":"user","content":[{"type":"text","text":"describe this"},{"type":"image_url","image_url":{"url":"` + url + `"}}]},` +
			`{"role":"assistant","content":"A landscape."},{"role":"user","content":"Tell me more."}]}`
	}
	toolResult := `{"model":"vllm-sr/auto","max_tokens":16,"messages":[{"role":"user","content":"Fetch the picture."},` +
		`{"role":"assistant","content":[{"type":"tool_use","id":"toolu_1","name":"fetch","input":{}}]},` +
		`{"role":"user","content":[{"type":"tool_result","tool_use_id":"toolu_1","content":[{"type":"image","source":{"type":"base64","media_type":"image/png","data":"` + pngData + `"}}]},{"type":"text","text":"Tell me more."}]}]}`
	gif := "data:image/gif;base64,R0lGODlhAQABAAAAACw="
	full, fullMatch := ", coverage: full", ", coverage: full, on_unscanned: match"
	for name, tc := range map[string]struct {
		path, body, rule, onError, sent, route string
		anthropic, responses                   bool
	}{
		"inline png":                       {body: current(png), sent: png, route: "deep"},
		"earlier turn png":                 {body: earlier(png), sent: png, route: "deep"},
		"tool result png":                  {path: "/v1/messages", body: toolResult, sent: pngData, route: "deep", anthropic: true},
		"generated image":                  {path: "/v1/responses", body: `{"model":"vllm-sr/auto","store":false,"input":[{"type":"image_generation_call","id":"ig_1","status":"completed","result":"` + pngData + `"},{"role":"user","content":"Tell me more."}]}`, sent: pngData, route: "deep", responses: true},
		"remote url, truncating rule":      {body: current(remote), sent: remote, route: "fast"},
		"remote url, on_unscanned match":   {body: current(remote), rule: fullMatch, sent: remote, route: "deep"},
		"gif, on_unscanned match":          {body: current(gif), rule: fullMatch, route: "deep"},
		"earlier remote url, match":        {body: earlier(remote), rule: fullMatch, sent: remote, route: "deep"},
		"remote url, unknown and on_error": {body: current(remote), rule: full, onError: ", on_error: match", sent: remote, route: "deep"},
		"remote url, unknown, no policy":   {body: current(remote), rule: full, sent: remote, route: "fast"},
	} {
		t.Run(name, func(t *testing.T) {
			api := &imageReadingAPI{}
			raw := remoteCardYAML(strings.NewReplacer("RULE", tc.rule, " ON_ERROR", tc.onError).Replace(recipe), "{question_types: [noul], modalities: [text, image]}")
			// A chat backend takes neither image tool results nor generated
			// images; a Messages or Responses one reads them.
			if tc.anthropic {
				raw = strings.ReplaceAll(raw, "      backend_refs:", "      api_format: anthropic\n      backend_refs:")
			}
			if tc.responses {
				raw = strings.ReplaceAll(raw, "      backend_refs:", "      api_format: responses\n      backend_refs:")
				raw = strings.Replace(raw, "global:\n", "global:\n  services: {response_api: {enabled: true, store_backend: memory}}\n", 1)
			}
			router := newLooperRouter(t, moduleSystemOneRouterConfig(t, raw, api))
			ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
			defer cancel()
			path := cmp.Or(tc.path, "/v1/chat/completions")
			header := routing.Header{
				{Name: ":method", Value: "POST"},
				{Name: ":path", Value: path},
				{Name: ":authority", Value: "router"},
				{Name: ":scheme", Value: "http"},
				{Name: "content-type", Value: "application/json"},
				{Name: "anthropic-version", Value: "2023-06-01"},
				{Name: "x-request-id", Value: "unread-image"},
			}
			plan, err := routing.NewEngine(router, routing.DefaultOptions).Plan(ctx, &routing.Request{Header: header, Body: []byte(tc.body)})
			if err != nil {
				t.Fatal(err)
			}
			defer plan.Finish(nil)
			if plan.Call == nil {
				t.Fatalf("the request was not dispatched: %s", plan.Immediate.Body)
			}
			if plan.Call.Route != tc.route {
				t.Fatalf("routed to %q, want %q", plan.Call.Route, tc.route)
			}
			if !strings.Contains(string(plan.Call.Request.Body), tc.sent) {
				t.Fatalf("the backend request lost the client's image: %s", plan.Call.Request.Body)
			}
			api.mu.Lock()
			defer api.mu.Unlock()
			if len(api.calls) != 1 {
				t.Fatalf("the decision model was asked %d times, want once: %v", len(api.calls), api.calls)
			}
			if strings.Contains(api.calls[0], remote) || strings.Contains(api.calls[0], gif) {
				t.Fatalf("the decision model was sent an image it cannot decode: %s", api.calls[0])
			}
		})
	}
}
