package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemone"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// A box question to vllm-sr/auto goes only to the pool member whose backend
// lists box questions and images; when no member does, the request is
// refused before any call, with a detail naming what is missing and every
// member.
func TestSystemOneAutoRoutesByWhatMembersList(t *testing.T) {
	for _, tc := range []struct {
		name       string
		visionCard string
		request    string
		status     int
		posts      []string
		body       []string
	}{
		{
			name:       "a mixed pool",
			visionCard: `{"id":"vision-native","surfaces":["decisions"],"question_types":["choice","noul","score","box"],"modalities":["text","image"]}`,
			status:     http.StatusOK, posts: []string{"vision-native"}, body: []string{`"selected_model":"vision"`},
		},
		{
			name:       "no capable member",
			visionCard: `{"id":"vision-native","surfaces":["decisions"],"question_types":["choice","noul","score"],"modalities":["text","image"]}`,
			status:     http.StatusBadRequest,
			body:       []string{"unsupported_capability", `no candidate model of decision \"answer\"`, "needs box, image", "text lacks box, image (listing)", "vision lacks box (listing)"},
		},
		{
			name:       "a question type auto does not route",
			visionCard: `{"id":"vision-native","surfaces":["decisions"],"question_types":["choice","noul","score","set"]}`,
			request:    `{"model":"vllm-sr/auto","state":"Mail me at a@example.com","questions":{"tags":{"type":"set","instructions":"Which apply?","criteria":{"pii":"Personal data"}}}}`,
			status:     http.StatusBadRequest,
			body:       []string{"invalid_request", `auto does not support question type \"set\"`},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var mu sync.Mutex
			var posts []string
			backend := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.Header().Set("Content-Type", "application/json")
				switch {
				case r.Method == http.MethodGet && r.URL.Path == "/v1/models":
					fmt.Fprintf(w, `{"models":[`+
						`{"name":"text-native","description":"","release_date":"","version":"5.0","features":["choice","noul","score"]},`+
						`{"name":"vision-native","description":"","release_date":"","version":"5.0","card":%s}]}`, tc.visionCard)
				case r.Method == http.MethodPost && r.URL.Path == "/v1/systemone":
					var request struct {
						Model string `json:"model"`
					}
					if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
						t.Error(err)
					}
					mu.Lock()
					posts = append(posts, request.Model)
					mu.Unlock()
					fmt.Fprintf(w, `{"model":%q,"answers":{"where":{"type":"box","box":{"top":0.1,"bottom":0.4,"left":0.2,"right":0.5},"confidence":0.8}},"usage":{"input_tokens":30,"output_tokens":0}}`, request.Model)
				default:
					t.Errorf("unexpected %s %s", r.Method, r.URL.Path)
					w.WriteHeader(http.StatusNotFound)
				}
			}))
			defer backend.Close()
			cfg, err := config.ParseYAMLBytes([]byte(fmt.Sprintf(`version: v0.5
listeners:
  - name: native
    port: 8801
    apis: {chat_completions: {}, responses: {}, messages: {}, images: {}, systemone: {schema: vllm-sr/4.0, models: [vllm-sr/auto]}}
providers:
  models:
    - name: text
      api_format: systemone
      schema: vllm-sr/5.0
      provider_model_id: text-native
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
    - name: vision
      api_format: systemone
      schema: vllm-sr/5.0
      provider_model_id: vision-native
      backend_refs: [{provider: systemone-compatible, endpoint: %s/v1}]
entrypoints:
  - model_names: [vllm-sr/auto]
    recipe: default
    default: true
  - {api: systemone, model_names: [vllm-sr/auto], recipe: native}
recipes:
  - name: default
  - name: native
    decisions:
      - name: answer
        rules: {}
        modelRefs: [{model: text}, {model: vision}]
        algorithm:
          type: escalate
          budget: {deadline: 3s, max_calls: 2}
          escalate:
            stages:
              - {name: text, model: text}
              - {name: vision, model: vision}
`, backend.URL, backend.URL)))
			if err != nil {
				t.Fatal(err)
			}
			classifiers, err := classification.BuildRecipeClassifiers(cfg)
			if err != nil {
				t.Fatal(err)
			}
			executors, err := prepareNativeExecutors(cfg)
			if err != nil {
				t.Fatal(err)
			}
			router := &OpenAIRouter{Config: cfg, RecipeClassifiers: classifiers, nativeExecutors: executors}
			remote, err := upstream.Build(cfg, upstream.Options{})
			if err != nil {
				t.Fatal(err)
			}
			defer remote.Close(context.Background())
			listener := &cfg.Listeners[0]
			handler := systemone.Handler(cfg, listener, systemone.ServingInvoker(cfg, listener.Name, router, nil, remote, &FrontendModels{}))
			body := tc.request
			if body == "" {
				body = `{"model":"vllm-sr/auto","state":"Where is the logo?","images":["data:image/png;base64,iVBORw0KGgo="],"questions":{"where":{"type":"box","instructions":"Find the logo."}}}`
			}
			w := httptest.NewRecorder()
			handler(w, httptest.NewRequest(http.MethodPost, "/v1/systemone", strings.NewReader(body)))
			if w.Code != tc.status {
				t.Fatalf("status %d, body %s", w.Code, w.Body.String())
			}
			for _, want := range tc.body {
				if !strings.Contains(w.Body.String(), want) {
					t.Fatalf("body %s lacks %s", w.Body.String(), want)
				}
			}
			mu.Lock()
			defer mu.Unlock()
			if !reflect.DeepEqual(posts, tc.posts) {
				t.Fatalf("posts %v, want %v", posts, tc.posts)
			}
		})
	}
}
