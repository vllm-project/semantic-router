package extproc

import (
	"context"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/authz"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// clientAuthTestConfig publishes company/fast and company/public on one
// recipe. The tests give company/fast its client keys in code, since the
// parser does not accept client_auth yet.
const clientAuthTestConfig = `
version: v0.5
providers:
  defaults:
    model: worker
  models:
    - name: worker
      provider_model_id: worker-prod
      api_format: openai
      backend_refs:
        - provider: vllm
          endpoint: http://127.0.0.1:18000/v1
          api_key: provider-secret
recipes:
  - name: default
    decisions:
      - name: answer
        rules: {operator: AND, conditions: []}
        modelRefs: [{model: worker}]
entrypoints:
  - model_names: [company/fast]
    recipe: default
  - model_names: [company/public]
    recipe: default
global:
  router:
    skip_processing: {enabled: true}
`

// countingResponseStore counts the reads of retained Responses objects.
type countingResponseStore struct {
	*MockResponseStore
	reads int
}

func (s *countingResponseStore) GetResponse(ctx context.Context, id string) (*responseapi.StoredResponse, error) {
	s.reads++
	return s.MockResponseStore.GetResponse(ctx, id)
}

func (s *countingResponseStore) GetConversationChain(ctx context.Context, id string) ([]*responseapi.StoredResponse, error) {
	s.reads++
	return s.MockResponseStore.GetConversationChain(ctx, id)
}

// clientAuthGeneration is a router and the snapshot it serves, company/fast
// taking the keys envs hold.
func clientAuthGeneration(t *testing.T, store *countingResponseStore, envs ...string) (*OpenAIRouter, *configsnapshot.Snapshot) {
	t.Helper()
	return clientAuthGenerationFrom(t, clientAuthTestConfig, store, envs...)
}

func clientAuthGenerationFrom(t *testing.T, document string, store *countingResponseStore, envs ...string) (*OpenAIRouter, *configsnapshot.Snapshot) {
	t.Helper()
	cfg, err := config.ParseYAMLBytes([]byte(document))
	require.NoError(t, err)
	if len(envs) > 0 {
		cfg.Entrypoints[0].ClientAuth = &config.EntrypointClientAuth{Type: config.ClientAuthBearerAPIKey, APIKeyEnvs: envs}
	}
	classifier, err := classification.NewClassifier(cfg)
	require.NoError(t, err)
	router := &OpenAIRouter{
		Config:             cfg,
		Classifier:         classifier,
		Cache:              cache.NewInMemoryCache(cache.InMemoryCacheOptions{Enabled: false}),
		CredentialResolver: authz.NewCredentialResolver(authz.NewStaticConfigProvider(cfg)),
		ResponseAPIFilter:  NewResponseAPIFilter(store),
	}
	snapshot, err := configsnapshot.NewManager(configsnapshot.Options{}).Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: cfg,
	})
	require.NoError(t, err)
	return router, snapshot
}

// clientAuthService serves a generation whose company/fast takes FAST_KEY.
func clientAuthService(t *testing.T, store *countingResponseStore, envs ...string) *RouterService {
	t.Helper()
	router, snapshot := clientAuthGeneration(t, store, envs...)
	service := NewRouterServiceForSnapshot(router, snapshot)
	t.Cleanup(func() { _ = service.Close() })
	return service
}

func clientAuthRequest(path string, header map[string]string, body string) []*ext_proc.ProcessingRequest {
	values := []*core.HeaderValue{
		{Key: ":method", Value: "POST"},
		{Key: ":path", Value: path},
		{Key: "content-type", Value: "application/json"},
	}
	for key, value := range header {
		values = append(values, &core.HeaderValue{Key: key, Value: value})
	}
	return []*ext_proc.ProcessingRequest{
		{Request: &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: values}}}},
		{Request: &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{Body: []byte(body), EndOfStream: true}}},
	}
}

// reloadingStream runs reload once the request headers are answered and
// before the body arrives, as a reload does while a slow body uploads.
type reloadingStream struct {
	*MockStream
	reload func()
}

func (s *reloadingStream) Recv() (*ext_proc.ProcessingRequest, error) {
	if s.RecvIndex == 1 && s.reload != nil {
		s.reload()
		s.reload = nil
	}
	return s.MockStream.Recv()
}

func processClientAuthRequest(t *testing.T, service *RouterService, requests []*ext_proc.ProcessingRequest, reload func()) *ext_proc.ProcessingResponse {
	t.Helper()
	stream := &reloadingStream{MockStream: NewMockStream(requests), reload: reload}
	require.NoError(t, service.Process(stream))
	require.Len(t, stream.Responses, 2)
	return stream.Responses[1]
}

// requireRefusal checks the status and the message, which every wire format
// carries, unlike the code.
func requireRefusal(t *testing.T, response *ext_proc.ProcessingResponse, status int, message string) {
	t.Helper()
	immediate := response.GetImmediateResponse()
	require.NotNil(t, immediate, "the request was not refused: %v", response)
	require.EqualValues(t, status, immediate.GetStatus().GetCode())
	assert.Contains(t, string(immediate.GetBody()), message)
}

const (
	refusedKey        = "The model takes an API key"
	refusedSuperseded = "configuration changed while the request arrived"
)

func requireDispatched(t *testing.T, response *ext_proc.ProcessingResponse) *ext_proc.CommonResponse {
	t.Helper()
	dispatch := response.GetRequestBody().GetResponse()
	require.NotNil(t, dispatch, "the request was not dispatched: %v", response.GetImmediateResponse())
	return dispatch
}

// A protected entrypoint admits a request only with one of its keys, as a
// Bearer token, on every inference API that names it. A refused request reads
// no retained object and is never dispatched.
func TestAProtectedEntrypointAdmitsOnlyItsKey(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	store := &countingResponseStore{MockResponseStore: NewMockResponseStore()}
	service := clientAuthService(t, store, "FAST_KEY")

	for _, api := range []struct {
		name, path, body string
	}{
		{"chat completions", "/v1/chat/completions", `{"model":"company/fast","messages":[{"role":"user","content":"hi"}]}`},
		{"untrimmed model", "/v1/chat/completions", `{"model":" company/fast ","messages":[{"role":"user","content":"hi"}]}`},
		{"responses with history", "/v1/responses", `{"model":"company/fast","input":"hi","previous_response_id":"resp_other"}`},
		{"messages", "/v1/messages", `{"model":"company/fast","max_tokens":16,"messages":[{"role":"user","content":"hi"}]}`},
		{"count tokens", "/v1/messages/count_tokens", `{"model":"company/fast","messages":[{"role":"user","content":"hi"}]}`},
		{"azure deployment", "/openai/deployments/company/fast/chat/completions?api-version=2024-10-21", `{"messages":[{"role":"user","content":"hi"}]}`},
		{"azure deployment, body naming a public model", "/openai/deployments/company/fast/chat/completions?api-version=2024-10-21", `{"model":"company/public","messages":[{"role":"user","content":"hi"}]}`},
	} {
		for _, client := range []struct {
			name   string
			header map[string]string
		}{
			{"no key", nil},
			{"another key", map[string]string{"authorization": "Bearer fast-2"}},
			{"the key without its scheme", map[string]string{"authorization": "fast-1"}},
			{"the key as api-key", map[string]string{"api-key": "fast-1"}},
			{"the key as x-api-key", map[string]string{"x-api-key": "fast-1"}},
		} {
			t.Run(api.name+"/"+client.name, func(t *testing.T) {
				response := processClientAuthRequest(t, service, clientAuthRequest(api.path, client.header, api.body), nil)
				requireRefusal(t, response, 401, refusedKey)
				assert.Contains(t, headerValuesByName(response.GetImmediateResponse().GetHeaders().GetSetHeaders()), "WWW-Authenticate")
			})
		}
	}
	assert.Zero(t, store.reads, "a refused request read retained history")

	t.Run("the key", func(t *testing.T) {
		response := processClientAuthRequest(t, service, clientAuthRequest("/v1/chat/completions",
			map[string]string{"authorization": "Bearer fast-1"},
			`{"model":"company/fast","messages":[{"role":"user","content":"hi"}]}`), nil)
		dispatch := requireDispatched(t, response)
		emitted := headerValuesByName(dispatch.GetHeaderMutation().GetSetHeaders())
		assert.Equal(t, "Bearer provider-secret", emitted["Authorization"], "the client key reached the provider")
	})
}

// A public entrypoint needs no key, and a key it does not know grants nothing
// and takes nothing away.
func TestAPublicEntrypointNeedsNoKey(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	service := clientAuthService(t, &countingResponseStore{MockResponseStore: NewMockResponseStore()}, "FAST_KEY")
	for _, header := range []map[string]string{nil, {"authorization": "Bearer placeholder"}} {
		response := processClientAuthRequest(t, service, clientAuthRequest("/v1/chat/completions", header,
			`{"model":"company/public","messages":[{"role":"user","content":"hi"}]}`), nil)
		requireDispatched(t, response)
	}
}

// A refused request leaves no trace of routing work on its context: no recipe
// resolved, no decision, no translation target and no retained object.
func TestARefusedRequestDoesNoRoutingWork(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	router, snapshot := clientAuthGeneration(t, &countingResponseStore{MockResponseStore: NewMockResponseStore()}, "FAST_KEY")
	service := NewRouterServiceForSnapshot(router, snapshot)
	t.Cleanup(func() { _ = service.Close() })

	requests := clientAuthRequest("/v1/responses", nil, `{"model":"company/fast","input":"hi","previous_response_id":"resp_other"}`)
	ctx := &RequestContext{Headers: map[string]string{}, TraceContext: context.Background()}
	_, err := router.handleRequestHeaders(requests[0].Request.(*ext_proc.ProcessingRequest_RequestHeaders), ctx)
	require.NoError(t, err)
	response, err := router.handleRequestBody(requests[1].Request.(*ext_proc.ProcessingRequest_RequestBody), ctx)
	require.NoError(t, err)
	requireRefusal(t, response, 401, refusedKey)

	assert.Equal(t, routingUnresolved, ctx.Routing.resolution, "a refused request selected a recipe")
	assert.Nil(t, ctx.VSRSelectedDecision)
	assert.Nil(t, ctx.SemanticRequest)
	assert.Nil(t, ctx.ResponseObjectState)
	assert.Empty(t, ctx.TargetFormat)
	assert.Empty(t, ctx.RequestModel)
}

// A request admitted in process, as a hop of a request already admitted,
// carries no client key of its own.
func TestAHopNeedsNoClientKey(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	router, snapshot := clientAuthGeneration(t, &countingResponseStore{MockResponseStore: NewMockResponseStore()}, "FAST_KEY")
	service := NewRouterServiceForSnapshot(router, snapshot)
	t.Cleanup(func() { _ = service.Close() })

	assert.NotNil(t, router.clientAdmissionRefusal(&RequestContext{Headers: map[string]string{}}, "company/fast"))
	assert.Nil(t, router.clientAdmissionRefusal(&RequestContext{Headers: map[string]string{}, Hop: &routing.Hop{Iteration: 1}}, "company/fast"))
}

// A stream leased before its request names a model authorizes nothing: a
// reload that changes how the model is protected while the body arrives
// refuses the request, and the next request follows the new policy. A reload
// that leaves the model's policy alone does not disturb it.
func TestAReloadWhileTheBodyArrivesDecidesAdmission(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	t.Setenv("FAST_KEY_NEXT", "fast-2")
	store := &countingResponseStore{MockResponseStore: NewMockResponseStore()}
	chat := func(header map[string]string) []*ext_proc.ProcessingRequest {
		return clientAuthRequest("/v1/chat/completions", header, `{"model":"company/fast","messages":[{"role":"user","content":"hi"}]}`)
	}
	reloadTo := func(service *RouterService, envs ...string) func() {
		return func() {
			router, snapshot := clientAuthGeneration(t, store, envs...)
			require.NoError(t, service.swapSnapshot(router, snapshot, nil))
		}
	}
	key := map[string]string{"authorization": "Bearer fast-1"}

	for _, tc := range []struct {
		name             string
		served, reloaded []string
		header           map[string]string
	}{
		{"protected while arriving", nil, []string{"FAST_KEY"}, nil},
		{"protected while arriving, with its key", nil, []string{"FAST_KEY"}, key},
		{"made public while arriving", []string{"FAST_KEY"}, nil, key},
		{"keys rotated while arriving", []string{"FAST_KEY"}, []string{"FAST_KEY_NEXT"}, key},
	} {
		t.Run(tc.name, func(t *testing.T) {
			service := clientAuthService(t, store, tc.served...)
			response := processClientAuthRequest(t, service, chat(tc.header), reloadTo(service, tc.reloaded...))
			requireRefusal(t, response, 409, refusedSuperseded)
		})
	}

	t.Run("the next request follows the new policy", func(t *testing.T) {
		service := clientAuthService(t, store)
		reloadTo(service, "FAST_KEY")()
		requireRefusal(t, processClientAuthRequest(t, service, chat(nil), nil), 401, refusedKey)
		requireDispatched(t, processClientAuthRequest(t, service, chat(key), nil))
	})

	t.Run("a reload that keeps the model's policy", func(t *testing.T) {
		service := clientAuthService(t, store, "FAST_KEY")
		requireDispatched(t, processClientAuthRequest(t, service, chat(key), reloadTo(service, "FAST_KEY")))
	})
}

// The skip-processing opt-out forwards a request without reading it, so a
// configuration whose entrypoints take keys does not honor it, and a request
// that took it is refused once a reload adds keys before its body arrives.
func TestSkipProcessingGivesWayToClientKeys(t *testing.T) {
	t.Setenv("FAST_KEY", "fast-1")
	store := &countingResponseStore{MockResponseStore: NewMockResponseStore()}
	skip := func() []*ext_proc.ProcessingRequest {
		return clientAuthRequest("/v1/chat/completions", map[string]string{"x-vsr-skip-processing": "true"},
			`{"model":"company/fast","messages":[{"role":"user","content":"hi"}]}`)
	}

	public := clientAuthService(t, store)
	skipped := processClientAuthRequest(t, public, skip(), nil)
	require.NotNil(t, skipped.GetRequestBody().GetResponse(), "skip-processing was not honored without client keys")
	assert.Nil(t, skipped.GetRequestBody().GetResponse().GetBodyMutation(), "a skipped request was rewritten")

	protected := clientAuthService(t, store, "FAST_KEY")
	requireRefusal(t, processClientAuthRequest(t, protected, skip(), nil), 401, refusedKey)

	reloaded := clientAuthService(t, store)
	response := processClientAuthRequest(t, reloaded, skip(), func() {
		router, snapshot := clientAuthGeneration(t, store, "FAST_KEY")
		require.NoError(t, reloaded.swapSnapshot(router, snapshot, nil))
	})
	requireRefusal(t, response, 409, refusedSuperseded)
}

func TestEntrypointsOnlyReloadDuringRequest(t *testing.T) {
	store := &countingResponseStore{MockResponseStore: NewMockResponseStore()}
	document := strings.Replace(clientAuthTestConfig, "  router:\n", "  router:\n    entrypoints_only: true\n", 1)
	for _, tc := range []struct {
		name, model   string
		skip, allowed bool
	}{
		{"backend no longer allowed", "worker", false, false},
		{"unparsed request must retry", "worker", true, false},
		{"public alias still allowed", "company/public", false, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			service := clientAuthService(t, store)
			header := map[string]string{}
			if tc.skip {
				header["x-vsr-skip-processing"] = "true"
			}
			requests := clientAuthRequest("/v1/chat/completions", header, `{"model":"`+tc.model+`","messages":[{"role":"user","content":"hi"}]}`)
			response := processClientAuthRequest(t, service, requests, func() {
				router, snapshot := clientAuthGenerationFrom(t, document, store)
				require.NoError(t, service.swapSnapshot(router, snapshot, nil))
			})
			if tc.allowed {
				requireDispatched(t, response)
			} else {
				requireRefusal(t, response, 409, refusedSuperseded)
			}
			// A new request sees strict mode. Admission precedes any history lookup.
			response = processClientAuthRequest(t, service, clientAuthRequest("/v1/responses", nil,
				`{"model":"worker","input":"hi","previous_response_id":"resp_previous"}`), nil)
			requireRefusal(t, response, 403, "not an explicitly configured entrypoint")
			require.Zero(t, store.reads)
		})
	}
}

func TestEntrypointsOnlyKeepsListenerGrantsAndStoredObjectAccess(t *testing.T) {
	store := &countingResponseStore{MockResponseStore: NewMockResponseStore()}
	document := strings.Replace(clientAuthTestConfig, "  router:\n", "  router:\n    entrypoints_only: true\n", 1)
	router, snapshot := clientAuthGenerationFrom(t, document, store)
	service := NewRouterServiceForSnapshot(router, snapshot)
	t.Cleanup(func() { _ = service.Close() })
	ctx := routing.WithUnservedModelRoute(routing.WithListenerModels(t.Context(), []string{"company/fast"}), "company/fast")
	session := router.newRoutingSession(ctx, nil)
	_, err := session.RequestHeaders(routing.Header{{Name: ":method", Value: "POST"}, {Name: ":path", Value: "/v1/chat/completions"}}, false)
	require.NoError(t, err)
	effect, err := session.RequestBody([]byte(`{"model":"company/public","messages":[{"role":"user","content":"hi"}]}`), true)
	require.NoError(t, err)
	require.NotNil(t, effect.Immediate, "a listener must not rewrite a withheld alias into a different entrypoint")
	require.Equal(t, 403, effect.Immediate.Status)

	const id = "resp_existing"
	require.NoError(t, store.StoreResponse(t.Context(), &responseapi.StoredResponse{ID: id, Model: "worker", Status: "completed"}))
	for _, method := range []string{"GET", "DELETE"} {
		session := router.newRoutingSession(t.Context(), nil)
		effect, err := session.RequestHeaders(routing.Header{{Name: ":method", Value: method}, {Name: ":path", Value: "/v1/responses/" + id}}, true)
		require.NoError(t, err)
		require.NotNil(t, effect.Immediate)
		require.Equal(t, 200, effect.Immediate.Status, "strict mode is not an object access policy")
	}
}
