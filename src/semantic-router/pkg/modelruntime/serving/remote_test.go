package serving_test

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// systemOneServer is the System One API of vllm-sr/d3-ultra: it lists its
// question types and presets, answers a choice question with its first
// option (by key) most probable and a span question (the pii preset) with a
// PERSON span over the first three bytes of its text, each having read its
// whole input, and keeps the body of every call.
type systemOneServer struct {
	listing string
	mu      sync.Mutex
	bodies  []string
}

func (s *systemOneServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method == http.MethodGet {
		_, _ = w.Write([]byte(s.listing))
		return
	}
	type question struct {
		Type     string                     `json:"type"`
		Preset   string                     `json:"preset"`
		Criteria map[string]json.RawMessage `json:"criteria"`
	}
	type state struct {
		State     string              `json:"state"`
		Questions map[string]question `json:"questions"`
	}
	var body struct {
		Model     string              `json:"model"`
		State     string              `json:"state"`
		Questions map[string]question `json:"questions"`
		States    map[string]state    `json:"states"`
	}
	raw := new(strings.Builder)
	if err := json.NewDecoder(teeReader(r, raw)).Decode(&body); err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	s.mu.Lock()
	s.bodies = append(s.bodies, raw.String())
	s.mu.Unlock()
	answer := func(text string, questions map[string]question) map[string]any {
		answers, spans := map[string]any{}, map[string]any{}
		for id, q := range questions {
			if q.Preset == "pii" || q.Type == "span" {
				answers[id] = map[string]any{"type": "noul", "noul": 0.95, "input_coverage": "complete"}
				spans[id] = []map[string]any{{"start": 0, "end": 3, "label": "PERSON", "probability": 0.9, "text": text[:3]}}
				continue
			}
			keys := make([]string, 0, len(q.Criteria))
			for key := range q.Criteria {
				keys = append(keys, key)
			}
			slices.Sort(keys)
			probabilities := map[string]float64{}
			for index, key := range keys {
				probabilities[key] = 0.3 / float64(len(keys)-1)
				if index == 0 {
					probabilities[key] = 0.7
				}
			}
			answers[id] = map[string]any{"type": "choice", "choice": keys[0], "probabilities": probabilities, "confidence": 0.7, "input_coverage": "complete"}
		}
		return map[string]any{"model": body.Model, "answers": answers, "spans": spans, "usage": map[string]int{"input_tokens": 9}}
	}
	response := answer(body.State, body.Questions)
	if len(body.States) > 0 {
		states := map[string]any{}
		for name, entry := range body.States {
			states[name] = answer(entry.State, entry.Questions)
		}
		response["states"] = states
	}
	_ = json.NewEncoder(w).Encode(response)
}

func (s *systemOneServer) calls() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.bodies)
}

type teeBody struct {
	r *http.Request
	w *strings.Builder
}

func (t teeBody) Read(p []byte) (int, error) {
	n, err := t.r.Body.Read(p)
	t.w.Write(p[:n])
	return n, err
}

func teeReader(r *http.Request, w *strings.Builder) teeBody { return teeBody{r: r, w: w} }

// remoteRuntime serves the systemone deployment hosted through a fake API
// that lists listing.
func remoteRuntime(t *testing.T, listing string) (*serving.Runtime, *systemOneServer, config.ModelDeployment) {
	t.Helper()
	api := &systemOneServer{listing: listing}
	server := httptest.NewServer(api)
	t.Cleanup(server.Close)
	deployment := config.ModelDeployment{Provider: config.SystemOneProvider, Endpoint: server.URL + "/v1/systemone", Model: "vllm-sr/d3-ultra"}
	cfg := &config.RouterConfig{}
	cfg.ModelDeployments = map[string]config.ModelDeployment{"hosted": deployment}
	remote, err := modelservice.NewRemoteSystemOne(cfg)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = remote.Close() })
	return serving.New(serving.WithRemote(nil, remote), nil), api, deployment
}

func remoteBinding(consumer, contract string, deployment config.ModelDeployment) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: consumer,
		Binding:    config.ModelBinding{Deployment: "hosted", Contract: contract},
		Deployment: deployment,
	}
}

func TestBuiltInSignalsAskASystemOneDeploymentInOneCall(t *testing.T) {
	runtime, api, deployment := remoteRuntime(t, `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["choice","noul","span"],"presets":["pii"]}]}`)
	guard, err := runtime.Sequence(context.Background(), remoteBinding("prompt_guard", config.RemoteClassifierContractLabelDistribution, deployment))
	if err != nil {
		t.Fatal(err)
	}
	if capability := guard.Capability(); capability.Provider != config.SystemOneProvider || capability.Device != modelservice.RemoteDevice || capability.Deployment != "hosted" {
		t.Fatalf("capability %+v", capability)
	}
	pii, err := runtime.Tokens(context.Background(), remoteBinding("pii_classifier", config.RemoteClassifierContractTokenSpans, deployment))
	if err != nil {
		t.Fatalf("%v; calls %v", err, api.calls())
	}
	prepared := len(api.calls())
	text := "Ann asked about person names"
	ctx, bundle := modelservice.WithBundle(context.Background(), time.Second)
	var guardOut tasks.LabelDistribution
	var spans tasks.TokenClassificationResult
	errs := make([]error, 2)
	var wg sync.WaitGroup
	callers := []func() error{
		func() (err error) {
			guardOut, err = guard.Call(ctx, string(config.DefaultRecipeName), text)
			return err
		},
		func() (err error) { spans, err = pii.Call(ctx, string(config.DefaultRecipeName), text); return err },
	}
	leaves := make([]func(), len(callers))
	for i := range callers {
		leaves[i] = bundle.Join()
	}
	// A bundle may flush when all registered participants are waiting, so
	// register everyone before starting any caller.
	for i, call := range callers {
		wg.Add(1)
		go func(i int, call func() error, leave func()) {
			defer wg.Done()
			defer leave()
			errs[i] = call()
		}(i, call, leaves[i])
	}
	wg.Wait()
	calls := api.calls()[prepared:]
	if err := errors.Join(errs...); err != nil || len(guardOut.Probabilities) != 2 || len(spans.Entities) != 1 {
		t.Fatalf("guard %v (%v), pii %v (%v); calls %v", guardOut, errs[0], spans, errs[1], calls)
	}
	if len(calls) != 1 {
		t.Fatalf("calls %d, want the stage's two questions in one call", len(calls))
	}
	for _, want := range []string{`"prompt_guard:attack"`, `"pii_classifier:pii"`, `"criteria":{"benign":`} {
		if !strings.Contains(calls[0], want) {
			t.Fatalf("the call %s does not carry %s", calls[0], want)
		}
	}
	for _, refused := range []string{`"truncate"`, `"max_tokens"`, `"choices"`} {
		if strings.Contains(calls[0], refused) {
			t.Fatalf("the call %s carries %s, which a System One API does not take", calls[0], refused)
		}
	}
}

func TestASystemOneDeploymentServesNoHead(t *testing.T) {
	runtime, _, deployment := remoteRuntime(t, `{"data":[{"id":"vllm-sr/d3-ultra","question_types":["choice","noul"]}]}`)
	if _, err := runtime.Sequence(context.Background(), remoteBinding("classifier.topics", config.RemoteClassifierContractLabelDistribution, deployment)); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a classify head of a systemone deployment: %v", err)
	}
	if _, err := runtime.Tokens(context.Background(), remoteBinding("pii_classifier", config.RemoteClassifierContractTokenSpans, deployment)); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("spans from a deployment that lists no span type or pii preset: %v", err)
	}
}
