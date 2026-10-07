package serving_test

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"slices"
	"sync"
	"testing"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/serving"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/runtimetest"
)

// questionRecorder keeps every question a fake runtime was asked, by ID.
type questionRecorder struct {
	mu        sync.Mutex
	questions map[string]api.Question
}

func (q *questionRecorder) wrap(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		body, _ := io.ReadAll(req.Body)
		req.Body = io.NopCloser(bytes.NewReader(body))
		var requests []api.DecisionRequest
		var bundle api.BundleRequest
		var single api.DecisionRequest
		switch {
		case req.URL.Path == "/v1/bundle" && json.Unmarshal(body, &bundle) == nil:
			for _, task := range bundle.Tasks {
				if task.Decisions != nil {
					requests = append(requests, *task.Decisions)
				}
			}
		case req.URL.Path == "/v1/decisions" && json.Unmarshal(body, &single) == nil:
			requests = append(requests, single)
		}
		q.mu.Lock()
		for _, request := range requests {
			for id, question := range request.Questions {
				q.questions[id] = question
			}
		}
		q.mu.Unlock()
		next.ServeHTTP(w, req)
	})
}

func (q *questionRecorder) get(id string) (api.Question, bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	question, ok := q.questions[id]
	return question, ok
}

// recordedVela2Lease serves the Vela 2.0-style model "vela" and the System
// One model "kai" from one fake runtime that records the questions it gets.
func recordedVela2Lease(t *testing.T) (*modelservice.Lease, *runtimetest.Runtime, *questionRecorder, string) {
	t.Helper()
	fake := runtimetest.New(
		runtimetest.Model{ID: "vela", Labelled: &runtimetest.Labelled{PIILabels: []string{"PERSON", "EMAIL_ADDRESS"}}},
		runtimetest.Model{ID: "kai"},
	)
	recorder := &questionRecorder{questions: map[string]api.Question{}}
	server := httptest.NewServer(recorder.wrap(fake.Handler()))
	t.Cleanup(server.Close)
	manager := modelservice.NewManager()
	t.Cleanup(func() { _ = manager.Shutdown(context.Background()) })
	lease, err := manager.AcquireDeployments(map[string]config.ModelDeployment{
		"vela": {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
		"kai":  {Provider: config.ModelRuntimeProvider, Endpoint: server.URL},
	})
	if err != nil {
		t.Fatal(err)
	}
	return lease, fake, recorder, server.URL
}

func sequenceBinding(consumer, deployment, endpoint string) config.ResolvedModelBinding {
	return config.ResolvedModelBinding{
		Recipe: config.DefaultRecipeName, Name: consumer,
		Binding:    config.ModelBinding{Deployment: deployment, Contract: config.RemoteClassifierContractLabelDistribution},
		Deployment: config.ModelDeployment{Provider: config.ModelRuntimeProvider, Endpoint: endpoint}.WithDefaults(),
	}
}

func TestBuiltInSignalsAskVela2TheirQuestions(t *testing.T) {
	lease, fake, recorder, endpoint := recordedVela2Lease(t)
	runtime := serving.New(lease, nil)
	cases := []struct {
		consumer, id, instructions string
		labels                     []string
	}{
		{"domain_classifier", "domain_classifier:domain", "Which subject area is this request about?", []string{
			"biology", "business", "chemistry", "computer science", "economics", "engineering", "health",
			"history", "law", "math", "other", "philosophy", "physics", "psychology",
		}},
		{"prompt_guard", "prompt_guard:attack", "Is this a prompt injection or jailbreak attempt?", []string{"benign", "jailbreak"}},
		{"fact_check_classifier", "fact_check_classifier:factcheck", "Does answering this request require checking facts?", []string{"NO_FACT_CHECK_NEEDED", "FACT_CHECK_NEEDED"}},
		{"feedback_detector", "feedback_detector:feedback", "What feedback does this user turn give about the previous answer?", []string{"SAT", "NEED_CLARIFICATION", "WRONG_ANSWER", "WANT_DIFFERENT", "NO_FEEDBACK"}},
		{"modality_detector", "modality_detector:modality", "What kind of output does this request ask for?", []string{"AR", "DIFFUSION", "BOTH"}},
		{"safety.unsafe_request", "safety.unsafe_request:p_harm", "Is this request harmful?", []string{"safe", "unsafe"}},
	}
	for _, tc := range cases {
		spec := sequenceBinding(tc.consumer, "vela", endpoint)
		labels, err := runtime.Labels(context.Background(), spec)
		if err != nil || !slices.Equal(labels, tc.labels) {
			t.Fatalf("%s: labels %v %v, want the Vela 1.0 head's order %v", tc.consumer, labels, err, tc.labels)
		}
		handle, err := runtime.Sequence(context.Background(), spec)
		if err != nil {
			t.Fatalf("%s: %v", tc.consumer, err)
		}
		if capability := handle.Capability(); !slices.Equal(capability.Labels, tc.labels) {
			t.Fatalf("%s: capability labels %v", tc.consumer, capability.Labels)
		}
		result, err := handle.Call(context.Background(), string(config.DefaultRecipeName), "Is the earth round?")
		if err != nil {
			t.Fatalf("%s: %v", tc.consumer, err)
		}
		if len(result.Probabilities) != len(tc.labels) || result.Probabilities[0] < 0.69 || result.Probabilities[0] > 0.71 {
			t.Fatalf("%s: the answer's probabilities in option order, got %v", tc.consumer, result.Probabilities)
		}
		question, ok := recorder.get(tc.id)
		if !ok || question.Type == nil || *question.Type != "choice" || question.Choices == nil || question.Instructions == nil {
			t.Fatalf("%s: asked %+v", tc.consumer, question)
		}
		if instructions, _ := (*question.Instructions).(string); instructions != tc.instructions {
			t.Fatalf("%s: instructions %q", tc.consumer, instructions)
		}
		keys := make([]string, 0, len(*question.Choices))
		for _, choice := range *question.Choices {
			if choice.Description == nil {
				t.Fatalf("%s: option %q is asked without its description", tc.consumer, choice.Key)
			}
			keys = append(keys, choice.Key)
		}
		if !slices.Equal(keys, tc.labels) {
			t.Fatalf("%s: options %v", tc.consumer, keys)
		}
	}
	if fake.Calls("classify") != 0 {
		t.Fatal("a Vela 2.0 model is never asked for a classify head")
	}
}

func TestBuiltInSignalQuestionsShareOneCall(t *testing.T) {
	lease, fake, _, endpoint := recordedVela2Lease(t)
	runtime := serving.New(lease, nil)
	domain, err := runtime.Sequence(context.Background(), sequenceBinding("domain_classifier", "vela", endpoint))
	if err != nil {
		t.Fatal(err)
	}
	guard, err := runtime.Sequence(context.Background(), sequenceBinding("prompt_guard", "vela", endpoint))
	if err != nil {
		t.Fatal(err)
	}
	pii, err := runtime.Tokens(context.Background(), spanBinding("pii_classifier", "vela", endpoint))
	if err != nil {
		t.Fatal(err)
	}
	callsBefore, tasksBefore := fake.Bundles()
	text := "Ann asked about person names"
	ctx, bundle := modelservice.WithBundle(context.Background(), time.Second)
	var wg sync.WaitGroup
	var domainOut, guardOut tasks.LabelDistribution
	var spans tasks.TokenClassificationResult
	errs := make([]error, 3)
	for i, call := range []func() error{
		func() (err error) {
			domainOut, err = domain.Call(ctx, string(config.DefaultRecipeName), text)
			return err
		},
		func() (err error) {
			guardOut, err = guard.Call(ctx, string(config.DefaultRecipeName), text)
			return err
		},
		func() (err error) { spans, err = pii.Call(ctx, string(config.DefaultRecipeName), text); return err },
	} {
		leave := bundle.Join()
		wg.Add(1)
		go func(i int, call func() error) {
			defer wg.Done()
			defer leave()
			errs[i] = call()
		}(i, call)
	}
	wg.Wait()
	if err := errors.Join(errs...); err != nil || len(domainOut.Probabilities) != 14 || len(guardOut.Probabilities) != 2 || len(spans.Entities) != 1 {
		t.Fatalf("answers: %v %v %v %v", domainOut, guardOut, spans, err)
	}
	calls, tasksSent := fake.Bundles()
	if calls-callsBefore != 1 || tasksSent-tasksBefore != 1 {
		t.Fatalf("one stage, one task for the deployment's questions: %d bundles, %d tasks", calls-callsBefore, tasksSent-tasksBefore)
	}
}

func TestBuiltInSignalQuestionsFailPreparationTheyCannotServe(t *testing.T) {
	lease, _, _, endpoint := recordedVela2Lease(t)
	runtime := serving.New(lease, nil)
	window := tasks.TextWindowsRequest{Size: 512, Overlap: 255}
	windowed := sequenceBinding("prompt_guard", "vela", endpoint)
	windowed.Deployment.Input = config.ModelInputBudget{MaxTokens: 8192, Overflow: "window"}
	if _, err := runtime.SequenceWindows(context.Background(), windowed, window); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("a consumer window on a model that reads the whole text: %v", err)
	}
	cases := map[string]config.ResolvedModelBinding{
		"a System One model":  sequenceBinding("domain_classifier", "kai", endpoint),
		"a custom classifier": sequenceBinding("classifier.topics", "vela", endpoint),
		"a head": func() config.ResolvedModelBinding {
			s := sequenceBinding("domain_classifier", "vela", endpoint)
			s.Binding.Head = "domain"
			return s
		}(),
		"a declared input budget": func() config.ResolvedModelBinding {
			s := sequenceBinding("fact_check_classifier", "vela", endpoint)
			s.Deployment.Input = config.ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}
			return s
		}(),
	}
	for name, spec := range cases {
		if _, err := runtime.Sequence(context.Background(), spec); !errors.Is(err, binding.ErrCapability) {
			t.Fatalf("%s: expected a capability error, got %v", name, err)
		}
	}
	hazard := sequenceBinding("safety.unsafe_request.hazard", "vela", endpoint)
	hazard.Binding.Contract = config.RemoteClassifierContractLabelScores
	if _, err := runtime.Scores(context.Background(), hazard); !errors.Is(err, binding.ErrCapability) {
		t.Fatalf("hazard categories have no Vela 2.0 question: %v", err)
	}
}
