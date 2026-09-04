package shadow

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const completionBody = `{"id":"chatcmpl-test","object":"chat.completion","created":1677652288,` +
	`"model":"arm-model","choices":[{"index":0,"message":{"role":"assistant","content":"shadow answer"},` +
	`"logprobs":null,"finish_reason":"stop"}],"usage":{"prompt_tokens":9,"completion_tokens":3,"total_tokens":12}}`

func testParams() *openai.ChatCompletionNewParams {
	return &openai.ChatCompletionNewParams{
		Messages: []openai.ChatCompletionMessageParamUnion{
			openai.SystemMessage("sys"),
			openai.UserMessage("hello"),
		},
	}
}

// armServer records the last received body and Authorization header.
type armServer struct {
	mu       sync.Mutex
	server   *httptest.Server
	body     map[string]interface{}
	authHead string
}

func newArmServer(t *testing.T, status int, delay time.Duration) *armServer {
	t.Helper()
	as := &armServer{}
	as.server = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		as.mu.Lock()
		defer as.mu.Unlock()
		as.authHead = r.Header.Get("Authorization")
		var body map[string]interface{}
		if err := json.NewDecoder(r.Body).Decode(&body); err == nil {
			as.body = body
		}
		if delay > 0 {
			time.Sleep(delay)
		}
		if status != http.StatusOK {
			http.Error(w, "boom", status)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if _, err := w.Write([]byte(completionBody)); err != nil {
			t.Errorf("write completion body: %v", err)
		}
	}))
	t.Cleanup(as.server.Close)
	return as
}

func (as *armServer) snapshot() (map[string]interface{}, string) {
	as.mu.Lock()
	defer as.mu.Unlock()
	return as.body, as.authHead
}

func TestDispatchSuccessAndByteIdentity(t *testing.T) {
	armA := newArmServer(t, http.StatusOK, 0)
	armB := newArmServer(t, http.StatusOK, 0)
	cfg := config.ShadowComparisonConfig{
		Enabled: true,
		Arms: []config.ShadowArmConfig{
			{Name: "arm-a", Model: "model-a", Endpoint: armA.server.URL},
			{Name: "arm-b", Model: "model-b", Endpoint: armB.server.URL},
		},
	}

	results := Dispatch(context.Background(), cfg, testParams(), nil)
	if len(results) != 2 {
		t.Fatalf("want 2 results, got %d", len(results))
	}
	for _, res := range results {
		if res.Outcome != OutcomeCompleted {
			t.Errorf("arm %s outcome = %s, want completed (%s)", res.Arm, res.Outcome, res.Err)
		}
		if res.Content != "shadow answer" {
			t.Errorf("arm %s content = %q", res.Arm, res.Content)
		}
		if res.PromptTokens != 9 || res.CompletionTokens != 3 {
			t.Errorf("arm %s usage parsed = prompt:%d comp:%d, want 9/3", res.Arm, res.PromptTokens, res.CompletionTokens)
		}
	}

	bodyA, _ := armA.snapshot()
	bodyB, _ := armB.snapshot()
	if bodyA == nil || bodyB == nil {
		t.Fatal("one arm never received a request")
	}
	if bodyA["model"] != "model-a" || bodyB["model"] != "model-b" {
		t.Fatalf("models not applied per arm: A=%v B=%v", bodyA["model"], bodyB["model"])
	}
	// The normalized input must be byte-identical across arms: drop the model
	// key and compare the rest.
	delete(bodyA, "model")
	delete(bodyB, "model")
	if !equalMap(bodyA, bodyB) {
		t.Errorf("normalized inputs differ between arms:\nA=%v\nB=%v", bodyA, bodyB)
	}
}

func TestDispatchArmFailureIsolated(t *testing.T) {
	bad := newArmServer(t, http.StatusInternalServerError, 0)
	good := newArmServer(t, http.StatusOK, 0)
	cfg := config.ShadowComparisonConfig{
		Enabled: true,
		Arms: []config.ShadowArmConfig{
			{Name: "bad", Model: "model-bad", Endpoint: bad.server.URL},
			{Name: "good", Model: "model-good", Endpoint: good.server.URL},
		},
	}

	results := Dispatch(context.Background(), cfg, testParams(), nil)
	byName := map[string]ArmResult{}
	for _, res := range results {
		byName[res.Arm] = res
	}
	if byName["bad"].Outcome != OutcomeFailed || byName["bad"].Err == "" {
		t.Errorf("bad arm should fail, got outcome=%s err=%q", byName["bad"].Outcome, byName["bad"].Err)
	}
	if byName["good"].Outcome != OutcomeCompleted {
		t.Errorf("good arm must not be affected by sibling failure: %s", byName["good"].Err)
	}
}

func TestDispatchAccessKey(t *testing.T) {
	arm := newArmServer(t, http.StatusOK, 0)
	cfg := config.ShadowComparisonConfig{
		Enabled: true,
		Arms:    []config.ShadowArmConfig{{Name: "arm", Model: "model", Endpoint: arm.server.URL}},
	}

	results := Dispatch(context.Background(), cfg, testParams(),
		func(armName string) (string, error) { return "sk-test", nil })
	if len(results) != 1 || results[0].Outcome != OutcomeCompleted {
		t.Fatalf("unexpected results: %+v", results)
	}
	_, auth := arm.snapshot()
	if auth != "Bearer sk-test" {
		t.Errorf("Authorization = %q, want Bearer sk-test", auth)
	}
}

func TestDispatchCancellationFailsArm(t *testing.T) {
	slow := newArmServer(t, http.StatusOK, 2*time.Second)
	cfg := config.ShadowComparisonConfig{
		Enabled: true,
		Arms:    []config.ShadowArmConfig{{Name: "slow", Model: "model-slow", Endpoint: slow.server.URL}},
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan []ArmResult, 1)
	go func() {
		done <- Dispatch(ctx, cfg, testParams(), nil)
	}()
	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case results := <-done:
		if len(results) != 1 || results[0].Outcome != OutcomeCancelled {
			t.Fatalf("cancelled arm should be cancelled, got %+v", results)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("dispatch did not return after cancellation")
	}
}

func TestDispatchNilParamsFailsClosed(t *testing.T) {
	arm := newArmServer(t, http.StatusOK, 0)
	cfg := config.ShadowComparisonConfig{
		Enabled: true,
		Arms:    []config.ShadowArmConfig{{Name: "arm", Model: "model", Endpoint: arm.server.URL}},
	}
	results := Dispatch(context.Background(), cfg, nil, nil)
	if len(results) != 1 || results[0].Outcome != OutcomeFailed || results[0].Err != "nil params" {
		t.Fatalf("nil params should fail closed, got %+v", results)
	}
}

func equalMap(a, b map[string]interface{}) bool {
	ja, _ := json.Marshal(a)
	jb, _ := json.Marshal(b)
	return string(ja) == string(jb)
}
