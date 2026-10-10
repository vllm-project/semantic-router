package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// fakeCaller answers hops from a script and records what it was sent.
type fakeCaller struct {
	mu     sync.Mutex
	sent   []*HopRequest
	answer func(ctx context.Context, req *HopRequest) (*HopResponse, error)
}

func (c *fakeCaller) Call(ctx context.Context, req *HopRequest) (*HopResponse, error) {
	c.mu.Lock()
	c.sent = append(c.sent, req)
	c.mu.Unlock()
	if c.answer == nil {
		return completion(modelOf(req), "ok", 1), nil
	}
	return c.answer(ctx, req)
}

func (c *fakeCaller) requests() []*HopRequest {
	c.mu.Lock()
	defer c.mu.Unlock()
	return append([]*HopRequest(nil), c.sent...)
}

func (c *fakeCaller) models() []string {
	var models []string
	for _, req := range c.requests() {
		models = append(models, req.Model)
	}
	return models
}

func modelOf(req *HopRequest) string {
	parsed, err := ParseRequest(req.Request.Body)
	if err != nil {
		return req.Model
	}
	return parsed.Model()
}

// completion is a chat completion answer reporting tokens of usage.
func completion(model, content string, tokens int64) *HopResponse {
	body, _ := json.Marshal(map[string]any{
		"object": "chat.completion",
		"model":  model,
		"choices": []map[string]any{{
			"index": 0, "message": map[string]any{"role": "assistant", "content": content}, "finish_reason": "stop",
		}},
		"usage": map[string]any{"prompt_tokens": tokens, "completion_tokens": 0, "total_tokens": tokens},
	})
	return &HopResponse{Status: 200, Header: routing.Header{{Name: "content-type", Value: "application/json"}}, Body: body}
}

func chatRequest(t *testing.T, text string) Request {
	t.Helper()
	request, err := ParseRequest([]byte(fmt.Sprintf(`{"model":"auto","messages":[{"role":"user","content":%q}]}`, text)))
	if err != nil {
		t.Fatal(err)
	}
	return request
}

func call(id, model string) Step {
	return Step{ID: id, Type: TypeCall, Node: &Call{Model: model}}
}

func respond() Step { return Step{ID: "respond", Type: TypeRespond, Node: &Respond{}} }

func aggregate(id string, strategy Aggregator) Step {
	return Step{ID: id, Type: TypeAggregate, Node: &Aggregate{Strategy: strategy}}
}

func parallel(id string, node *Parallel) Step { return Step{ID: id, Type: TypeParallel, Node: node} }

func branchesOf(models ...string) []Sequence {
	branches := make([]Sequence, len(models))
	for i, model := range models {
		branches[i] = Sequence{call(fmt.Sprintf("call-%s", model), model)}
	}
	return branches
}

func run(t *testing.T, program *Program, caller Caller, opts ...func(*Options)) (*Outcome, error) {
	t.Helper()
	options := Options{Caller: caller, Hop: routing.Hop{Decision: "panel", Recipe: "default"}}
	for _, opt := range opts {
		opt(&options)
	}
	return Run(context.Background(), program, Input{Request: chatRequest(t, "hello")}, options)
}

func answerText(t *testing.T, outcome *Outcome) string {
	t.Helper()
	if outcome == nil || outcome.Response == nil || outcome.Response.Answer == nil {
		t.Fatalf("no answer: %+v", outcome)
	}
	return completionText(outcome.Response.Answer.Body)
}

type panicNode struct{}

func (panicNode) Run(context.Context, *Exec, *State) error { panic("boom") }
