package graph

import (
	"maps"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// State is what flows through a graph: the request the next call sends, the
// results of the latest step, named values, and the response once a respond
// step has run. Each parallel branch works on its own fork.
type State struct {
	Request  Request
	Results  []*Result
	Values   map[string]any
	Response *Response
}

// fork is the state a parallel branch starts from: the same request and
// values, and no results yet.
func (st *State) fork() *State {
	return &State{Request: st.Request.Clone(), Values: maps.Clone(st.Values)}
}

// Key names a typed value in State.Values, so the nodes that share a value
// agree on its type.
type Key[T any] struct{ name string }

// NewKey returns the key for name.
func NewKey[T any](name string) Key[T] { return Key[T]{name: name} }

// Name returns the key's name.
func (k Key[T]) Name() string { return k.name }

// Get returns the value under k, if st holds one of type T.
func (k Key[T]) Get(st *State) (T, bool) {
	value, ok := st.Values[k.name].(T)
	return value, ok
}

// Set stores value under k.
func (k Key[T]) Set(st *State, value T) {
	if st.Values == nil {
		st.Values = map[string]any{}
	}
	st.Values[k.name] = value
}

// In returns the value under k in values, such as an Input's or an
// Outcome's.
func (k Key[T]) In(values map[string]any) (T, bool) {
	value, ok := values[k.name].(T)
	return value, ok
}

// Put stores value under k in values.
func (k Key[T]) Put(values map[string]any, value T) { values[k.name] = value }

// Round holds a loop's current round, from 1.
var Round = NewKey[int]("graph.round")

// Usage counts the tokens of model calls.
type Usage struct {
	PromptTokens     int64 `json:"prompt_tokens,omitempty"`
	CompletionTokens int64 `json:"completion_tokens,omitempty"`
	TotalTokens      int64 `json:"total_tokens,omitempty"`
}

// Add returns the sum of u and other.
func (u Usage) Add(other Usage) Usage {
	return Usage{
		PromptTokens:     u.PromptTokens + other.PromptTokens,
		CompletionTokens: u.CompletionTokens + other.CompletionTokens,
		TotalTokens:      u.TotalTokens + other.TotalTokens,
	}
}

// Result is one model call's outcome.
type Result struct {
	// Step is the step that made the call.
	Step  string
	Model string
	// Status is the response status as the hop returned it; zero when the
	// call failed before a response.
	Status int
	Header routing.Header
	// Body is the response body: a chat completion, or its event stream.
	Body    []byte
	Usage   Usage
	Latency time.Duration
	// Err is why the call failed; nil for a 2xx response.
	Err error

	content *string
}

// OK reports whether the call succeeded.
func (r *Result) OK() bool {
	return r != nil && r.Err == nil && r.Status >= http.StatusOK && r.Status < http.StatusMultipleChoices
}

// Content returns the text of the response's first choice.
func (r *Result) Content() string {
	if r == nil {
		return ""
	}
	if r.content == nil {
		text := completionText(r.Body)
		r.content = &text
	}
	return *r.content
}

// Response is how a graph answers its request: with an answer it composed,
// or by handing one final model call to the gateway.
type Response struct {
	Answer *routing.Response
	Final  *Final
}

// Final is the model call that ends a graph whose last step streams: the
// gateway sends Request to Model as an ordinary routed call, so the model's
// answer reaches the client with the request's own response phases.
type Final struct {
	Model   string
	Request Request
}
