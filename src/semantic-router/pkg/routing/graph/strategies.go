package graph

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"sync/atomic"
	"text/template"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// ErrNoSuccess reports an aggregate step with no successful result to work
// from.
var ErrNoSuccess = errors.New("graph: no successful result to aggregate")

// First keeps the first successful result.
type First struct{}

// Aggregate keeps the first success.
func (First) Aggregate(_ context.Context, _ *Exec, st *State) error {
	for _, result := range st.Results {
		if result.OK() {
			st.Results = []*Result{result}
			return nil
		}
	}
	return ErrNoSuccess
}

// Vote keeps the answer most successful results agree on, compared after
// trimming space; a tie goes to the answer seen first.
type Vote struct{}

// Aggregate counts the votes.
func (Vote) Aggregate(_ context.Context, _ *Exec, st *State) error {
	counts := map[string]int{}
	firsts := map[string]*Result{}
	var winner *Result
	best := 0
	for _, result := range st.Results {
		if !result.OK() {
			continue
		}
		answer := strings.TrimSpace(result.Content())
		if firsts[answer] == nil {
			firsts[answer] = result
		}
		counts[answer]++
		if counts[answer] > best {
			best, winner = counts[answer], firsts[answer]
		}
	}
	if winner == nil {
		return ErrNoSuccess
	}
	st.Results = []*Result{winner}
	return nil
}

// Concat composes one answer from every successful result's text, in order,
// joined by Separator (a blank line when empty).
type Concat struct {
	Separator string
}

// Aggregate composes the answer.
func (c Concat) Aggregate(_ context.Context, _ *Exec, st *State) error {
	separator := c.Separator
	if separator == "" {
		separator = "\n\n"
	}
	var texts []string
	var usage Usage
	var last *Result
	for _, result := range st.Results {
		if result.OK() {
			texts = append(texts, result.Content())
			usage = usage.Add(result.Usage)
			last = result
		}
	}
	if last == nil {
		return ErrNoSuccess
	}
	body, err := composeCompletion(last.Model, []string{strings.Join(texts, separator)}, usage)
	if err != nil {
		return err
	}
	st.Results = []*Result{composedResult(last.Model, body, usage)}
	return nil
}

// Choices composes one answer whose choices are the successful results'
// texts, in order, so the client compares them.
type Choices struct{}

// Aggregate composes the answer.
func (Choices) Aggregate(_ context.Context, _ *Exec, st *State) error {
	var texts []string
	var usage Usage
	var last *Result
	for _, result := range st.Results {
		if result.OK() {
			texts = append(texts, result.Content())
			usage = usage.Add(result.Usage)
			last = result
		}
	}
	if last == nil {
		return ErrNoSuccess
	}
	body, err := composeCompletion(last.Model, texts, usage)
	if err != nil {
		return err
	}
	st.Results = []*Result{composedResult(last.Model, body, usage)}
	return nil
}

var composedID atomic.Int64

// composeCompletion encodes a chat completion with one choice per text.
func composeCompletion(model string, texts []string, usage Usage) ([]byte, error) {
	type message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	}
	type choice struct {
		Index        int     `json:"index"`
		Message      message `json:"message"`
		FinishReason string  `json:"finish_reason"`
	}
	choices := make([]choice, len(texts))
	for i, text := range texts {
		choices[i] = choice{Index: i, Message: message{Role: "assistant", Content: text}, FinishReason: "stop"}
	}
	return json.Marshal(struct {
		ID      string   `json:"id"`
		Object  string   `json:"object"`
		Created int64    `json:"created"`
		Model   string   `json:"model"`
		Choices []choice `json:"choices"`
		Usage   Usage    `json:"usage"`
	}{
		ID:      fmt.Sprintf("chatcmpl-graph-%d", composedID.Add(1)),
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   model,
		Choices: choices,
		Usage:   usage,
	})
}

// composedResult is an answer a strategy composed without a call. Its usage
// adds up the calls it read from, as the answer reports it; the run charged
// those calls already.
func composedResult(model string, body []byte, usage Usage) *Result {
	return &Result{
		Model:  model,
		Status: http.StatusOK,
		Header: routing.Header{{Name: "content-type", Value: "application/json"}},
		Body:   body,
		Usage:  usage,
	}
}

// SystemMode is how a system prompt meets the request's own.
type SystemMode string

const (
	// SystemReplace replaces the request's system prompt, or adds one.
	SystemReplace SystemMode = "replace"
	// SystemPrepend puts the prompt before the request's system prompt.
	SystemPrepend SystemMode = "prepend"
)

// SystemPrompt sets the system prompt of the request the next call sends.
type SystemPrompt struct {
	Content string
	Mode    SystemMode
}

// Transform sets the prompt.
func (s SystemPrompt) Transform(_ context.Context, _ *Exec, st *State) error {
	messages, err := st.Request.Messages()
	if err != nil {
		return err
	}
	for i, message := range messages {
		if MessageRole(message) != "system" {
			continue
		}
		content := s.Content
		if s.Mode == SystemPrepend {
			content = s.Content + "\n\n" + MessageText(message)
		}
		messages[i] = Message("system", content)
		return st.Request.SetMessages(messages)
	}
	return st.Request.SetMessages(append([]map[string]json.RawMessage{Message("system", s.Content)}, messages...))
}

// AppendResults adds each latest result's text to the request as a message
// of Role (assistant when empty), so the next call sees what came before.
type AppendResults struct {
	Role string
}

// Transform appends the results.
func (a AppendResults) Transform(_ context.Context, _ *Exec, st *State) error {
	role := a.Role
	if role == "" {
		role = "assistant"
	}
	messages, err := st.Request.Messages()
	if err != nil {
		return err
	}
	for _, result := range st.Results {
		if result.OK() {
			messages = append(messages, Message(role, result.Content()))
		}
	}
	return st.Request.SetMessages(messages)
}

// PromptData is what a Prompt template sees.
type PromptData struct {
	// Original is the text of the request's last user message.
	Original string
	// Results are the latest successful results' texts, with their models.
	Results []PromptResult
	Round   int
}

// PromptResult is one result as a Prompt template sees it.
type PromptResult struct {
	Model   string
	Content string
}

// Prompt rewrites the request's last user message from a template, such as a
// synthesis prompt that quotes a panel's answers.
type Prompt struct {
	Template *template.Template
}

// Transform renders the template into the last user message.
func (p Prompt) Transform(_ context.Context, _ *Exec, st *State) error {
	messages, err := st.Request.Messages()
	if err != nil {
		return err
	}
	last := -1
	for i, message := range messages {
		if MessageRole(message) == "user" {
			last = i
		}
	}
	data := PromptData{}
	data.Round, _ = Round.Get(st)
	if last >= 0 {
		data.Original = MessageText(messages[last])
	}
	for _, result := range st.Results {
		if result.OK() {
			data.Results = append(data.Results, PromptResult{Model: result.Model, Content: result.Content()})
		}
	}
	var text bytes.Buffer
	if err := p.Template.Execute(&text, data); err != nil {
		return fmt.Errorf("graph: render the prompt: %w", err)
	}
	if last < 0 {
		messages = append(messages, Message("user", text.String()))
	} else {
		messages[last] = Message("user", text.String())
	}
	return st.Request.SetMessages(messages)
}
