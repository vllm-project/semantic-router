package modelservice

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"sync/atomic"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// DefaultBundleTasks is the runtime's default --max-bundle-tasks, the cap a
// client assumes until /v1/models reports the process's own. A runtime
// refuses a larger /v1/bundle request as a whole (413).
const DefaultBundleTasks = 64

// Client calls one runtime through the generated contract client.
type Client struct {
	endpoint string
	api      *api.ClientWithResponses
	// bundleTasks is the most tasks one /v1/bundle request to this runtime carries.
	bundleTasks atomic.Int64
	// maxInputs is each served model's cap on the inputs of one surface request.
	maxInputs atomic.Pointer[map[string]int]
}

// NewClient builds a client for unix:///path, http://host:port or https://host:port.
func NewClient(endpoint string) (*Client, error) {
	base, httpClient, err := newHTTPClient(endpoint)
	if err != nil {
		return nil, err
	}
	generated, err := api.NewClientWithResponses(base, api.WithHTTPClient(httpClient))
	if err != nil {
		return nil, err
	}
	client := &Client{endpoint: endpoint, api: generated}
	client.bundleTasks.Store(DefaultBundleTasks)
	return client, nil
}

// Endpoint is the address the client calls.
func (c *Client) Endpoint() string { return c.endpoint }

// Ready reports whether the runtime answers /health with 200.
func (c *Client) Ready(ctx context.Context) (bool, string, error) {
	response, err := c.api.GetHealthWithResponse(ctx)
	if err != nil {
		return false, "", fmt.Errorf("%w: %w", ErrFailed, err)
	}
	status := ""
	for _, health := range []*api.Health{response.JSON200, response.JSON503} {
		if health != nil {
			status = string(health.Status)
		}
	}
	return response.StatusCode() == http.StatusOK, status, nil
}

// Models returns the runtime's model descriptions and records the bundle cap
// and the per-model input caps the process reports.
func (c *Client) Models(ctx context.Context) ([]api.ModelCard, error) {
	response, err := c.api.ListModelsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrFailed, err)
	}
	if response.JSON200 == nil {
		return nil, fmt.Errorf("%w: /v1/models returned %s", ErrFailed, response.Status())
	}
	if limit := response.JSON200.Limits.MaxBundleTasks; limit > 0 {
		c.bundleTasks.Store(int64(limit))
	}
	inputs := make(map[string]int, len(response.JSON200.Data))
	for _, card := range response.JSON200.Data {
		if card.Limits != nil && card.Limits.MaxInputs != nil {
			inputs[card.Id] = *card.Limits.MaxInputs
		}
	}
	c.maxInputs.Store(&inputs)
	return response.JSON200.Data, nil
}

// inputCap is the input cap of the served model a request names (the only
// model when it names none), or 0 when the runtime has not reported one.
func (c *Client) inputCap(model *string) int {
	inputs := c.maxInputs.Load()
	if inputs == nil {
		return 0
	}
	if model != nil {
		return (*inputs)[*model]
	}
	if len(*inputs) == 1 {
		for _, limit := range *inputs {
			return limit
		}
	}
	return 0
}

// Decide sends every question in one /v1/decisions call, inside the
// context's bundle when there is one. The context deadline is also sent as
// options.deadline_ms so the runtime drops work it cannot start in time.
func (c *Client) Decide(ctx context.Context, request Request) (Response, error) {
	response, _, err := c.decide(ctx, request, nil, "")
	return response, err
}

// decide is Decide with the served model's result cache for a bundled call:
// the bundle may answer it together with other calls, so its flush looks up
// and stores the call the runtime actually answers. The timing is that of the
// exchange that carried the call.
func (c *Client) decide(ctx context.Context, request Request, cache *resultCache, deployment string) (Response, exchangeTiming, error) {
	body, err := encodeDecisionRequest(ctx, request)
	if err != nil {
		return Response{}, exchangeTiming{}, err
	}
	if bundle := bundleFrom(ctx); bundle != nil {
		return bundle.decide(ctx, c, &decisionCall{request: request, cache: cache, deployment: deployment}, body)
	}
	result, timing, err := c.exchange(ctx, api.BundleTask{Decisions: &body})
	if err != nil {
		return Response{}, timing, err
	}
	if result.Decisions == nil {
		return Response{}, timing, fmt.Errorf("%w: missing decision response body", ErrFailed)
	}
	return decodeResponse(*result.Decisions, request.Questions), timing, nil
}

func encodeDecisionRequest(ctx context.Context, request Request) (api.DecisionRequest, error) {
	deadline, err := remainingMillis(ctx)
	if err != nil {
		return api.DecisionRequest{}, err
	}
	var state interface{} = request.State
	if request.Parts != nil {
		state = request.Parts
	}
	body := api.DecisionRequest{
		Model:     optionalString(request.Model),
		State:     state,
		Questions: make(map[string]api.Question, len(request.Questions)),
		Options:   &api.RequestOptions{DeadlineMs: deadline},
	}
	for _, question := range request.Questions {
		body.Questions[question.ID] = encodeQuestion(question)
	}
	return body, nil
}

// labelCriteria encodes Set and Span labels as the criteria object in label
// order: a label's position is part of the question the model reads.
type labelCriteria []Choice

func (labels labelCriteria) MarshalJSON() ([]byte, error) {
	var buffer bytes.Buffer
	buffer.WriteByte('{')
	for index, label := range labels {
		if index > 0 {
			buffer.WriteByte(',')
		}
		key, err := json.Marshal(label.Key)
		if err != nil {
			return nil, err
		}
		buffer.Write(key)
		buffer.WriteByte(':')
		if label.Description == "" {
			buffer.WriteString("null")
			continue
		}
		description, err := json.Marshal(label.Description)
		if err != nil {
			return nil, err
		}
		buffer.Write(description)
	}
	buffer.WriteByte('}')
	return buffer.Bytes(), nil
}

func encodeQuestion(question Question) api.Question {
	if question.Preset != "" {
		preset := question.Preset
		return api.Question{Preset: &preset, Threshold: question.Threshold}
	}
	questionType := question.Type
	var instructions interface{} = question.Instructions
	encoded := api.Question{Type: &questionType, Instructions: &instructions, Threshold: question.Threshold}
	if len(question.Labels) > 0 {
		var criteria interface{} = labelCriteria(question.Labels)
		encoded.Criteria = &criteria
	}
	if question.Head != "" {
		head := api.QuestionHead(question.Head)
		encoded.Head = &head
	}
	if len(question.Choices) > 0 {
		choices := make([]api.ChoiceOption, len(question.Choices))
		for index, choice := range question.Choices {
			choices[index] = api.ChoiceOption{Key: choice.Key}
			if choice.Description != "" {
				var description interface{} = choice.Description
				choices[index].Description = &description
			}
		}
		encoded.Choices = &choices
	}
	if len(question.Levels) > 0 {
		levels := make([]interface{}, len(question.Levels))
		for index, level := range question.Levels {
			levels[index] = level
		}
		encoded.Levels = &levels
	}
	return encoded
}

// decodeResponse reads one answer per asked question. A Set question has no
// entry in answers: its labels come from sets (the per-label Noul answers
// under "<id>.<label>" repeat them), and a Span answer adds its spans to the
// Noul the runtime reports for it.
func decodeResponse(body api.DecisionResponse, questions []Question) Response {
	decoded := Response{Model: body.Model, Answers: make(map[string]Answer, len(questions)), InputTokens: body.Usage.InputTokens}
	for _, question := range questions {
		if answer, ok := decodeQuestionAnswer(body, question); ok {
			decoded.Answers[question.ID] = answer
		}
	}
	return decoded
}

func decodeQuestionAnswer(body api.DecisionResponse, question Question) (Answer, bool) {
	id := question.ID
	answer, answered := body.Answers[id]
	if answered && answer.Error != nil {
		return decodeAnswer(answer), true
	}
	threshold, _ := lookup(body.Thresholds, id)
	if set, ok := lookup(body.Sets, id); ok {
		decoded := Answer{Type: "set", Probabilities: set.Probabilities, Selected: set.Selected, Threshold: threshold}
		return checkedAnswer(decoded), true
	}
	if spans, ok := lookup(body.Spans, id); ok && answered {
		decoded := decodeAnswer(answer)
		decoded.Type = "span"
		if decoded.Error != "" {
			return decoded, true
		}
		decoded.Spans, decoded.Head, decoded.Threshold = spans, lookupString(body.SpanHeads, id), threshold
		return checkedAnswer(decoded), true
	}
	if !answered {
		return Answer{}, false
	}
	if question.Type == "set" || question.Type == "span" {
		return Answer{Type: question.Type, Error: "invalid_model_output"}, true
	}
	return decodeAnswer(answer), true
}

func lookup[T any](values *map[string]T, key string) (T, bool) {
	var zero T
	if values == nil {
		return zero, false
	}
	value, ok := (*values)[key]
	return value, ok
}

func lookupString(values *map[string]string, key string) string {
	value, _ := lookup(values, key)
	return value
}

func decodeAnswer(answer api.Answer) Answer {
	decoded := Answer{}
	if answer.Type != nil {
		decoded.Type = *answer.Type
	}
	if answer.Error != nil {
		decoded.Error = string(*answer.Error)
		return decoded
	}
	if answer.Choice != nil {
		decoded.Choice = *answer.Choice
	}
	if answer.Noul != nil {
		decoded.Noul = *answer.Noul
	}
	if answer.Score != nil {
		decoded.Score = *answer.Score
	}
	if answer.Confidence != nil {
		decoded.Confidence = *answer.Confidence
	}
	if answer.Probabilities != nil {
		decoded.Probabilities = *answer.Probabilities
	}
	return checkedAnswer(decoded)
}

// checkedAnswer replaces an answer holding a non-finite number with an
// invalid_model_output error.
func checkedAnswer(answer Answer) Answer {
	if !finiteAnswer(answer) {
		return Answer{Type: answer.Type, Error: "invalid_model_output"}
	}
	return answer
}

func finiteAnswer(answer Answer) bool {
	values := []float64{answer.Noul, answer.Score, answer.Confidence, answer.Threshold}
	for _, probability := range answer.Probabilities {
		values = append(values, probability)
	}
	for _, span := range answer.Spans {
		values = append(values, span.Probability)
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}
