package modelservice

import (
	"context"
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
	deadline, err := remainingMillis(ctx)
	if err != nil {
		return Response{}, err
	}
	body := api.DecisionRequest{
		Model:     optionalString(request.Model),
		State:     request.State,
		Questions: make(map[string]api.Question, len(request.Questions)),
		Options:   &api.RequestOptions{DeadlineMs: deadline},
	}
	for _, question := range request.Questions {
		body.Questions[question.ID] = encodeQuestion(question)
	}
	result, err := c.exchange(ctx, api.BundleTask{Decisions: &body})
	if err != nil {
		return Response{}, err
	}
	if result.Decisions == nil {
		return Response{}, fmt.Errorf("%w: missing decision response body", ErrFailed)
	}
	return checkAnswers(request, decodeResponse(*result.Decisions)), nil
}

func encodeQuestion(question Question) api.Question {
	questionType := question.Type
	var instructions interface{} = question.Instructions
	encoded := api.Question{Type: &questionType, Instructions: &instructions}
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

func decodeResponse(body api.DecisionResponse) Response {
	decoded := Response{Model: body.Model, Answers: make(map[string]Answer, len(body.Answers)), InputTokens: body.Usage.InputTokens}
	for id, answer := range body.Answers {
		decoded.Answers[id] = decodeAnswer(answer)
	}
	return decoded
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
	// A Noul or Score answer is its value; a missing one must not read as 0.
	if (decoded.Type == "noul" && answer.Noul == nil) || (decoded.Type == "score" && answer.Score == nil) {
		return Answer{Type: decoded.Type, Error: "invalid_model_output"}
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
	if !finiteAnswer(decoded) {
		return Answer{Type: decoded.Type, Error: "invalid_model_output"}
	}
	return decoded
}

func finiteAnswer(answer Answer) bool {
	values := []float64{answer.Noul, answer.Score, answer.Confidence}
	for _, probability := range answer.Probabilities {
		values = append(values, probability)
	}
	for _, value := range values {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}
