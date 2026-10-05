package modelservice

import (
	"context"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// Client calls one runtime through the generated contract client.
type Client struct {
	endpoint string
	api      *api.ClientWithResponses
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
	return &Client{endpoint: endpoint, api: generated}, nil
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

// Models returns the runtime's model descriptions.
func (c *Client) Models(ctx context.Context) ([]api.ModelCard, error) {
	response, err := c.api.ListModelsWithResponse(ctx)
	if err != nil {
		return nil, fmt.Errorf("%w: %w", ErrFailed, err)
	}
	if response.JSON200 == nil {
		return nil, fmt.Errorf("%w: /v1/models returned %s", ErrFailed, response.Status())
	}
	return response.JSON200.Data, nil
}

// Decide sends every question in one /v1/decisions call. The context deadline
// is also sent as options.deadline_ms so the runtime drops work it cannot
// start in time.
func (c *Client) Decide(ctx context.Context, request Request) (Response, error) {
	body := api.DecisionRequest{
		State:     request.State,
		Questions: make(map[string]api.Question, len(request.Questions)),
	}
	options := api.RequestOptions{}
	if deadline, ok := ctx.Deadline(); ok {
		remaining := float64(time.Until(deadline).Microseconds()) / 1000.0
		if remaining <= 0 {
			return Response{}, context.DeadlineExceeded
		}
		options.DeadlineMs = &remaining
	}
	body.Options = &options
	for _, question := range request.Questions {
		body.Questions[question.ID] = encodeQuestion(question)
	}
	response, err := c.api.CreateDecisionsWithResponse(ctx, body)
	if err != nil {
		if ctx.Err() != nil {
			return Response{}, ctx.Err()
		}
		return Response{}, fmt.Errorf("%w: %w", ErrFailed, err)
	}
	switch response.StatusCode() {
	case http.StatusOK:
	case http.StatusTooManyRequests:
		return Response{}, ErrOverloaded
	case http.StatusServiceUnavailable:
		return Response{}, ErrUnavailable
	case http.StatusBadRequest, http.StatusNotFound, http.StatusRequestEntityTooLarge:
		return Response{}, fmt.Errorf("%w: %s", ErrRejected, errorMessage(response))
	default:
		return Response{}, fmt.Errorf("%w: %s", ErrFailed, errorMessage(response))
	}
	if response.JSON200 == nil {
		return Response{}, fmt.Errorf("%w: missing decision response body", ErrFailed)
	}
	return checkAnswers(request, decodeResponse(*response.JSON200)), nil
}

func encodeQuestion(question Question) api.Question {
	encoded := api.Question{Type: question.Type, Instructions: question.Instructions}
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

func errorMessage(response *api.CreateDecisionsResponse) string {
	for _, body := range []*api.Error{
		response.JSON400, response.JSON404, response.JSON413,
		response.JSON429, response.JSON500, response.JSON503,
	} {
		if body != nil {
			return fmt.Sprintf("%s (%s)", body.Error.Code, body.Error.Message)
		}
	}
	return response.Status()
}
