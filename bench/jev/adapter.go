// jev-eval is a research-only protocol probe, not a registered Router backend.
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

const contract = config.RemoteClassifierContractLabelDistribution

type question struct {
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
}

type request struct {
	Model     string              `json:"model"`
	State     string              `json:"state"`
	Questions map[string]question `json:"questions"`
}

type answer struct {
	Type          string              `json:"type"`
	Choice        string              `json:"choice"`
	Confidence    *float64            `json:"confidence"`
	Probabilities map[string]*float64 `json:"probabilities"`
}

type response struct {
	Model   string            `json:"model"`
	Answers map[string]answer `json:"answers"`
	Usage   json.RawMessage   `json:"usage,omitempty"`
}

type adapter struct {
	client   *connector.Client
	model    string
	question question
}

func newAdapter(endpoint, key string, q question, timeout time.Duration) (*adapter, error) {
	client, err := connector.New(endpoint, func(_ context.Context, r *http.Request) error {
		r.Header.Set("Authorization", "Bearer "+key)
		return nil
	}, connector.Options{
		AttemptTimeout: timeout, MaxRetries: 0,
		MaxRequestBytes: 1 << 20, MaxResponseBytes: 1 << 20, MaxErrorBytes: 8192,
	})
	if err != nil {
		return nil, err
	}
	return &adapter{client: client, model: "jev-1.13.0", question: q}, nil
}

func (a *adapter) evaluate(ctx context.Context, text string) (request, connector.Result, *response, error) {
	req := request{Model: a.model, State: text, Questions: map[string]question{"intent": a.question}}
	body, err := json.Marshal(req)
	if err != nil {
		return req, connector.Result{}, nil, err
	}
	wire, err := a.client.DoRequest(ctx, connector.Operation{
		Name: "jev_choice_research", Method: http.MethodPost,
		Path: "/v1/systemone", SuccessStatusCode: http.StatusOK, RetrySafe: false,
	}, connector.Request{Body: body})
	if err != nil {
		return req, wire, nil, err
	}
	out, err := validateResponse(wire.Body, a.model, a.question.Criteria)
	return req, wire, out, err
}

// validateResponse is an offline research oracle for the maintainer's stated
// contract, not a new production validator. native.validateDistribution is
// unexported and links native inference; no production code is changed here.
// Preserve the original probabilities: never normalize, drop labels, or use
// confidence to fill missing mass.
func validateResponse(raw []byte, model string, labels map[string]string) (*response, error) {
	var out response
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, fmt.Errorf("invalid response JSON")
	}
	if out.Model != model {
		return nil, fmt.Errorf("response model does not match pinned model")
	}
	a, ok := out.Answers["intent"]
	if !ok || len(out.Answers) != 1 || a.Type != "choice" {
		return nil, fmt.Errorf("expected exactly one intent Choice answer")
	}
	if len(labels) == 0 || len(a.Probabilities) != len(labels) {
		return nil, fmt.Errorf("label set mismatch")
	}
	var sum, maximum float64
	for label := range labels {
		p, present := a.Probabilities[label]
		if !present || p == nil || !probability(*p) {
			return nil, fmt.Errorf("missing or invalid label probability")
		}
		sum += *p
		maximum = math.Max(maximum, *p)
	}
	if math.Abs(sum-1) > 1e-3 {
		return nil, fmt.Errorf("label probabilities do not sum to one")
	}
	selected, ok := a.Probabilities[a.Choice]
	if !ok || selected == nil || *selected != maximum {
		return nil, fmt.Errorf("choice is not a maximum-probability configured label")
	}
	if a.Confidence == nil || !probability(*a.Confidence) {
		return nil, fmt.Errorf("missing or invalid confidence")
	}
	return &out, nil
}

func probability(p float64) bool { return !math.IsNaN(p) && !math.IsInf(p, 0) && p >= 0 && p <= 1 }

func validateQuestion(q question) error {
	if q.Type != "choice" || strings.TrimSpace(q.Instructions) == "" || len(q.Criteria) < 2 || len(q.Criteria) > 255 {
		return fmt.Errorf("question requires type choice, instructions, and 2..255 labels")
	}
	for label, description := range q.Criteria {
		if strings.TrimSpace(label) == "" || strings.TrimSpace(description) == "" {
			return fmt.Errorf("labels and descriptions must be nonempty")
		}
	}
	return nil
}
