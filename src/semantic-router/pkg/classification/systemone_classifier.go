package classification

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
)

// systemOneMaxOptions is the Choice arity the SystemOne request contract
// accepts. A rule declaring more labels than this cannot be served, so it is
// rejected at configuration time rather than on every request.
const systemOneMaxOptions = 255

// systemOneTieTolerance is how far below the most probable option a choice may
// sit and still count as tied with it, the tolerance #4086 uses for the same
// check.
const systemOneTieTolerance = 2e-5

// SystemOneClassifierInference implements SequenceClassifierBackend by asking a
// SystemOne endpoint one Choice question whose options are the signal's
// declared labels. Unlike http_classify, the question travels with the request:
//
//	POST {endpoint}/v1/systemone
//	  {"state": "<text>", "model": "<model>",
//	   "questions": {"<signal>": {"type": "choice",
//	                              "instructions": "<rule instructions>",
//	                              "criteria": {"safe": null, "unsafe": null}}}}
//	  -> {"model": "<model>", "usage": {...},
//	      "answers": {"<signal>": {"type": "choice", "choice": "safe",
//	                               "confidence": 0.96,
//	                               "probabilities": {"safe": 0.98, "unsafe": 0.02}}}}
//
// An absolute endpoint address, such as http://host/v1, is the API base and
// only /systemone is appended to it, the target remoteOperationIdentity records.
//
// The answer's probabilities are keyed by option name, so they go through the
// same alignScoresToMapping validator http_classify uses and land on the same
// class indices every other backend produces. A serving endpoint that answers
// more than the one question asked, answers it with a different question type,
// names another model, chooses an option its own distribution does not rank
// first, or reports a confidence outside [0, 1] is a contract violation rather
// than a result, so it fails instead of reporting the labels it did return.
// Confidence is not recomputed from the distribution: #4086 documents its
// value as Decision's own statistic, not one other SystemOne providers share.
type SystemOneClassifierInference struct {
	connector    *connector.Client
	operation    connector.Operation
	timeout      time.Duration
	mapping      sequenceLabelMapping
	model        string
	questionID   string
	instructions string
	options      []string
}

// NewSystemOneClassifierInference creates an http_systemone-backed inference
// instance. questionID names the question in the request and the answer map; it
// is the signal's own name so a response can be attributed without positional
// assumptions.
func NewSystemOneClassifierInference(
	cfg *config.ExternalModelConfig,
	mapping sequenceLabelMapping,
	questionID string,
	instructions string,
	deadline time.Duration,
) (*SystemOneClassifierInference, error) {
	if cfg == nil {
		return nil, fmt.Errorf("http_systemone external model config is required")
	}
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("http_systemone endpoint address is required")
	}
	if strings.TrimSpace(cfg.ModelName) == "" {
		return nil, fmt.Errorf("http_systemone requires llm_model_name: the request carries an explicit model")
	}
	if isNilMapping(mapping) {
		return nil, fmt.Errorf("label mapping is required for http_systemone")
	}
	if strings.TrimSpace(questionID) == "" {
		return nil, fmt.Errorf("http_systemone question id is required")
	}
	if strings.TrimSpace(instructions) == "" {
		return nil, fmt.Errorf("http_systemone requires instructions: the Choice contract rejects a null question")
	}
	// The Choice contract accepts two to 255 options, and the label mapping is
	// what the options are built from, so both bounds are checked here as well
	// as in config validation: this constructor is reachable from an
	// already-decoded config.
	options, err := systemOneOptionsFromMapping(mapping)
	if err != nil {
		return nil, err
	}

	address := strings.TrimSpace(cfg.ModelEndpoint.Address)
	baseURL, operationPath := address, "/systemone"
	if !strings.Contains(address, "://") {
		scheme := strings.ToLower(strings.TrimSpace(cfg.ModelEndpoint.Protocol))
		if scheme == "" {
			scheme = "http"
		}
		baseURL, operationPath = fmt.Sprintf("%s://%s:%d", scheme, address, cfg.ModelEndpoint.Port), "/v1/systemone"
	}

	// A SystemOne call is a typed forward pass rather than a generative one, so
	// it shares http_classify's fail-fast default instead of http_chat's.
	timeout := 5 * time.Second
	if deadline > 0 {
		timeout = deadline
	} else if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	remote, err := connector.New(baseURL, bearerAuthorizer(cfg.AccessKey), connector.Options{
		AttemptTimeout:   timeout,
		MaxRetries:       1,
		MaxRequestBytes:  cfg.GetMaxRequestBytes(),
		MaxResponseBytes: cfg.GetMaxResponseBytes(),
		MaxErrorBytes:    maxClassifyErrorBodyBytes,
	})
	if err != nil {
		return nil, fmt.Errorf("create http_systemone connector: %w", err)
	}

	return &SystemOneClassifierInference{
		connector: remote,
		operation: connector.Operation{
			Name:      "http_systemone",
			Method:    http.MethodPost,
			Path:      operationPath,
			RetrySafe: true,
		},
		timeout:      timeout,
		mapping:      mapping,
		model:        strings.TrimSpace(cfg.ModelName),
		questionID:   questionID,
		instructions: instructions,
		options:      options,
	}, nil
}

// systemOneOptionsFromMapping reads the declared labels in class-index order.
// Option order does not change the answer, but a stable request body keeps
// request logs and caches comparable across restarts.
func systemOneOptionsFromMapping(mapping sequenceLabelMapping) ([]string, error) {
	count := mapping.LabelCount()
	if count < 2 {
		return nil, fmt.Errorf("http_systemone label mapping must define at least 2 labels, got %d", count)
	}
	if count > systemOneMaxOptions {
		return nil, fmt.Errorf(
			"http_systemone label mapping defines %d labels, more than the %d a Choice question accepts",
			count, systemOneMaxOptions)
	}
	options := make([]string, 0, count)
	for index := 0; index < count; index++ {
		label, ok := mapping.LabelFromIndex(index)
		if !ok {
			return nil, fmt.Errorf("http_systemone label mapping has no label for class index %d", index)
		}
		options = append(options, label)
	}
	return options, nil
}

type systemOneChoiceQuestion struct {
	Type         string             `json:"type"`
	Instructions string             `json:"instructions"`
	Criteria     map[string]*string `json:"criteria"`
}

type systemOneRequest struct {
	State     string                             `json:"state"`
	Model     string                             `json:"model"`
	Questions map[string]systemOneChoiceQuestion `json:"questions"`
}

type systemOneAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Confidence    float64            `json:"confidence"`
	Probabilities map[string]float64 `json:"probabilities"`
}

type systemOneResponse struct {
	Model   string                     `json:"model"`
	Answers map[string]systemOneAnswer `json:"answers"`
}

// Classify implements the SequenceClassifierBackend interface. The deadline
// comes from the caller's ctx bounded by the configured timeout, so a caller
// that has already given up cancels the remote call too.
func (s *SystemOneClassifierInference) Classify(ctx context.Context, text string) (SequenceClassificationResult, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	// Option descriptions are optional in the contract and the declared labels
	// carry no descriptions, so every criterion is an explicit null rather than
	// an invented gloss the model would then be asked to match.
	criteria := make(map[string]*string, len(s.options))
	for _, option := range s.options {
		criteria[option] = nil
	}
	reqBody, err := json.Marshal(systemOneRequest{
		State: text,
		Model: s.model,
		Questions: map[string]systemOneChoiceQuestion{
			s.questionID: {Type: "choice", Instructions: s.instructions, Criteria: criteria},
		},
	})
	if err != nil {
		return SequenceClassificationResult{}, fmt.Errorf("failed to marshal http_systemone request: %w", err)
	}
	responseBody, err := s.connector.Do(ctx, s.operation, reqBody)
	if err != nil {
		return SequenceClassificationResult{}, formatSystemOneConnectorError(err)
	}

	var decoded systemOneResponse
	if err = json.Unmarshal(responseBody, &decoded); err != nil {
		return SequenceClassificationResult{}, fmt.Errorf("failed to parse http_systemone response: %w", err)
	}
	scores, err := s.scoresFromAnswers(decoded)
	if err != nil {
		return SequenceClassificationResult{}, err
	}
	return alignScoresToMapping(s.mapping, scores)
}

// scoresFromAnswers resolves the one answer this request asked for and turns its
// option distribution into the label/score pairs alignScoresToMapping validates.
func (s *SystemOneClassifierInference) scoresFromAnswers(decoded systemOneResponse) ([]httpClassifyLabelScore, error) {
	if decoded.Model != s.model {
		return nil, fmt.Errorf("http_systemone response names model %q, want the requested %q", decoded.Model, s.model)
	}
	if len(decoded.Answers) != 1 {
		return nil, fmt.Errorf(
			"http_systemone response answered %d questions, want exactly the 1 asked", len(decoded.Answers))
	}
	answer, ok := decoded.Answers[s.questionID]
	if !ok {
		return nil, fmt.Errorf("http_systemone response has no answer for question %q", s.questionID)
	}
	if answer.Type != "choice" {
		return nil, fmt.Errorf(
			"http_systemone answer for question %q has type %q, want \"choice\"", s.questionID, answer.Type)
	}
	// Reading the options in declared order keeps the checks below independent
	// of map iteration order.
	scores := make([]httpClassifyLabelScore, 0, len(s.options))
	for _, option := range s.options {
		probability, ok := answer.Probabilities[option]
		if !ok {
			return nil, fmt.Errorf("http_systemone answer for question %q has no probability for option %q", s.questionID, option)
		}
		scores = append(scores, httpClassifyLabelScore{Label: option, Score: float32(probability)})
	}
	if len(answer.Probabilities) != len(s.options) {
		return nil, fmt.Errorf(
			"http_systemone answer for question %q has %d probabilities, want the %d options asked",
			s.questionID, len(answer.Probabilities), len(s.options))
	}
	if err := checkSystemOneChoice(s.questionID, s.options, answer); err != nil {
		return nil, err
	}
	return scores, nil
}

// checkSystemOneChoice rejects an answer whose choice is not a most probable
// option, which would otherwise route on scores its own choice disagrees with,
// and an answer whose confidence is not a probability.
func checkSystemOneChoice(questionID string, options []string, answer systemOneAnswer) error {
	top := math.Inf(-1)
	for _, option := range options {
		top = math.Max(top, answer.Probabilities[option])
	}
	chosen, ok := answer.Probabilities[answer.Choice]
	if !ok || top-chosen > systemOneTieTolerance {
		return fmt.Errorf(
			"http_systemone answer for question %q chose %q, which is not a most probable option", questionID, answer.Choice)
	}
	if answer.Confidence < 0 || answer.Confidence > 1 {
		return fmt.Errorf(
			"http_systemone answer for question %q has confidence %v, want a value in [0, 1]", questionID, answer.Confidence)
	}
	return nil
}

func formatSystemOneConnectorError(err error) error {
	var connectorErr *connector.Error
	if !errors.As(err, &connectorErr) {
		return fmt.Errorf("http_systemone request failed: %w", err)
	}
	switch connectorErr.Kind {
	case connector.KindStatus:
		body, truncated := connectorErr.ResponseBody()
		return fmt.Errorf(
			"http_systemone endpoint returned status %d: %s (truncated=%t): %w",
			connectorErr.StatusCode, string(body), truncated, connectorErr,
		)
	case connector.KindResponse:
		return fmt.Errorf("failed to read http_systemone response: %w", connectorErr)
	default:
		return fmt.Errorf("http_systemone request failed: %w", connectorErr)
	}
}

// Close releases idle connections owned by the remote connector.
func (s *SystemOneClassifierInference) Close() error {
	if s == nil || s.connector == nil {
		return nil
	}
	return s.connector.Close()
}
