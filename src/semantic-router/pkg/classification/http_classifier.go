package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// probabilitySumTolerance bounds how far an http_classify response's scores
// may drift from summing to 1.0, tolerating floating-point rounding from
// whatever serialized them without accepting a response with an
// unaccounted-for probability mass.
const probabilitySumTolerance = 0.02

// HTTPClassifierJailbreakInference implements SequenceClassifierBackend by
// calling an external sequence-classifier endpoint over HTTP. The wire
// contract mirrors the widely-used HuggingFace text-classification pipeline
// shape: a request carrying the input text, and a response listing every
// label's score (not just the top prediction), so a server wrapping an
// existing `transformers` pipeline or a Text Embeddings Inference (TEI)
// classify endpoint can be plugged in with minimal glue:
//
//	POST {endpoint}/classify
//	  {"inputs": "<text>"}
//	  -> [{"label": "safe", "score": 0.99}, {"label": "jailbreak", "score": 0.01}]
//
// Response labels are matched against the configured jailbreak_mapping by
// name (not by array position, which the server is free to order however it
// likes - e.g. sorted by score) so the resulting class index always lines up
// with the same mapping every other backend uses.
type HTTPClassifierJailbreakInference struct {
	httpClient *http.Client
	baseURL    string
	accessKey  string
	timeout    time.Duration
	mapping    *JailbreakMapping
}

// NewHTTPClassifierJailbreakInference creates a new http_classify-backed
// jailbreak inference instance from an external model config.
func NewHTTPClassifierJailbreakInference(cfg *config.ExternalModelConfig, mapping *JailbreakMapping) (*HTTPClassifierJailbreakInference, error) {
	if cfg.ModelEndpoint.Address == "" {
		return nil, fmt.Errorf("http_classify endpoint address is required for guardrail")
	}
	if mapping == nil {
		return nil, fmt.Errorf("jailbreak mapping is required for http_classify")
	}

	scheme := cfg.ModelEndpoint.Protocol
	if scheme == "" {
		scheme = "http"
	}
	baseURL := fmt.Sprintf("%s://%s:%d", scheme, cfg.ModelEndpoint.Address, cfg.ModelEndpoint.Port)

	timeout := 30 * time.Second
	if cfg.TimeoutSeconds > 0 {
		timeout = time.Duration(cfg.TimeoutSeconds) * time.Second
	}

	return &HTTPClassifierJailbreakInference{
		httpClient: &http.Client{Timeout: timeout},
		baseURL:    baseURL,
		accessKey:  cfg.AccessKey,
		timeout:    timeout,
		mapping:    mapping,
	}, nil
}

type httpClassifyRequest struct {
	Inputs string `json:"inputs"`
}

type httpClassifyLabelScore struct {
	Label string  `json:"label"`
	Score float32 `json:"score"`
}

// Classify implements the SequenceClassifierBackend interface.
func (h *HTTPClassifierJailbreakInference) Classify(text string) (candle_binding.ClassResultWithProbs, error) {
	ctx, cancel := context.WithTimeout(context.Background(), h.timeout)
	defer cancel()

	httpReq, err := h.buildClassifyRequest(ctx, text)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}

	scores, err := h.doClassifyRequest(httpReq)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, err
	}

	return alignScoresToMapping(h.mapping, scores)
}

// buildClassifyRequest builds the outgoing http_classify HTTP request.
func (h *HTTPClassifierJailbreakInference) buildClassifyRequest(ctx context.Context, text string) (*http.Request, error) {
	reqBody, err := json.Marshal(httpClassifyRequest{Inputs: text})
	if err != nil {
		return nil, fmt.Errorf("failed to marshal http_classify request: %w", err)
	}

	url := fmt.Sprintf("%s/classify", h.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create http_classify request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if h.accessKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", h.accessKey))
	}
	return httpReq, nil
}

// doClassifyRequest sends the request and parses the label/score list from a
// successful response.
func (h *HTTPClassifierJailbreakInference) doClassifyRequest(httpReq *http.Request) ([]httpClassifyLabelScore, error) {
	resp, err := h.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("http_classify request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read http_classify response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("http_classify endpoint returned status %d: %s", resp.StatusCode, string(body))
	}

	var scores []httpClassifyLabelScore
	if err := json.Unmarshal(body, &scores); err != nil {
		return nil, fmt.Errorf("failed to parse http_classify response: %w", err)
	}
	if len(scores) == 0 {
		return nil, fmt.Errorf("http_classify response contained no labels")
	}
	return scores, nil
}

// alignScoresToMapping matches response labels against the jailbreak_mapping
// by name (not by array position) and builds the full-distribution result.
// It requires the response to be a complete, valid distribution over exactly
// the configured labels - not just "at least one label matched" - because an
// incomplete response (e.g. a top-k response that omits the positive label)
// would otherwise default the missing entries to 0.0 and silently under-report
// risk instead of surfacing the mismatch.
func alignScoresToMapping(mapping *JailbreakMapping, scores []httpClassifyLabelScore) (candle_binding.ClassResultWithProbs, error) {
	numClasses := mapping.GetJailbreakTypeCount()
	probabilities := make([]float32, numClasses)
	seenIdx := make([]bool, numClasses)
	seenLabels := make([]string, 0, len(scores))
	bestIdx := -1
	var bestScore, sum float32

	for _, s := range scores {
		seenLabels = append(seenLabels, s.Label)
		idx, ok := mapping.GetIndexForJailbreakType(s.Label)
		if !ok || idx < 0 || idx >= numClasses {
			return candle_binding.ClassResultWithProbs{}, fmt.Errorf(
				"http_classify response label %q is not in the configured jailbreak_mapping", s.Label)
		}
		if seenIdx[idx] {
			return candle_binding.ClassResultWithProbs{}, fmt.Errorf(
				"http_classify response contains duplicate label %q", s.Label)
		}
		if score64 := float64(s.Score); math.IsNaN(score64) || math.IsInf(score64, 0) || s.Score < 0 || s.Score > 1 {
			return candle_binding.ClassResultWithProbs{}, fmt.Errorf(
				"http_classify response label %q has an invalid score %v (must be finite and within [0, 1])", s.Label, s.Score)
		}

		seenIdx[idx] = true
		probabilities[idx] = s.Score
		sum += s.Score
		if bestIdx == -1 || s.Score > bestScore {
			bestIdx = idx
			bestScore = s.Score
		}
	}

	for idx, present := range seenIdx {
		if present {
			continue
		}
		missingLabel, _ := mapping.GetJailbreakTypeFromIndex(idx)
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf(
			"http_classify response is missing label %q from the configured jailbreak_mapping (got %v)", missingLabel, seenLabels)
	}

	if math.Abs(float64(sum)-1.0) > probabilitySumTolerance {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf(
			"http_classify response scores sum to %v, want ~1.0 (labels: %v)", sum, seenLabels)
	}

	return candle_binding.ClassResultWithProbs{
		Class:         bestIdx,
		Confidence:    bestScore,
		Probabilities: probabilities,
	}, nil
}
