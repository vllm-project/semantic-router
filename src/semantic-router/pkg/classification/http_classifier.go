package classification

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

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

	reqBody, err := json.Marshal(httpClassifyRequest{Inputs: text})
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("failed to marshal http_classify request: %w", err)
	}

	url := fmt.Sprintf("%s/classify", h.baseURL)
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(reqBody))
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("failed to create http_classify request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")
	if h.accessKey != "" {
		httpReq.Header.Set("Authorization", fmt.Sprintf("Bearer %s", h.accessKey))
	}

	resp, err := h.httpClient.Do(httpReq)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("http_classify request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("failed to read http_classify response: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("http_classify endpoint returned status %d: %s", resp.StatusCode, string(body))
	}

	var scores []httpClassifyLabelScore
	if err := json.Unmarshal(body, &scores); err != nil {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("failed to parse http_classify response: %w", err)
	}
	if len(scores) == 0 {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("http_classify response contained no labels")
	}

	numClasses := h.mapping.GetJailbreakTypeCount()
	probabilities := make([]float32, numClasses)
	bestIdx := -1
	var bestScore float32
	var seenLabels []string
	for _, s := range scores {
		seenLabels = append(seenLabels, s.Label)
		idx, ok := h.mapping.GetIndexForJailbreakType(s.Label)
		if !ok || idx < 0 || idx >= numClasses {
			continue
		}
		probabilities[idx] = s.Score
		if bestIdx == -1 || s.Score > bestScore {
			bestIdx = idx
			bestScore = s.Score
		}
	}
	if bestIdx == -1 {
		return candle_binding.ClassResultWithProbs{}, fmt.Errorf("http_classify response labels %v did not match any configured jailbreak_mapping label", seenLabels)
	}

	return candle_binding.ClassResultWithProbs{
		Class:         bestIdx,
		Confidence:    bestScore,
		Probabilities: probabilities,
		NumClasses:    numClasses,
	}, nil
}
