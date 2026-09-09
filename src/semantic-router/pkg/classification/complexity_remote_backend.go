package classification

import (
	"bytes"
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

// scoringHTTPBackend implements ScoringBackend over the shared connector.
//
// It cannot reuse HTTPClassifierInference: that type exists to align a label
// distribution onto a mapping and therefore requires one with at least two
// labels. score.v1 has no labels at all - the score is the whole product - so
// this is a separate, thinner reader over the same transport, which keeps the
// connector's timeout, retry, byte caps and error mapping shared.
type scoringHTTPBackend struct {
	connector *connector.Client
	timeout   time.Duration
}

var scoreOperation = connector.Operation{
	Name:      "http_classify_score",
	Method:    http.MethodPost,
	Path:      "/classify",
	RetrySafe: true,
}

// scoreEntry is the HuggingFace text-classification response shape. score.v1
// reads it because it shares the http_classify protocol, so a regression head
// deployed behind the usual endpoint needs no shim; a bare {"score": x} object
// decodes into the same struct. The label is meaningless on this contract and
// is read only to keep the decoder honest about the shape.
type scoreEntry struct {
	Label string `json:"label"`
	// A pointer, so an absent field and an explicit null are distinguishable
	// from a reported 0. Zero is a meaningful score - the easiest end of a
	// [0,1] scorer's range - so decoding a malformed response into it would
	// turn a backend fault into a confident verdict.
	Score *float64 `json:"score"`
}

func newScoringHTTPBackend(cfg *config.ExternalModelConfig, deadline time.Duration) (ScoringBackend, error) {
	if cfg == nil {
		return nil, fmt.Errorf("score.v1 external model config is required")
	}
	if strings.TrimSpace(cfg.ModelEndpoint.Address) == "" {
		return nil, fmt.Errorf("score.v1 endpoint address is required")
	}

	scheme := strings.ToLower(strings.TrimSpace(cfg.ModelEndpoint.Protocol))
	if scheme == "" {
		scheme = "http"
	}
	baseURL := fmt.Sprintf("%s://%s:%d", scheme, strings.TrimSpace(cfg.ModelEndpoint.Address), cfg.ModelEndpoint.Port)

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
		return nil, fmt.Errorf("create score.v1 connector: %w", err)
	}

	return &scoringHTTPBackend{connector: remote, timeout: timeout}, nil
}

// Score implements ScoringBackend. The deadline comes from the caller's ctx
// bounded by the configured timeout, so a caller that gives up cancels the
// outbound request instead of leaving it running.
//
// The value is returned unclamped: a regression head is not a probability, and
// what a given number means is decided by the rule's boundaries, not here.
func (s *scoringHTTPBackend) Score(ctx context.Context, text string) (float64, error) {
	ctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	reqBody, err := json.Marshal(httpClassifyRequest{Inputs: text})
	if err != nil {
		return 0, fmt.Errorf("failed to marshal score.v1 request: %w", err)
	}
	responseBody, err := s.connector.Do(ctx, scoreOperation, reqBody)
	if err != nil {
		return 0, formatHTTPClassifyConnectorError(err)
	}

	return decodeScoreResponse(responseBody)
}

// decodeScoreResponse reads the score out of either shape a score.v1 endpoint
// may return: the HuggingFace text-classification array with exactly one
// entry, or a bare {"score": x} object. The first byte tells them apart, so
// accepting both adds no ambiguity - and it spares a regression endpoint that
// was never a classifier from wrapping its one number in a one-element list
// under a label it has no use for.
func decodeScoreResponse(body []byte) (float64, error) {
	// A UTF-8 byte-order mark is legal to emit and some runtimes prepend one.
	// Left in place it makes the first byte neither '[' nor '{', and the shape
	// check below would blame the response format - sending the operator to
	// rewrite a payload that was already correct.
	trimmed := bytes.TrimPrefix(body, []byte("\xef\xbb\xbf"))
	trimmed = bytes.TrimLeft(trimmed, " \t\r\n")
	if len(trimmed) == 0 {
		return 0, fmt.Errorf("score.v1 response is empty")
	}
	var entry scoreEntry
	switch trimmed[0] {
	case '[':
		var entries []scoreEntry
		if err := json.Unmarshal(trimmed, &entries); err != nil {
			return 0, fmt.Errorf("failed to parse score.v1 response: %w", err)
		}
		// Anything other than exactly one entry leaves "which is the score"
		// undefined, so it fails rather than picking one.
		if len(entries) != 1 {
			return 0, fmt.Errorf("score.v1 response must carry exactly one entry, got %d", len(entries))
		}
		entry = entries[0]
	case '{':
		if err := json.Unmarshal(trimmed, &entry); err != nil {
			return 0, fmt.Errorf("failed to parse score.v1 response: %w", err)
		}
	default:
		return 0, fmt.Errorf("score.v1 response must be a JSON object or a one-entry array, got %q", string(trimmed[:1]))
	}
	if entry.Score == nil {
		return 0, fmt.Errorf("score.v1 response entry carries no score")
	}
	score := *entry.Score
	// Every comparison against NaN is false, so a rule would answer medium for
	// it regardless of its boundaries; an infinity makes one verdict
	// unreachable. Neither can be compared meaningfully.
	if math.IsNaN(score) || math.IsInf(score, 0) {
		return 0, fmt.Errorf("score.v1 response carries a non-finite score: %v", score)
	}
	return score, nil
}

// Close releases idle connections owned by the remote connector. A classifier
// is rebuilt per recipe and again on every dynamic-config reload, so a backend
// that is never closed leaks an idle HTTP transport each time.
func (s *scoringHTTPBackend) Close() error {
	if s == nil || s.connector == nil {
		return nil
	}
	return s.connector.Close()
}
