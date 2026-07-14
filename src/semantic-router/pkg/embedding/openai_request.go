package embedding

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
)

func (p *OpenAICompatibleProvider) newEmbeddingBatchRequest(
	ctx context.Context,
	texts []string,
	apiKey string,
) (*http.Request, context.CancelFunc, error) {
	body, err := json.Marshal(embeddingsRequest{Model: p.model, Input: texts, Dimensions: p.dimensions})
	if err != nil {
		return nil, nil, err
	}

	attemptCtx := ctx
	cancel := context.CancelFunc(func() {})
	if p.timeout > 0 {
		attemptCtx, cancel = context.WithTimeout(ctx, p.timeout)
	}

	req, err := http.NewRequestWithContext(attemptCtx, http.MethodPost, p.endpoint, bytes.NewReader(body))
	if err != nil {
		cancel()
		return nil, nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	if apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	return req, cancel, nil
}
