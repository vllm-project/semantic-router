package embedding

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

func (p *OpenAICompatibleProvider) responseByteLimit(batchSize int) int64 {
	dimension := p.Dimension()
	if dimension <= 0 {
		dimension = unknownEmbeddingDimensionForBudget
	}
	return embeddingResponseByteLimit(batchSize, dimension)
}

func embeddingResponseByteLimit(batchSize int, dimension int) int64 {
	if batchSize <= 0 {
		return embeddingResponseBaseBytes
	}
	if dimension <= 0 {
		dimension = unknownEmbeddingDimensionForBudget
	}
	maxVectorValues := int64(maxEmbeddingResponseBytes-embeddingResponseBaseBytes-embeddingResponsePerVectorBytes) /
		embeddingResponsePerValueBytes
	if int64(dimension) > maxVectorValues {
		return maxEmbeddingResponseBytes
	}
	perVector := int64(embeddingResponsePerVectorBytes) + int64(dimension)*embeddingResponsePerValueBytes
	maxVectors := (maxEmbeddingResponseBytes - embeddingResponseBaseBytes) / perVector
	if int64(batchSize) > maxVectors {
		return maxEmbeddingResponseBytes
	}
	return embeddingResponseBaseBytes + int64(batchSize)*perVector
}

func decodeEmbeddingResponse(resp *http.Response, limit int64) (embeddingsResponse, error) {
	if resp.ContentLength > limit {
		return embeddingsResponse{}, responseTooLargeError(limit)
	}
	payload, err := io.ReadAll(io.LimitReader(resp.Body, limit+1))
	if err != nil {
		return embeddingsResponse{}, newEmbeddingResponseError(
			embeddingResponseReadFailure,
			"embedding provider response could not be read",
			err,
		)
	}
	if int64(len(payload)) > limit {
		return embeddingsResponse{}, responseTooLargeError(limit)
	}

	decoder := json.NewDecoder(bytes.NewReader(payload))
	var decoded embeddingsResponse
	if err := decoder.Decode(&decoded); err != nil {
		return embeddingsResponse{}, newEmbeddingResponseError(
			embeddingResponseInvalidJSON,
			"embedding provider returned invalid JSON",
			err,
		)
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return embeddingsResponse{}, newEmbeddingResponseError(
			embeddingResponseTrailingData,
			"embedding provider response contains trailing data",
			err,
		)
	}
	return decoded, nil
}

func responseTooLargeError(limit int64) error {
	return newEmbeddingResponseError(
		embeddingResponseTooLarge,
		fmt.Sprintf("embedding provider response exceeds the %d-byte limit", limit),
		nil,
	)
}

func newEmbeddingResponseError(kind embeddingResponseErrorKind, message string, cause error) error {
	return &embeddingResponseError{kind: kind, message: message, cause: cause}
}

type embeddingHTTPError struct {
	statusCode int
	message    string
	retryable  bool
}

func (e *embeddingHTTPError) Error() string {
	return e.message
}

func responseError(resp *http.Response) error {
	// Drain small error bodies to EOF so net/http can reuse the connection. A
	// malicious or broken upstream cannot make us consume more than this cap.
	_, _ = io.Copy(io.Discard, io.LimitReader(resp.Body, maxErrorBodyDrainBytes))
	message := fmt.Sprintf("embedding provider returned status %d (%s)", resp.StatusCode, http.StatusText(resp.StatusCode))
	if resp.StatusCode == http.StatusUnauthorized || resp.StatusCode == http.StatusForbidden {
		message = fmt.Sprintf("embedding provider authentication failed with status %d", resp.StatusCode)
	}
	return &embeddingHTTPError{
		statusCode: resp.StatusCode,
		message:    message,
		retryable:  resp.StatusCode == http.StatusTooManyRequests || resp.StatusCode >= http.StatusInternalServerError,
	}
}
