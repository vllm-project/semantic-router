//go:build !windows && cgo

package apiserver

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// defaultJSONRequestBodyLimit bounds generic JSON API payloads. Config and KB
// mutation endpoints need more room than classifier prompts, while endpoint-
// specific handlers can still pass a tighter limit.
const defaultJSONRequestBodyLimit int64 = 10 * 1024 * 1024

var errRequestBodyTooLarge = errors.New("request body too large")

func (s *ClassificationAPIServer) parseJSONRequest(r *http.Request, v interface{}) error {
	return s.parseJSONRequestWithLimit(r, v, defaultJSONRequestBodyLimit)
}

func (s *ClassificationAPIServer) parseJSONRequestWithLimit(r *http.Request, v interface{}, maxBytes int64) error {
	body, err := readJSONRequestBody(r, maxBytes)
	if err != nil {
		return err
	}
	return decodeJSONBody(body, v)
}

func readJSONRequestBody(r *http.Request, maxBytes int64) ([]byte, error) {
	defer func() {
		_ = r.Body.Close()
	}()

	if maxBytes <= 0 {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			return nil, fmt.Errorf("failed to read request body: %w", err)
		}
		return body, nil
	}

	body, err := io.ReadAll(io.LimitReader(r.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("failed to read request body: %w", err)
	}
	if int64(len(body)) > maxBytes {
		return nil, fmt.Errorf("%w: exceeds %d bytes", errRequestBodyTooLarge, maxBytes)
	}
	return body, nil
}

func decodeJSONBody(body []byte, v interface{}) error {
	if err := json.Unmarshal(body, v); err != nil {
		return fmt.Errorf("failed to parse JSON: %w", err)
	}
	return nil
}

func (s *ClassificationAPIServer) writeJSONRequestError(w http.ResponseWriter, err error) {
	if errors.Is(err, errRequestBodyTooLarge) {
		s.writeErrorResponse(w, http.StatusRequestEntityTooLarge, "REQUEST_BODY_TOO_LARGE", err.Error())
		return
	}
	s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", err.Error())
}
