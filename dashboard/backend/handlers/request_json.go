package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

const (
	// Small control-plane requests should never need to carry documents.
	smallJSONRequestBodyLimit int64 = 64 << 10
	// Builder/setup requests may carry a DSL or one canonical config document.
	documentRequestBodyLimit int64 = 1 << 20
	// Deploy requests can carry the current DSL, fragment, and canonical base.
	deployJSONRequestBodyLimit int64 = 8 << 20
)

// decodeBoundedJSON reads exactly one strict JSON value through a hard byte
// limit. It rejects lossy Unicode decoding so identifiers and message content
// cannot change between validation, authorization, and persistence.
func decodeBoundedJSON(w http.ResponseWriter, r *http.Request, maxBytes int64, dst any) (int, error) {
	raw, status, err := readBoundedRequestBody(w, r, maxBytes)
	if err != nil {
		return status, err
	}
	return decodeStrictJSONBytes(raw, dst)
}

// readBoundedRequestBody applies the same hard transport boundary to raw YAML
// and proxy payloads that cannot use the JSON decoder.
func readBoundedRequestBody(w http.ResponseWriter, r *http.Request, maxBytes int64) ([]byte, int, error) {
	if r == nil || r.Body == nil || maxBytes <= 0 {
		return nil, http.StatusBadRequest, errors.New("request body is required")
	}
	if r.ContentLength > maxBytes {
		return nil, http.StatusRequestEntityTooLarge, fmt.Errorf("request body exceeds %d bytes", maxBytes)
	}
	body := http.MaxBytesReader(w, r.Body, maxBytes)
	raw, err := io.ReadAll(body)
	if err != nil {
		var tooLarge *http.MaxBytesError
		if errors.As(err, &tooLarge) {
			return nil, http.StatusRequestEntityTooLarge, fmt.Errorf("request body exceeds %d bytes", maxBytes)
		}
		return nil, http.StatusBadRequest, errors.New("request body could not be read")
	}
	return raw, 0, nil
}

// decodeStrictJSONBytes decodes one object without accepting unknown fields,
// trailing values, or lossy Unicode replacement. Callers must bound raw before
// invoking it (HTTP and WebSocket transports both impose a 64 KiB ceiling).
func decodeStrictJSONBytes(raw []byte, dst any) (int, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return http.StatusBadRequest, errors.New("request body is required")
	}
	if !jsonunicode.Valid(raw) {
		return http.StatusBadRequest, errors.New("request body contains invalid Unicode")
	}

	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(dst); err != nil {
		return http.StatusBadRequest, errors.New("request body is invalid")
	}
	var trailing any
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		return http.StatusBadRequest, errors.New("request body must contain one JSON value")
	}
	return 0, nil
}
