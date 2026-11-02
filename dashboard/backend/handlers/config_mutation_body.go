package handlers

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
)

// ConfigMutationBodyLimit is the maximum request body accepted by the dashboard
// config mutation handlers. The handlers enforce it with http.MaxBytesReader,
// which counts bytes as they are read and does not trust Content-Length.
const ConfigMutationBodyLimit int64 = 2 << 20

// readLimitedMutationBody reads one config mutation body up to
// ConfigMutationBodyLimit. Over the limit it writes HTTP 413 and returns false.
func readLimitedMutationBody(w http.ResponseWriter, r *http.Request) ([]byte, bool) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, ConfigMutationBodyLimit))
	if err == nil {
		return body, true
	}
	var tooLarge *http.MaxBytesError
	if errors.As(err, &tooLarge) {
		http.Error(w, "Request body too large", http.StatusRequestEntityTooLarge)
		return nil, false
	}
	http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
	return nil, false
}

// decodeLimitedJSON reads a bounded body and accepts exactly one JSON value.
// A second value is HTTP 400. A body over ConfigMutationBodyLimit is HTTP 413,
// including trailing whitespace that a single Decoder.Decode would not read.
func decodeLimitedJSON(w http.ResponseWriter, r *http.Request, dest any) bool {
	body, ok := readLimitedMutationBody(w, r)
	if !ok {
		return false
	}
	decoder := json.NewDecoder(bytes.NewReader(body))
	if err := decoder.Decode(dest); err != nil {
		http.Error(w, fmt.Sprintf("Invalid request body: %v", err), http.StatusBadRequest)
		return false
	}
	var trailing json.RawMessage
	if err := decoder.Decode(&trailing); !errors.Is(err, io.EOF) {
		http.Error(w, "Invalid request body: expected a single JSON value", http.StatusBadRequest)
		return false
	}
	return true
}
