//go:build !windows && cgo

package apiserver

import (
	"errors"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/jsonunicode"
)

var errInvalidInferenceJSONUnicode = errors.New("request body must contain valid Unicode")

// parseInferenceJSONRequest validates raw Unicode before encoding/json can
// lossily replace invalid UTF-8 or unpaired UTF-16 surrogate escapes with
// U+FFFD. It is shared by every native embedding/similarity JSON surface.
func (s *ClassificationAPIServer) parseInferenceJSONRequest(r *http.Request, destination interface{}) error {
	body, err := readJSONRequestBody(r, defaultJSONRequestBodyLimit)
	if err != nil {
		return err
	}
	if !jsonunicode.Valid(body) {
		return errInvalidInferenceJSONUnicode
	}
	return decodeJSONBody(body, destination)
}
