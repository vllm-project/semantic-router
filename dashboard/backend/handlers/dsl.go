package handlers

import (
	"encoding/json"
	"io"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/dsl/editor"
)

const maxDSLSourceBytes = 1 << 20

// Bound concurrent editor work without leaving timed-out compilation goroutines
// running in the background. Editor operations only transform supplied text.
var dslEditorSlots = make(chan struct{}, 4)

func DSLEditorHandler(operation string) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			return
		}
		var input struct {
			Source *string `json:"source"`
		}
		decoder := json.NewDecoder(http.MaxBytesReader(w, r.Body, 2*maxDSLSourceBytes))
		decoder.DisallowUnknownFields()
		if err := decoder.Decode(&input); err != nil || input.Source == nil || len(*input.Source) > maxDSLSourceBytes {
			http.Error(w, "Expected bounded JSON source", http.StatusBadRequest)
			return
		}
		if err := decoder.Decode(new(any)); err != io.EOF {
			http.Error(w, "Expected one JSON document", http.StatusBadRequest)
			return
		}
		select {
		case dslEditorSlots <- struct{}{}:
			defer func() { <-dslEditorSlots }()
		default:
			w.Header().Set("Retry-After", "1")
			http.Error(w, "Editor is busy; retry shortly", http.StatusServiceUnavailable)
			return
		}
		if r.Context().Err() != nil {
			return
		}
		var result any
		switch operation {
		case "compile":
			result = editor.Compile(*input.Source)
		case "validate":
			result = editor.Validate(*input.Source)
		case "parse":
			result = editor.Parse(*input.Source)
		case "decompile":
			result = editor.Decompile(*input.Source)
		case "format":
			result = editor.Format(*input.Source)
		default:
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "no-store")
		_ = json.NewEncoder(w).Encode(result)
	}
}
