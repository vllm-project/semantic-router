// Command selectorparity replays an exported model-selection artifact through
// the router's selectors. The training parity tests send it a JSON request on
// stdin and compare its selections with the trained Python model's.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
)

type query struct {
	Embedding []float64 `json:"embedding"`
	Category  string    `json:"category"`
}

type request struct {
	Algorithm  string          `json:"algorithm"`
	Artifact   json.RawMessage `json:"artifact"`
	Candidates []string        `json:"candidates"`
	Queries    []query         `json:"queries"`
}

// response carries a load error, or one selection per query (null when the
// selector rejected that query).
type response struct {
	Error      string    `json:"error,omitempty"`
	Selections []*string `json:"selections"`
}

func main() {
	var req request
	if err := json.NewDecoder(os.Stdin).Decode(&req); err != nil {
		fmt.Fprintf(os.Stderr, "decode request: %v\n", err)
		os.Exit(2)
	}
	if err := json.NewEncoder(os.Stdout).Encode(replay(req)); err != nil {
		fmt.Fprintf(os.Stderr, "encode response: %v\n", err)
		os.Exit(2)
	}
}

func replay(req request) response {
	dir, err := os.MkdirTemp("", "selector-parity-")
	if err != nil {
		return response{Error: err.Error()}
	}
	defer func() { _ = os.RemoveAll(dir) }()
	if err = os.WriteFile(filepath.Join(dir, req.Algorithm+"_model.json"), req.Artifact, 0o600); err != nil {
		return response{Error: err.Error()}
	}
	selector, err := modelselection.NewSelector(&config.MLModelSelectionConfig{Type: req.Algorithm, ModelsPath: dir})
	if err != nil {
		return response{Error: err.Error()}
	}
	refs := make([]config.ModelRef, len(req.Candidates))
	for i, name := range req.Candidates {
		refs[i] = config.ModelRef{Model: name}
	}
	selections := make([]*string, len(req.Queries))
	for i, q := range req.Queries {
		ctx := &modelselection.SelectionContext{QueryEmbedding: q.Embedding, CategoryName: q.Category}
		if ref, err := selector.Select(ctx, refs); err == nil {
			name := ref.Model
			selections[i] = &name
		}
	}
	return response{Selections: selections}
}
