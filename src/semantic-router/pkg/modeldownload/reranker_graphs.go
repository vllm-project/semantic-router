package modeldownload

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Resolve file availability only. The provider still validates the trained
// heads, representation contract and selected graph's execution metadata.
func requiredRerankerGraphGroups(spec ModelSpec) ([][]string, error) {
	groups := append([][]string(nil), spec.RequiredFileGroups...)
	if len(spec.RerankerSelections) == 0 {
		return groups, nil
	}
	data, err := os.ReadFile(filepath.Join(spec.LocalPath, "config.json"))
	if err != nil {
		return nil, err
	}
	var encoder struct {
		Layers    int `json:"num_hidden_layers"`
		Dimension int `json:"hidden_size"`
	}
	if err = json.Unmarshal(data, &encoder); err != nil {
		return nil, fmt.Errorf("read reranker encoder dimensions: %w", err)
	}
	if encoder.Layers <= 0 || encoder.Dimension <= 0 {
		return nil, fmt.Errorf("reranker encoder dimensions must be positive")
	}
	for _, selection := range spec.RerankerSelections {
		layer, dimension := selection.Layer, selection.Dimension
		if layer == 0 {
			layer = encoder.Layers
		}
		if dimension == 0 {
			dimension = encoder.Dimension
		}
		files := []string{
			fmt.Sprintf("onnx/model_layer_%d_dim_%d.onnx", layer, dimension),
			fmt.Sprintf("onnx/layer-%d/dim-%d/model.onnx", layer, dimension),
		}
		if layer == encoder.Layers && dimension == encoder.Dimension {
			files = append(files, "onnx/model.onnx")
		}
		groups = append(groups, files)
	}
	return groups, nil
}
