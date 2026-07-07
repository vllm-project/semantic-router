package config

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// legacyMmBertLayers is the historical hardcoded early-exit layer set. It is
// used only as a fallback when a model ships without an onnx/model_config.json
// manifest, so validation stays backward compatible instead of rejecting every
// layer.
var legacyMmBertLayers = []int{3, 6, 11, 22}

// mmBertModelConfig is the subset of the model's onnx/model_config.json that
// declares which early-exit layers the shipped model actually provides.
type mmBertModelConfig struct {
	AvailableLayers []int `json:"available_layers"`
}

// MmBertAvailableLayers returns the early-exit layers the mmBERT model at
// modelPath supports, read from its own onnx/model_config.json
// (available_layers). This manifest is the single source of truth: the layers
// must match the ONNX sessions the model actually ships, not a hardcoded list
// that can drift from the model.
//
// Falls back to the historical default layer set when the manifest is missing,
// unreadable, or does not declare available_layers.
func MmBertAvailableLayers(modelPath string) []int {
	if modelPath == "" {
		return legacyMmBertLayers
	}

	manifestPath := filepath.Join(modelPath, "onnx", "model_config.json")
	data, err := os.ReadFile(manifestPath)
	if err != nil {
		return legacyMmBertLayers
	}

	var cfg mmBertModelConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return legacyMmBertLayers
	}
	if len(cfg.AvailableLayers) == 0 {
		return legacyMmBertLayers
	}
	return cfg.AvailableLayers
}

// IsValidMmBertLayer reports whether layer is one of the model's advertised
// early-exit layers.
func IsValidMmBertLayer(layer int, available []int) bool {
	for _, l := range available {
		if l == layer {
			return true
		}
	}
	return false
}
