package modeldownload

import (
	"fmt"
	"os"
	"path/filepath"
)

// DefaultRequiredFiles are the files typically needed for a model to be considered complete
var DefaultRequiredFiles = []string{
	"config.json",
}

var modelWeightPatterns = []string{
	"model.safetensors",
	"model.safetensors.index.json",
	"pytorch_model.bin",
	"pytorch_model.bin.index.json",
	"pytorch_model-*.bin",
	"adapter_model.safetensors",
	"adapter_model.bin",
	"*.onnx",
	"*.pt",
	"*.safetensors",
	"*.bin",
}

// IsModelComplete checks if a model is fully downloaded by verifying required files exist
func IsModelComplete(localPath string, requiredFiles []string) (bool, error) {
	// Check if directory exists
	info, err := os.Stat(localPath)
	if err != nil {
		if os.IsNotExist(err) {
			return false, nil // Model doesn't exist, not an error
		}
		return false, fmt.Errorf("failed to stat model directory %s: %w", localPath, err)
	}

	if !info.IsDir() {
		return false, fmt.Errorf("model path %s is not a directory", localPath)
	}

	// Use default required files if none specified
	if len(requiredFiles) == 0 {
		requiredFiles = DefaultRequiredFiles
	}

	// Check each required file
	for _, file := range requiredFiles {
		filePath := filepath.Join(localPath, file)
		if _, err := os.Stat(filePath); err != nil {
			if os.IsNotExist(err) {
				return false, nil // File missing, model incomplete
			}
			return false, fmt.Errorf("failed to check file %s: %w", filePath, err)
		}
	}

	hasWeights, weightErr := hasModelWeights(localPath)
	if weightErr != nil {
		return false, weightErr
	}
	if !hasWeights {
		return false, nil
	}

	return true, nil
}

func hasModelWeights(localPath string) (bool, error) {
	for _, pattern := range modelWeightPatterns {
		matches, err := filepath.Glob(filepath.Join(localPath, pattern))
		if err != nil {
			return false, fmt.Errorf("failed to scan weight files in %s: %w", localPath, err)
		}
		if len(matches) > 0 {
			return true, nil
		}
	}
	return false, nil
}

// GetMissingModels returns a list of models that are not complete
func GetMissingModels(specs []ModelSpec) ([]ModelSpec, error) {
	var missing []ModelSpec

	for _, spec := range specs {
		complete, err := IsModelComplete(spec.LocalPath, spec.RequiredFiles)
		if err != nil {
			return nil, fmt.Errorf("failed to check model %s: %w", spec.LocalPath, err)
		}

		if !complete {
			missing = append(missing, spec)
		}
	}

	return missing, nil
}
