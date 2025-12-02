package model

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// ModelInfo represents information about a model
type ModelInfo struct {
	ID           string
	Name         string
	Path         string
	Type         string // "lora" or "legacy"
	Architecture string // "bert", "roberta", "modernbert"
	Downloaded   bool
	Size         int64
	Purpose      string // "intent", "pii", "security", "base"
}

// ModelManager handles model operations
type ModelManager struct {
	ModelsDir string
}

// NewModelManager creates a new model manager
func NewModelManager(modelsDir string) *ModelManager {
	if modelsDir == "" {
		modelsDir = "./models"
	}
	return &ModelManager{
		ModelsDir: modelsDir,
	}
}

// ListModels lists all models (downloaded and configured)
func (mm *ModelManager) ListModels() ([]ModelInfo, error) {
	// Ensure models directory exists
	if _, err := os.Stat(mm.ModelsDir); os.IsNotExist(err) {
		cli.Info(fmt.Sprintf("Models directory does not exist: %s", mm.ModelsDir))
		return []ModelInfo{}, nil
	}

	// Discover models using existing functionality
	paths, err := classification.AutoDiscoverModels(mm.ModelsDir)
	if err != nil {
		return nil, fmt.Errorf("failed to discover models: %w", err)
	}

	var models []ModelInfo

	// Add LoRA models if found
	if paths.HasLoRAModels() {
		models = append(models, ModelInfo{
			ID:           "lora-intent-classifier",
			Name:         "LoRA Intent Classifier",
			Path:         paths.LoRAIntentClassifier,
			Type:         "lora",
			Architecture: paths.LoRAArchitecture,
			Downloaded:   true,
			Size:         mm.getDirectorySize(paths.LoRAIntentClassifier),
			Purpose:      "intent",
		})
		models = append(models, ModelInfo{
			ID:           "lora-pii-detector",
			Name:         "LoRA PII Detector",
			Path:         paths.LoRAPIIClassifier,
			Type:         "lora",
			Architecture: paths.LoRAArchitecture,
			Downloaded:   true,
			Size:         mm.getDirectorySize(paths.LoRAPIIClassifier),
			Purpose:      "pii",
		})
		models = append(models, ModelInfo{
			ID:           "lora-security-classifier",
			Name:         "LoRA Security Classifier",
			Path:         paths.LoRASecurityClassifier,
			Type:         "lora",
			Architecture: paths.LoRAArchitecture,
			Downloaded:   true,
			Size:         mm.getDirectorySize(paths.LoRASecurityClassifier),
			Purpose:      "security",
		})
	}

	// Add legacy models if found
	if paths.HasLegacyModels() {
		if paths.ModernBertBase != "" {
			models = append(models, ModelInfo{
				ID:           "modernbert-base",
				Name:         "ModernBERT Base",
				Path:         paths.ModernBertBase,
				Type:         "legacy",
				Architecture: "modernbert",
				Downloaded:   true,
				Size:         mm.getDirectorySize(paths.ModernBertBase),
				Purpose:      "base",
			})
		}
		if paths.IntentClassifier != "" {
			models = append(models, ModelInfo{
				ID:           "intent-classifier",
				Name:         "Intent Classifier",
				Path:         paths.IntentClassifier,
				Type:         "legacy",
				Architecture: "modernbert",
				Downloaded:   true,
				Size:         mm.getDirectorySize(paths.IntentClassifier),
				Purpose:      "intent",
			})
		}
		if paths.PIIClassifier != "" {
			models = append(models, ModelInfo{
				ID:           "pii-classifier",
				Name:         "PII Classifier",
				Path:         paths.PIIClassifier,
				Type:         "legacy",
				Architecture: "modernbert",
				Downloaded:   true,
				Size:         mm.getDirectorySize(paths.PIIClassifier),
				Purpose:      "pii",
			})
		}
		if paths.SecurityClassifier != "" {
			models = append(models, ModelInfo{
				ID:           "security-classifier",
				Name:         "Security Classifier",
				Path:         paths.SecurityClassifier,
				Type:         "legacy",
				Architecture: "modernbert",
				Downloaded:   true,
				Size:         mm.getDirectorySize(paths.SecurityClassifier),
				Purpose:      "security",
			})
		}
	}

	return models, nil
}

// ValidateModel validates a specific model
func (mm *ModelManager) ValidateModel(modelID string) error {
	models, err := mm.ListModels()
	if err != nil {
		return err
	}

	// Find the model
	var targetModel *ModelInfo
	for i := range models {
		if models[i].ID == modelID {
			targetModel = &models[i]
			break
		}
	}

	if targetModel == nil {
		return fmt.Errorf("model not found: %s", modelID)
	}

	// Check if directory exists
	if _, err := os.Stat(targetModel.Path); os.IsNotExist(err) {
		return fmt.Errorf("model directory does not exist: %s", targetModel.Path)
	}

	// Check for essential model files
	essentialFiles := []string{"config.json"}
	modelFiles := []string{"pytorch_model.bin", "model.safetensors"}

	// Check essential files
	for _, file := range essentialFiles {
		filePath := filepath.Join(targetModel.Path, file)
		if _, err := os.Stat(filePath); os.IsNotExist(err) {
			return fmt.Errorf("missing essential file: %s", file)
		}
	}

	// Check at least one model file exists
	hasModelFile := false
	for _, file := range modelFiles {
		filePath := filepath.Join(targetModel.Path, file)
		if _, err := os.Stat(filePath); err == nil {
			hasModelFile = true
			break
		}
	}

	if !hasModelFile {
		return fmt.Errorf("no model weights found (expected pytorch_model.bin or model.safetensors)")
	}

	return nil
}

// ValidateAllModels validates all downloaded models
func (mm *ModelManager) ValidateAllModels() (map[string]error, error) {
	models, err := mm.ListModels()
	if err != nil {
		return nil, err
	}

	results := make(map[string]error)
	for _, model := range models {
		if model.Downloaded {
			results[model.ID] = mm.ValidateModel(model.ID)
		}
	}

	return results, nil
}

// GetModelInfo returns detailed information about a model
func (mm *ModelManager) GetModelInfo(modelID string) (*ModelInfo, error) {
	models, err := mm.ListModels()
	if err != nil {
		return nil, err
	}

	for i := range models {
		if models[i].ID == modelID {
			return &models[i], nil
		}
	}

	return nil, fmt.Errorf("model not found: %s", modelID)
}

// RemoveModel removes a model from disk
func (mm *ModelManager) RemoveModel(modelID string) error {
	model, err := mm.GetModelInfo(modelID)
	if err != nil {
		return err
	}

	if !model.Downloaded {
		return fmt.Errorf("model is not downloaded: %s", modelID)
	}

	// Remove the model directory
	if err := os.RemoveAll(model.Path); err != nil {
		return fmt.Errorf("failed to remove model directory: %w", err)
	}

	return nil
}

// DownloadModel downloads a model from HuggingFace
func (mm *ModelManager) DownloadModel(modelID string, progressCallback func(downloaded, total int64)) error {
	// For now, this is a placeholder that calls the existing make command
	// In the future, this could be implemented with direct HuggingFace API calls
	cli.Warning("Model download currently uses the Makefile 'download-models' command")
	cli.Info("Downloading all configured models...")

	return fmt.Errorf("direct model download not yet implemented - use 'make download-models'")
}

// getDirectorySize calculates the total size of a directory
func (mm *ModelManager) getDirectorySize(path string) int64 {
	var size int64
	_ = filepath.Walk(path, func(_ string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			size += info.Size()
		}
		return nil
	})
	return size
}

// FormatSize formats a byte size in human-readable format
func FormatSize(bytes int64) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := int64(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB", float64(bytes)/float64(div), "KMGTPE"[exp])
}

// DownloadFile downloads a file from a URL with progress tracking
func DownloadFile(filepath string, url string, progressCallback func(downloaded, total int64)) error {
	// Create the file
	out, err := os.Create(filepath)
	if err != nil {
		return err
	}
	defer out.Close()

	// Get the data
	//nolint:gosec // G107: URL is constructed internally and validated
	resp, err := http.Get(url)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	// Check server response
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("bad status: %s", resp.Status)
	}

	// Create a progress reader if callback provided
	var reader io.Reader = resp.Body
	if progressCallback != nil {
		reader = &progressReader{
			reader:   resp.Body,
			total:    resp.ContentLength,
			callback: progressCallback,
		}
	}

	// Write the body to file
	_, err = io.Copy(out, reader)
	if err != nil {
		return err
	}

	return nil
}

// progressReader wraps an io.Reader to track progress
type progressReader struct {
	reader     io.Reader
	total      int64
	downloaded int64
	callback   func(downloaded, total int64)
	lastUpdate time.Time
}

func (pr *progressReader) Read(p []byte) (int, error) {
	n, err := pr.reader.Read(p)
	pr.downloaded += int64(n)

	// Call callback every 100ms to avoid too frequent updates
	if pr.callback != nil && time.Since(pr.lastUpdate) > 100*time.Millisecond {
		pr.callback(pr.downloaded, pr.total)
		pr.lastUpdate = time.Now()
	}

	return n, err
}

// GetModelStatus returns the overall status of models
func (mm *ModelManager) GetModelStatus() map[string]interface{} {
	// Use existing functionality
	return classification.GetModelDiscoveryInfo(mm.ModelsDir)
}

// ensureModelsDirectory creates the models directory if it doesn't exist
func (mm *ModelManager) ensureModelsDirectory() error {
	if _, err := os.Stat(mm.ModelsDir); os.IsNotExist(err) {
		if err := os.MkdirAll(mm.ModelsDir, 0o755); err != nil {
			return fmt.Errorf("failed to create models directory: %w", err)
		}
	}
	return nil
}

// GetConfiguredModels returns models configured in config file
// This would need to be implemented with config file parsing
func (mm *ModelManager) GetConfiguredModels() ([]string, error) {
	// Placeholder - would parse config file to get configured models
	return []string{}, nil
}

// isModelDownloaded checks if a model is downloaded
func (mm *ModelManager) isModelDownloaded(modelID string) bool {
	models, err := mm.ListModels()
	if err != nil {
		return false
	}

	for _, model := range models {
		if model.ID == modelID && model.Downloaded {
			return true
		}
	}
	return false
}

// getModelPath returns the expected path for a model
func (mm *ModelManager) getModelPath(modelID string) string {
	// Convert model ID to directory name
	dirName := strings.ReplaceAll(modelID, "-", "_")
	return filepath.Join(mm.ModelsDir, dirName)
}
