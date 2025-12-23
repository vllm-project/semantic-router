package modeldownload

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// DownloadModel downloads a model using huggingface-cli
func DownloadModel(spec ModelSpec, config DownloadConfig) error {
	return DownloadModelWithProgress(spec, config)
}

// DownloadModelWithProgress downloads a model with real-time progress output
func DownloadModelWithProgress(spec ModelSpec, config DownloadConfig) error {
	logging.Infof("Downloading model: %s", spec.LocalPath)

	// Build huggingface-cli command
	args := []string{
		"download",
		spec.RepoID,
		"--local-dir", spec.LocalPath,
	}

	// Add revision if specified
	if spec.Revision != "" && spec.Revision != "main" {
		args = append(args, "--revision", spec.Revision)
	}

	cmd := exec.Command("hf", args...)

	// Set environment variables
	env := os.Environ()
	if config.HFEndpoint != "" {
		env = append(env, fmt.Sprintf("HF_ENDPOINT=%s", config.HFEndpoint))
	}
	if config.HFToken != "" {
		env = append(env, fmt.Sprintf("HF_TOKEN=%s", config.HFToken))
	}
	if config.HFHome != "" {
		env = append(env, fmt.Sprintf("HF_HOME=%s", config.HFHome))
	}
	cmd.Env = env

	// Stream output in real-time to stdout/stderr
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr

	// Run command with real-time output
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("failed to download model %s: %w", spec.RepoID, err)
	}

	logging.Infof("Successfully downloaded model: %s", spec.LocalPath)

	return nil
}

// EnsureModels ensures all required models are downloaded
func EnsureModels(specs []ModelSpec, config DownloadConfig) error {
	// Check which models are missing
	missing, err := GetMissingModels(specs)
	if err != nil {
		return fmt.Errorf("failed to check models: %w", err)
	}

	// Build set of missing paths for quick lookup
	missingPaths := make(map[string]bool)
	for _, spec := range missing {
		missingPaths[spec.LocalPath] = true
	}

	// Log status of each model
	for _, spec := range specs {
		if missingPaths[spec.LocalPath] {
			logging.Infof("✗ %s (need download)", spec.LocalPath)
		} else {
			logging.Infof("✓ %s (ready)", spec.LocalPath)
		}
	}

	if len(missing) == 0 {
		logging.Infof("All %d models are ready", len(specs))
		return nil
	}

	// Download missing models serially
	successCount := 0
	for _, spec := range missing {
		if err := DownloadModelWithProgress(spec, config); err != nil {
			logging.Warnf("Failed to download model %s: %v", spec.RepoID, err)
			continue
		}
		successCount++
	}

	for _, spec := range missing {
		requiredFiles := spec.RequiredFiles
		if len(requiredFiles) == 0 {
			requiredFiles = DefaultRequiredFiles
		}
		complete, err := IsModelComplete(spec.LocalPath, requiredFiles)
		if err != nil {
			continue
		}
		if !complete {
			// Get missing files list
			for _, file := range requiredFiles {
				if _, statErr := os.Stat(filepath.Join(spec.LocalPath, file)); os.IsNotExist(statErr) {
					// File is missing, will be downloaded
					_ = statErr // Suppress unused variable warning
				}
			}
		}
	}

	if successCount < len(missing) {
		return fmt.Errorf("failed to download %d out of %d models", len(missing)-successCount, len(missing))
	}

	return nil
}

// CheckHuggingFaceCLI checks if huggingface-cli is available
func CheckHuggingFaceCLI() error {
	// Try 'hf env' command first (new recommended command)
	cmd := exec.Command("hf", "env")
	output, err := cmd.CombinedOutput()
	if err == nil {
		// 'hf' command succeeded, extract version from output
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "huggingface_hub version:") {
				version := strings.TrimSpace(strings.TrimPrefix(line, "- huggingface_hub version:"))
				logging.Debugf("huggingface-cli version: %s", version)
				return nil
			}
		}
		// Version line not found, but command succeeded
		logging.Infof("Found huggingface-cli (hf command available)")
		return nil
	}

	// If 'hf' command fails, try legacy 'huggingface-cli' command
	cmd = exec.Command("huggingface-cli", "--help")
	if helpErr := cmd.Run(); helpErr != nil {
		return fmt.Errorf("huggingface-cli not found: %w\nPlease install it with: pip install huggingface_hub[cli]", helpErr)
	}

	// CLI exists, log without triggering deprecation warning
	logging.Infof("Found huggingface-cli (using legacy command)")
	return nil
}
