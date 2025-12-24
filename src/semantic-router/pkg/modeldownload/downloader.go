package modeldownload

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// hfCommand stores the detected HuggingFace CLI command ("hf" or "huggingface-cli")
var hfCommand string

// ErrGatedModelSkipped is a sentinel error indicating a gated model was gracefully skipped
var ErrGatedModelSkipped = fmt.Errorf("gated model skipped")

// DownloadModel downloads a model using huggingface-cli
func DownloadModel(spec ModelSpec, config DownloadConfig) error {
	return DownloadModelWithProgress(spec, config)
}

// IsGatedModelError checks if an error indicates a gated model that requires authentication
func IsGatedModelError(err error, repoID string) bool {
	if err == nil {
		return false
	}

	errStr := strings.ToLower(err.Error())
	repoIDLower := strings.ToLower(repoID)

	// Known gated models
	knownGatedModels := []string{"embeddinggemma", "gemma"}
	isKnownGated := false
	for _, gatedName := range knownGatedModels {
		if strings.Contains(repoIDLower, gatedName) {
			isKnownGated = true
			break
		}
	}

	// Check for authentication-related error patterns
	isAuthError := strings.Contains(errStr, "401") ||
		strings.Contains(errStr, "unauthorized") ||
		strings.Contains(errStr, "gated") ||
		strings.Contains(errStr, "repository not found") ||
		strings.Contains(errStr, "404") ||
		strings.Contains(errStr, "authentication required")

	return isKnownGated || isAuthError
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

	// Use detected CLI command, default to "hf"
	cliCmd := hfCommand
	if cliCmd == "" {
		cliCmd = "hf"
	}
	cmd := exec.Command(cliCmd, args...)

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
		// Check if this is a gated model error
		if IsGatedModelError(err, spec.RepoID) {
			logging.Warnf("⚠️  Skipping gated model '%s' (repo: %s): %v", spec.LocalPath, spec.RepoID, err)
			logging.Warnf("   This is expected if HF_TOKEN is not available (e.g., PRs from forks)")
			logging.Warnf("   To download gated models, set HF_TOKEN environment variable")
			return fmt.Errorf("%w: %s", ErrGatedModelSkipped, spec.RepoID)
		}
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
	skippedCount := 0
	for _, spec := range missing {
		if err := DownloadModelWithProgress(spec, config); err != nil {
			// Check if this was a gated model that was gracefully skipped
			if errors.Is(err, ErrGatedModelSkipped) || strings.Contains(err.Error(), ErrGatedModelSkipped.Error()) {
				skippedCount++
				logging.Infof("✓ %s (skipped - gated model, HF_TOKEN not available)", spec.LocalPath)
				continue
			}
			logging.Warnf("Failed to download model %s: %v", spec.RepoID, err)
			continue
		}
		successCount++
	}

	// Only return error if we failed to download non-gated models
	// Gated models that were skipped don't count as failures
	if successCount+skippedCount < len(missing) {
		return fmt.Errorf("failed to download %d out of %d models", len(missing)-successCount-skippedCount, len(missing))
	}

	if skippedCount > 0 {
		logging.Infof("Downloaded %d models, skipped %d gated models (HF_TOKEN not available)", successCount, skippedCount)
	} else {
		logging.Infof("Successfully downloaded all %d models", successCount)
	}

	return nil
}

// CheckHuggingFaceCLI checks if huggingface-cli is available and sets hfCommand
func CheckHuggingFaceCLI() error {
	// Try 'hf env' command first (new recommended command)
	cmd := exec.Command("hf", "env")
	output, err := cmd.CombinedOutput()
	if err == nil {
		hfCommand = "hf"
		// Extract version from output
		lines := strings.Split(string(output), "\n")
		for _, line := range lines {
			if strings.Contains(line, "huggingface_hub version:") {
				version := strings.TrimSpace(strings.TrimPrefix(line, "- huggingface_hub version:"))
				logging.Debugf("huggingface-cli version: %s", version)
				return nil
			}
		}
		logging.Infof("Found huggingface-cli (hf command available)")
		return nil
	}

	// If 'hf' command fails, try legacy 'huggingface-cli' command
	cmd = exec.Command("huggingface-cli", "--help")
	if helpErr := cmd.Run(); helpErr != nil {
		return fmt.Errorf("huggingface-cli not found: %w\nPlease install it with: pip install huggingface_hub[cli]", helpErr)
	}

	hfCommand = "huggingface-cli"
	logging.Infof("Found huggingface-cli (using legacy command)")
	return nil
}
