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

// ProgressState captures downloader progress for external readiness reporting.
type ProgressState struct {
	Phase            string
	DownloadingModel string
	PendingModels    []string
	ReadyModels      int
	TotalModels      int
	Message          string
}

// ProgressReporter receives model download progress updates.
type ProgressReporter func(ProgressState)

// DownloadModel downloads a model using huggingface-cli
func DownloadModel(spec ModelSpec, config DownloadConfig) error {
	return DownloadModelWithProgress(spec, config)
}

// IsGatedModelError checks if an error indicates a gated model that requires authentication.
// hfToken is the HF_TOKEN value from the download config; an empty string means no token is set.
func IsGatedModelError(err error, repoID string, hfToken string) bool {
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

	// When no HF_TOKEN is set, the HF CLI streams its output directly to os.Stderr
	// (not captured in err), so auth failures surface only as a plain "exit status 1".
	// Treat any download failure without a token as a soft skip so the router
	// degrades gracefully instead of crashing (e.g., CI forks without secrets).
	noToken := hfToken == ""

	return isKnownGated || isAuthError || noToken
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
		if IsGatedModelError(err, spec.RepoID, config.HFToken) {
			logging.Warnf("⚠️  Skipping model '%s' (repo: %s): %v", spec.LocalPath, spec.RepoID, err)
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
	return EnsureModelsWithProgress(specs, config, nil)
}

// EnsureModelsWithProgress ensures all required models are downloaded and reports progress.
func EnsureModelsWithProgress(specs []ModelSpec, config DownloadConfig, reporter ProgressReporter) error {
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
			logging.Infof("%s (ready)", spec.LocalPath)
		}
	}

	pendingModels := make([]string, 0, len(missing))
	for _, spec := range missing {
		pendingModels = append(pendingModels, spec.LocalPath)
	}
	readyCount := len(specs) - len(missing)
	reportProgress(reporter, ProgressState{
		Phase:         "checking",
		PendingModels: cloneStrings(pendingModels),
		ReadyModels:   readyCount,
		TotalModels:   len(specs),
		Message:       "Checking required router models...",
	})

	if len(missing) == 0 {
		logging.Infof("All %d models are ready", len(specs))
		reportProgress(reporter, ProgressState{
			Phase:       "completed",
			ReadyModels: len(specs),
			TotalModels: len(specs),
			Message:     "All required router models are ready.",
		})
		return nil
	}

	successCount, skippedCount := downloadMissingModels(missing, config, specs, &pendingModels, &readyCount, reporter)

	if successCount+skippedCount < len(missing) {
		return fmt.Errorf("failed to download %d out of %d models", len(missing)-successCount-skippedCount, len(missing))
	}

	if skippedCount > 0 {
		logging.Infof("Downloaded %d models, skipped %d gated models (HF_TOKEN not available)", successCount, skippedCount)
	} else {
		logging.Infof("Successfully downloaded all %d models", successCount)
	}

	reportProgress(reporter, ProgressState{
		Phase:       "completed",
		ReadyModels: readyCount,
		TotalModels: len(specs),
		Message:     "All required router models are ready.",
	})

	return nil
}

// downloadMissingModels downloads each missing model serially, reporting progress.
// Returns the number of successfully downloaded and gracefully skipped models.
func downloadMissingModels(
	missing []ModelSpec,
	config DownloadConfig,
	specs []ModelSpec,
	pendingModels *[]string,
	readyCount *int,
	reporter ProgressReporter,
) (successCount, skippedCount int) {
	for _, spec := range missing {
		reportProgress(reporter, ProgressState{
			Phase:            "downloading",
			DownloadingModel: spec.LocalPath,
			PendingModels:    cloneStrings(*pendingModels),
			ReadyModels:      *readyCount,
			TotalModels:      len(specs),
			Message:          fmt.Sprintf("Downloading model %s", spec.LocalPath),
		})
		if err := DownloadModelWithProgress(spec, config); err != nil {
			if errors.Is(err, ErrGatedModelSkipped) || strings.Contains(err.Error(), ErrGatedModelSkipped.Error()) {
				skippedCount++
				logging.Infof("%s (skipped - gated model, HF_TOKEN not available)", spec.LocalPath)
				*readyCount++
				*pendingModels = removeString(*pendingModels, spec.LocalPath)
				reportProgress(reporter, ProgressState{
					Phase:         "checking",
					PendingModels: cloneStrings(*pendingModels),
					ReadyModels:   *readyCount,
					TotalModels:   len(specs),
					Message:       fmt.Sprintf("Skipped gated model %s", spec.LocalPath),
				})
				continue
			}
			logging.Warnf("Failed to download model %s: %v", spec.RepoID, err)
			continue
		}
		successCount++
		*readyCount++
		*pendingModels = removeString(*pendingModels, spec.LocalPath)
		reportProgress(reporter, ProgressState{
			Phase:         "checking",
			PendingModels: cloneStrings(*pendingModels),
			ReadyModels:   *readyCount,
			TotalModels:   len(specs),
			Message:       fmt.Sprintf("Model %s is ready", spec.LocalPath),
		})
	}
	return successCount, skippedCount
}

func reportProgress(reporter ProgressReporter, state ProgressState) {
	if reporter != nil {
		reporter(state)
	}
}

func cloneStrings(values []string) []string {
	if len(values) == 0 {
		return nil
	}
	cloned := make([]string, len(values))
	copy(cloned, values)
	return cloned
}

func removeString(values []string, target string) []string {
	if len(values) == 0 {
		return values
	}
	next := make([]string, 0, len(values))
	removed := false
	for _, value := range values {
		if !removed && value == target {
			removed = true
			continue
		}
		next = append(next, value)
	}
	return next
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
