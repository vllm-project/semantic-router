package modeldownload

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// maxCaptureBytes bounds how much CLI output we retain for error classification.
const maxCaptureBytes = 64 * 1024

// tailWriter keeps only the most recent maxCaptureBytes written to it. The HF CLI
// can stream a large volume of progress output, so capturing all of it would be
// wasteful; the error text we classify on always lives at the tail.
type tailWriter struct {
	buf []byte
}

func (w *tailWriter) Write(p []byte) (int, error) {
	w.buf = append(w.buf, p...)
	if len(w.buf) > maxCaptureBytes {
		w.buf = w.buf[len(w.buf)-maxCaptureBytes:]
	}
	return len(p), nil
}

func (w *tailWriter) String() string { return string(w.buf) }

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

// IsGatedModelError reports whether a download failure is due to the model being
// gated behind HuggingFace authentication, as opposed to a transient or public-repo
// failure (rate limit, network error) or a missing/typo'd repo that must fail loudly.
//
// cliOutput is the captured stdout/stderr from the HF CLI. The CLI writes its real
// error text there rather than returning it in err (which is usually just
// "exit status 1"), so classification must inspect the captured output, not err
// alone. The presence or absence of an HF_TOKEN is deliberately NOT consulted: an
// invalid repo id, a rate limit, and a gated repo are all indistinguishable by token
// state, and HuggingFace returns 429 for several of them regardless of token.
func IsGatedModelError(err error, cliOutput string, repoID string) bool {
	if err == nil {
		return false
	}

	haystack := strings.ToLower(err.Error() + "\n" + cliOutput)

	// Auth/gated patterns the HF CLI emits for repos that require access approval.
	// Note: 404 / "repository not found" is intentionally absent — a missing or
	// mistyped repo is a real error that must surface, not a graceful gated skip.
	gatedPatterns := []string{
		"gated",
		"restricted",
		"awaiting a review",
		"must be authenticated",
		"access to model",
		"access to this repo",
		"unauthorized",
		"authentication required",
		"401",
		"403",
	}
	for _, p := range gatedPatterns {
		if strings.Contains(haystack, p) {
			return true
		}
	}

	// Fallback for when the CLI emits nothing classifiable (e.g. only
	// "exit status 1" with no captured output): a small allowlist of model
	// families that are known to be gated on HuggingFace.
	repoIDLower := strings.ToLower(repoID)
	for _, gatedName := range []string{"embeddinggemma", "gemma"} {
		if strings.Contains(repoIDLower, gatedName) {
			return true
		}
	}

	return false
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

	// Stream output in real-time while capturing stderr (bounded) so download
	// failures can be classified. The HF CLI writes its real error text to stderr
	// rather than returning it in err, which is usually just "exit status 1".
	var captured tailWriter
	cmd.Stdout = os.Stdout
	cmd.Stderr = io.MultiWriter(os.Stderr, &captured)

	// Run command with real-time output
	if err := cmd.Run(); err != nil {
		if IsGatedModelError(err, captured.String(), spec.RepoID) {
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
			logging.ComponentDebugEvent("router", "required_model_status", map[string]interface{}{
				"model_ref":         spec.LocalPath,
				"status":            "missing",
				"requires_download": true,
			})
		} else {
			logging.ComponentDebugEvent("router", "required_model_status", map[string]interface{}{
				"model_ref":         spec.LocalPath,
				"status":            "ready",
				"requires_download": false,
			})
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
