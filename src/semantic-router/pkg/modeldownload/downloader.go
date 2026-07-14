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

// ErrGatedModelSkipped identifies a download that could not proceed because a
// required gated model is unavailable without credentials. Callers must not
// treat this as readiness: RequiredModelsUnavailableError keeps startup fail
// closed while still allowing CI and diagnostics to distinguish the cause.
var ErrGatedModelSkipped = errors.New("gated model unavailable")

// RequiredModelsUnavailableError is the typed, non-ready outcome returned when
// any required model remains missing after the download pass.
type RequiredModelsUnavailableError struct {
	PendingModels []string
	GatedModels   []string
	FailedModels  []string
}

func (e *RequiredModelsUnavailableError) Error() string {
	if e == nil {
		return "required router models are unavailable"
	}
	return fmt.Sprintf(
		"%d required router models remain unavailable (%d gated, %d failed)",
		len(e.PendingModels),
		len(e.GatedModels),
		len(e.FailedModels),
	)
}

func (e *RequiredModelsUnavailableError) Unwrap() error {
	if e != nil && len(e.GatedModels) > 0 {
		return ErrGatedModelSkipped
	}
	return nil
}

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
	// A configured token means authentication was attempted. A 401/403 can then
	// indicate a bad token, missing grant, or misspelled repo and must fail hard.
	if strings.TrimSpace(hfToken) != "" {
		return false
	}

	errStr := strings.ToLower(err.Error())
	repoIDLower := strings.ToLower(repoID)

	knownGatedModels := map[string]struct{}{
		"google/embeddinggemma-300m": {},
	}
	_, isKnownGated := knownGatedModels[repoIDLower]

	// Only explicit authentication/gated-access evidence is admissible. A 404
	// or "repository not found" is ambiguous and commonly means a typo, so it
	// must never become a credential-free skip.
	isAuthError := containsAny(errStr,
		"401",
		"403",
		"gated repository",
		"gated repo",
		"gated model",
		"authentication required",
		"access to this model is restricted",
	)

	// A missing token is not proof that a failure came from a gated repository:
	// DNS, TLS, disk, and CLI failures have the same exit status. Only explicit
	// authentication evidence or a maintained known-gated model is skippable.
	return isKnownGated || isAuthError
}

func containsAny(value string, candidates ...string) bool {
	for _, candidate := range candidates {
		if strings.Contains(value, candidate) {
			return true
		}
	}
	return false
}

// DownloadModelWithProgress downloads a model with real-time progress output
func DownloadModelWithProgress(spec ModelSpec, config DownloadConfig) error {
	if err := validateModelDownloadDestination(spec.LocalPath); err != nil {
		return err
	}
	if err := validateModelSource(spec); err != nil {
		return err
	}
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
			logging.Warnf("Required gated model '%s' (repo: %s) is unavailable: %v", spec.LocalPath, spec.RepoID, err)
			logging.Warnf("Set HF_TOKEN with access to the gated repository; router startup remains non-ready")
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
	// Validate the complete untrusted inventory before filesystem probes,
	// progress logs, or process execution. DownloadModelWithProgress repeats
	// these checks at the final exec seam as defense in depth.
	for _, spec := range specs {
		if err := validateModelDownloadDestination(spec.LocalPath); err != nil {
			return fmt.Errorf("invalid model download destination: %w", err)
		}
		if err := validateModelSource(spec); err != nil {
			return fmt.Errorf("invalid model download source: %w", err)
		}
	}

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

	result := downloadMissingModels(missing, config, specs, &pendingModels, &readyCount, reporter)

	if len(pendingModels) > 0 {
		unavailable := &RequiredModelsUnavailableError{
			PendingModels: cloneStrings(pendingModels),
			GatedModels:   cloneStrings(result.gatedModels),
			FailedModels:  cloneStrings(result.failedModels),
		}
		reportProgress(reporter, ProgressState{
			Phase:         "failed",
			PendingModels: cloneStrings(pendingModels),
			ReadyModels:   readyCount,
			TotalModels:   len(specs),
			Message:       unavailable.Error(),
		})
		return unavailable
	}

	logging.Infof("Successfully downloaded all %d models", result.successCount)

	reportProgress(reporter, ProgressState{
		Phase:       "completed",
		ReadyModels: readyCount,
		TotalModels: len(specs),
		Message:     "All required router models are ready.",
	})

	return nil
}

// downloadMissingModels downloads each missing model serially, reporting progress.
// Required gated and failed models remain pending and never contribute to
// ReadyModels.
func downloadMissingModels(
	missing []ModelSpec,
	config DownloadConfig,
	specs []ModelSpec,
	pendingModels *[]string,
	readyCount *int,
	reporter ProgressReporter,
) downloadResult {
	result := downloadResult{}
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
			if errors.Is(err, ErrGatedModelSkipped) {
				result.gatedModels = append(result.gatedModels, spec.LocalPath)
				logging.Warnf("%s remains pending because gated-model credentials are unavailable", spec.LocalPath)
				reportProgress(reporter, ProgressState{
					Phase:         "blocked",
					PendingModels: cloneStrings(*pendingModels),
					ReadyModels:   *readyCount,
					TotalModels:   len(specs),
					Message:       fmt.Sprintf("Required gated model %s remains unavailable", spec.LocalPath),
				})
				continue
			}
			result.failedModels = append(result.failedModels, spec.LocalPath)
			logging.Warnf("Failed to download model %s: %v", spec.RepoID, err)
			continue
		}
		result.successCount++
		(*readyCount)++
		*pendingModels = removeString(*pendingModels, spec.LocalPath)
		reportProgress(reporter, ProgressState{
			Phase:         "checking",
			PendingModels: cloneStrings(*pendingModels),
			ReadyModels:   *readyCount,
			TotalModels:   len(specs),
			Message:       fmt.Sprintf("Model %s is ready", spec.LocalPath),
		})
	}
	return result
}

type downloadResult struct {
	successCount int
	gatedModels  []string
	failedModels []string
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
