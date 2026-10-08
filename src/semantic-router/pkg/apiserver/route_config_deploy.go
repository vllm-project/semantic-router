//go:build !windows

package apiserver

import (
	"context"
	"fmt"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/k8s/configwriter"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// deployMu ensures only one deploy operation at a time
var deployMu sync.Mutex

// maxBackups is the maximum number of config backups to keep
const maxBackups = 10

const configVersionTimestampLayout = "20060102-150405"

type configVersionSource string

const (
	configVersionSourceAPI        configVersionSource = "api"
	configVersionSourceRollback   configVersionSource = "rollback"
	configVersionSourceStartup    configVersionSource = "startup"
	configVersionSourceFile       configVersionSource = "file"
	configVersionSourceKubernetes configVersionSource = "kubernetes"
	configVersionSourceUnknown    configVersionSource = "unknown"
)

var knownConfigVersionSources = map[configVersionSource]configVersionSource{
	configVersionSourceAPI:        configVersionSourceAPI,
	configVersionSourceRollback:   configVersionSourceRollback,
	configVersionSourceStartup:    configVersionSourceStartup,
	configVersionSourceFile:       configVersionSourceFile,
	configVersionSourceKubernetes: configVersionSourceKubernetes,
}

// configVersionPattern accepts the timestamp version and its collision suffix.
// The rollback endpoint interpolates this value into a filesystem path, so the
// allowlist deliberately excludes every other character and prevents traversal
// out of the backup directory.
var configVersionPattern = regexp.MustCompile(`^[0-9]{8}-[0-9]{6}(?:-[0-9]{3,9})?$`)

// RouterConfigUpdateRequest is the JSON body for a router config update request.
type RouterConfigUpdateRequest struct {
	// YAML is the router config YAML payload.
	YAML string `json:"yaml"`
}

type routerConfigRollbackRequest struct {
	// Version names a backup by its timestamp, or a configuration snapshot
	// version by its number.
	Version string `json:"version,omitempty"`
	// ConfigVersion names a configuration snapshot version.
	ConfigVersion uint64 `json:"config_version,omitempty"`
}

// RouterConfigUpdateResponse is the JSON response for a router config mutation.
type RouterConfigUpdateResponse struct {
	Status               string `json:"status"`
	Version              string `json:"version"`
	ETag                 string `json:"etag,omitempty"`
	ActivationStatus     string `json:"activation_status,omitempty"`
	GeneratedRuntimeHash string `json:"generated_runtime_hash,omitempty"`
	// ConfigVersion is the configuration snapshot version the update
	// activated (its ACK); it is absent until the update is active.
	ConfigVersion uint64 `json:"config_version,omitempty"`
	// RollbackOf is the configuration version a rollback restored.
	RollbackOf uint64 `json:"rollback_of,omitempty"`
	Message    string `json:"message,omitempty"`
	// Activation is the update's attempt; a rejected one (its NACK) carries
	// structured reasons.
	Activation *configActivationResponse `json:"activation,omitempty"`
}

type configHashResponse struct {
	SourceConfigHash     string `json:"source_config_hash"`
	GeneratedRuntimeHash string `json:"generated_runtime_hash"`
	ActiveRuntimeHash    string `json:"active_runtime_hash,omitempty"`
	// ActiveVersion is the version of the configuration snapshot that serves.
	ActiveVersion    uint64                    `json:"active_version,omitempty"`
	ActivationStatus string                    `json:"activation_status"`
	Activation       *configActivationResponse `json:"activation,omitempty"`
	// LastRejection is the most recent rejected update, if any.
	LastRejection *configActivationResponse `json:"last_rejection,omitempty"`
}

// RouterConfigVersionEntry is one entry of the configuration history: a
// version the Router activated, or a document a mutation replaced.
type RouterConfigVersionEntry struct {
	Version   string              `json:"version"`
	Timestamp string              `json:"timestamp"`
	Source    configVersionSource `json:"source"`
	// Filename is the recorded document's file; empty when the entry
	// recorded no document.
	Filename string `json:"filename"`
	// ConfigVersion is the configuration snapshot version the entry
	// activated; absent for a replaced document that never activated as
	// recorded.
	ConfigVersion uint64 `json:"config_version,omitempty"`
	// Hash is the SHA-256 of the recorded document.
	Hash string `json:"hash,omitempty"`
	// Active marks the version that serves.
	Active bool `json:"active,omitempty"`
	// RollbackOf is the version a rollback entry restored.
	RollbackOf uint64 `json:"rollback_of,omitempty"`
}

// handleConfigRollback handles POST /api/v1/config/rollback.
func (s *ClassificationAPIServer) handleConfigRollback(w http.ResponseWriter, r *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	guard, ok := s.acquireConfigMutationGuard(w)
	if !ok {
		return
	}
	defer guard.Release()

	request, ok := s.parseRollbackVersion(w, r)
	if !ok {
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	target, ok := s.resolveRollbackTarget(w, request, configBackupDir(paths.sourcePath))
	if !ok {
		return
	}
	existingData, ok := s.loadCompatibleRollbackSource(
		w,
		paths.sourcePath,
		target.config,
	)
	if !ok {
		return
	}
	if !checkConfigPrecondition(w, r, existingData) {
		return
	}

	// Keep the current config recoverable before rolling back.
	_, backupDir, err := s.preserveReplacedDocument(paths.sourcePath, existingData, configVersionSourceRollback)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "BACKUP_ERROR", fmt.Sprintf("Failed to back up existing config: %v", err))
		return
	}

	origin := managementOrigin(r, configsnapshot.SourceRollback)
	origin.RollbackOf = target.version
	afterAttempt := s.configActivationAttempt()
	if !s.writeRouterConfigFiles(w, paths, existingData, target.document, origin) {
		return
	}

	logging.Infof(
		"Config rolled back to version %s via API: sourceConfigPath=%s, runtimeConfigPath=%s",
		target.id,
		paths.sourcePath,
		paths.runtimePath,
	)

	s.writeRollbackSuccess(w, target, paths.runtimePath, backupDir, afterAttempt)
}

func backupFilePath(backupDir, version string) string {
	return filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
}

func (s *ClassificationAPIServer) loadRollbackBackup(
	w http.ResponseWriter,
	backupFile string,
	version string,
) ([]byte, *config.RouterConfig, bool) {
	backupData, err := os.ReadFile(backupFile)
	if err != nil {
		s.writeErrorResponse(
			w,
			http.StatusNotFound,
			"VERSION_NOT_FOUND",
			fmt.Sprintf("Backup version %s not found", version),
		)
		return nil, nil, false
	}
	backupCfg, err := config.ParseYAMLBytes(backupData)
	if err != nil {
		s.writeErrorResponse(
			w,
			http.StatusBadRequest,
			"BACKUP_INVALID",
			fmt.Sprintf("Backup config is invalid: %v", err),
		)
		return nil, nil, false
	}
	return backupData, backupCfg, true
}

// loadCompatibleRollbackSource reads the persisted document a rollback
// replaces and checks that the restored one can replace the serving
// configuration without a restart. The persisted document may be a rejected
// change that never served, so it stands in for the serving configuration only
// in an API without the Router's runtime.
func (s *ClassificationAPIServer) loadCompatibleRollbackSource(
	w http.ResponseWriter,
	sourcePath string,
	backupCfg *config.RouterConfig,
) ([]byte, bool) {
	existingData, err := readPersistedSourceConfig(sourcePath)
	if err != nil && !os.IsNotExist(err) {
		s.writeErrorResponse(
			w,
			http.StatusInternalServerError,
			"READ_ERROR",
			fmt.Sprintf("Failed to read current config: %v", err),
		)
		return nil, false
	}
	currentCfg := s.servingConfig()
	if currentCfg == nil {
		if len(existingData) == 0 {
			return existingData, true
		}
		if currentCfg, err = config.ParseYAMLBytes(existingData); err != nil {
			s.writeErrorResponse(
				w,
				http.StatusInternalServerError,
				"CURRENT_CONFIG_INVALID",
				fmt.Sprintf("Current config is invalid: %v", err),
			)
			return nil, false
		}
	}
	if err := validateParsedHotReloadCompatibilityInMode(s.gatewayMode, currentCfg, backupCfg); err != nil {
		s.writeErrorResponse(
			w,
			http.StatusConflict,
			"RESTART_REQUIRED",
			err.Error(),
		)
		return nil, false
	}
	return existingData, true
}

func (s *ClassificationAPIServer) writeRollbackSuccess(
	w http.ResponseWriter,
	target rollbackTarget,
	runtimePath string,
	backupDir string,
	afterAttempt uint64,
) {
	version, backupData := target.id, target.document
	etag := configDocumentETag(backupData)
	runtimeHash, runtimeStatus := s.waitForRuntimeConfigActivation(runtimePath, backupData, afterAttempt)
	statusCode := http.StatusOK
	status := "success"
	message := fmt.Sprintf("Rolled back to version %s. Router reload is active.", version)
	if runtimeStatus == "pending" {
		statusCode = http.StatusAccepted
		status = "accepted"
		message = fmt.Sprintf("Rolled back to version %s on disk; runtime activation is pending. Poll /api/v1/config/hash until activation_status is active.", version)
	} else if runtimeStatus == "persisted" {
		statusCode = http.StatusAccepted
		status = "accepted"
		message = fmt.Sprintf("Rolled back to version %s in the Kubernetes ConfigMap; it takes effect on the router's next restart.", version)
	} else if runtimeStatus == "failed" {
		statusCode = http.StatusServiceUnavailable
		status = "activation_failed"
		message = "The rollback is persisted, but runtime activation failed. Inspect activation and correct or roll back the persisted configuration."
	} else if runtimeStatus == "unknown" {
		message = "The rollback is persisted; runtime activation could not be observed."
	}
	w.Header().Set("ETag", etag)
	cleanupConfigBackups(backupDir)
	s.writeJSONResponse(w, statusCode, RouterConfigUpdateResponse{
		Status:               status,
		Version:              version,
		ETag:                 etag,
		ActivationStatus:     runtimeStatus,
		GeneratedRuntimeHash: runtimeHash,
		ConfigVersion:        s.activatedConfigVersion(runtimeStatus, runtimeHash),
		RollbackOf:           target.version,
		Message:              message,
		Activation:           s.configActivationAfter(runtimeHash, afterAttempt),
	})
}

// cleanupConfigBackups trims the backups this API keeps on its own; a
// configuration history trims itself, and then there is no directory to trim.
func cleanupConfigBackups(backupDir string) {
	if backupDir != "" {
		configCleanupBackups(backupDir)
	}
}

// activatedConfigVersion is the snapshot version of an active mutation.
func (s *ClassificationAPIServer) activatedConfigVersion(runtimeStatus, runtimeHash string) uint64 {
	if runtimeStatus != "active" {
		return 0
	}
	return s.activeConfigVersion(runtimeHash)
}

// parseRollbackVersion reads which recorded version a rollback restores: a
// backup timestamp, or a configuration version as config_version or as a
// number in version.
func (s *ClassificationAPIServer) parseRollbackVersion(w http.ResponseWriter, r *http.Request) (routerConfigRollbackRequest, bool) {
	var req routerConfigRollbackRequest
	if err := s.parseStrictJSONRequest(r, &req); err != nil {
		s.writeJSONRequestError(w, err)
		return req, false
	}
	switch {
	case req.Version == "" && req.ConfigVersion == 0:
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "version is required")
		return req, false
	case req.Version != "" && req.ConfigVersion != 0:
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT", "give either version or config_version, not both")
		return req, false
	case snapshotVersionPattern.MatchString(req.Version):
		req.ConfigVersion, _ = strconv.ParseUint(req.Version, 10, 64)
		req.Version = ""
		return req, true
	}
	// The value is interpolated into a backup path, so the strict allowlist also
	// prevents path traversal outside the backup directory.
	if req.ConfigVersion == 0 && !configVersionPattern.MatchString(req.Version) {
		s.writeErrorResponse(w, http.StatusBadRequest, "INVALID_INPUT",
			"version must be a configuration version number, or match YYYYMMDD-HHMMSS with an optional numeric sequence suffix")
		return req, false
	}
	return req, true
}

// handleConfigVersions handles GET /api/v1/config/versions.
func (s *ClassificationAPIServer) handleConfigVersions(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeJSONResponse(w, http.StatusOK, []RouterConfigVersionEntry{})
		return
	}
	if lifecycle := s.configLifecycle(); lifecycle != nil {
		s.writeJSONResponse(w, http.StatusOK, s.historyVersionEntries(lifecycle.History()))
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	backupDir := configBackupDir(paths.sourcePath)

	versions := []RouterConfigVersionEntry{}

	entries, err := os.ReadDir(backupDir)
	if err != nil {
		if os.IsNotExist(err) {
			s.writeJSONResponse(w, http.StatusOK, versions)
			return
		}
		s.writeErrorResponse(w, http.StatusInternalServerError, "BACKUP_READ_ERROR", fmt.Sprintf("Failed to read config backups: %v", err))
		return
	}

	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "config.") || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}
		name := entry.Name()
		versionStr := strings.TrimPrefix(name, "config.")
		versionStr = strings.TrimSuffix(versionStr, ".yaml")

		timestampVersion := versionStr
		if len(timestampVersion) > len(configVersionTimestampLayout) {
			timestampVersion = timestampVersion[:len(configVersionTimestampLayout)]
		}
		t, err := time.Parse(configVersionTimestampLayout, timestampVersion)
		timestamp := versionStr
		if err == nil {
			timestamp = t.Format("2006-01-02 15:04:05")
		}

		versions = append(versions, RouterConfigVersionEntry{
			Version:   versionStr,
			Timestamp: timestamp,
			Source:    readConfigVersionSource(backupDir, versionStr),
			Filename:  name,
		})
	}

	sort.Slice(versions, func(i, j int) bool {
		return versions[i].Version > versions[j].Version
	})

	s.writeJSONResponse(w, http.StatusOK, versions)
}

// nextConfigVersion preserves human-sortable timestamps while guaranteeing
// immutable backup filenames for bursts of updates serialized by deployMu.
func nextConfigVersion(backupDir string, now time.Time) string {
	base := now.Format(configVersionTimestampLayout)
	for sequence := 0; ; sequence++ {
		version := base
		if sequence > 0 {
			version = fmt.Sprintf("%s-%03d", base, sequence)
		}
		_, err := os.Stat(filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version)))
		if os.IsNotExist(err) {
			return version
		}
		if err != nil {
			// The eventual backup write will report the concrete filesystem error.
			return version
		}
	}
}

// handleConfigGet handles GET /api/v1/config and returns the current router config as JSON.
// Access requires config.read; plaintext secrets require secret_view (otherwise redacted).
func (s *ClassificationAPIServer) handleConfigGet(w http.ResponseWriter, r *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	data, err := readPersistedSourceConfig(paths.sourcePath)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", fmt.Sprintf("Failed to read config: %v", err))
		return
	}
	w.Header().Set("ETag", configDocumentETag(data))
	s.setActiveConfigHeaders(w)

	var cfgMap interface{}
	if err := yaml.Unmarshal(data, &cfgMap); err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "PARSE_ERROR", fmt.Sprintf("Failed to parse config: %v", err))
		return
	}

	s.writeJSONResponse(w, http.StatusOK, s.maybeRedactConfigView(r, cfgMap))
}

func configCleanupBackups(backupDir string) {
	entries, err := os.ReadDir(backupDir)
	if err != nil {
		return
	}

	var backups []os.DirEntry
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "config.") && strings.HasSuffix(entry.Name(), ".yaml") {
			backups = append(backups, entry)
		}
	}

	if len(backups) <= maxBackups {
		return
	}

	sort.Slice(backups, func(i, j int) bool {
		return backups[i].Name() < backups[j].Name()
	})

	toRemove := len(backups) - maxBackups
	for i := 0; i < toRemove; i++ {
		path := filepath.Join(backupDir, backups[i].Name())
		if err := os.Remove(path); err != nil {
			logging.Warnf("Failed to remove old backup %s: %v", path, err)
		} else {
			version := strings.TrimSuffix(strings.TrimPrefix(backups[i].Name(), "config."), ".yaml")
			_ = os.Remove(configVersionSourcePath(backupDir, version))
			logging.Infof("Removed old backup: %s", backups[i].Name())
		}
	}
}

func configVersionSourcePath(backupDir, version string) string {
	return filepath.Join(backupDir, fmt.Sprintf("config.%s.source", version))
}

func writeConfigVersionSource(backupDir, version string, source configVersionSource) error {
	return writePrivateConfigArtifact(configVersionSourcePath(backupDir, version), []byte(source+"\n"))
}

func recordConfigBackup(backupDir, version string, data []byte, source configVersionSource) error {
	if len(data) == 0 {
		return nil
	}
	if err := ensurePrivateConfigBackupDir(backupDir); err != nil {
		return fmt.Errorf("prepare private config backup directory: %w", err)
	}
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := writePrivateConfigArtifact(backupFile, data); err != nil {
		return fmt.Errorf("create config backup: %w", err)
	}
	if err := writeConfigVersionSource(backupDir, version, source); err != nil {
		_ = os.Remove(backupFile)
		return fmt.Errorf("write config backup source metadata: %w", err)
	}
	logging.Infof("Config backup created: %s", backupFile)
	return nil
}

func ensurePrivateConfigBackupDir(backupDir string) error {
	if err := os.MkdirAll(backupDir, 0o700); err != nil {
		return err
	}
	return os.Chmod(backupDir, 0o700)
}

func writePrivateConfigArtifact(path string, data []byte) error {
	file, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_EXCL, 0o600)
	if err != nil {
		return err
	}
	if _, writeErr := file.Write(data); writeErr != nil {
		_ = file.Close()
		_ = os.Remove(path)
		return writeErr
	}
	if syncErr := file.Sync(); syncErr != nil {
		_ = file.Close()
		_ = os.Remove(path)
		return syncErr
	}
	if closeErr := file.Close(); closeErr != nil {
		_ = os.Remove(path)
		return closeErr
	}
	return nil
}

func readConfigVersionSource(backupDir, version string) configVersionSource {
	data, err := os.ReadFile(configVersionSourcePath(backupDir, version))
	if err != nil {
		return configVersionSourceUnknown
	}
	if source, ok := knownConfigVersionSources[configVersionSource(strings.TrimSpace(string(data)))]; ok {
		return source
	}
	return configVersionSourceUnknown
}

// handleConfigHash reports persisted, generated-runtime, and active config
// hashes separately. A source write is not proof that the in-memory router has
// completed model preparation and its atomic swap.
func (s *ClassificationAPIServer) handleConfigHash(w http.ResponseWriter, _ *http.Request) {
	if s.configPath == "" {
		s.writeErrorResponse(w, http.StatusInternalServerError, "NO_CONFIG_PATH", "Router configPath not set")
		return
	}

	paths := resolveConfigPersistencePaths(s.configPath)
	sourceHash, runtimeHash, kubernetesTarget, err := s.resolveConfigSourceAndRuntimeHash(paths)
	if err != nil {
		s.writeErrorResponse(w, http.StatusInternalServerError, "READ_ERROR", err.Error())
		return
	}

	activeHash := s.activeConfigDocumentHash()
	status := s.configActivationStatus(runtimeHash, activeHash)
	if kubernetesTarget && status != "active" {
		// A mismatch here can never resolve by polling: the ConfigMap write
		// never touches the running process, so only a restart picks it up.
		status = "persisted"
	}
	s.writeJSONResponse(w, http.StatusOK, configHashResponse{
		SourceConfigHash:     sourceHash,
		GeneratedRuntimeHash: runtimeHash,
		ActiveRuntimeHash:    activeHash,
		ActiveVersion:        s.activeConfigVersion(activeHash),
		ActivationStatus:     status,
		Activation:           s.configActivation(runtimeHash),
		LastRejection:        s.lastConfigRejection(),
	})
}

// resolveConfigSourceAndRuntimeHash reads the current source and runtime
// document hashes. On a Kubernetes ConfigMap target, both come from the
// ConfigMap itself (the single document that backs the mounted file, per
// issue #3688): the mounted file is read-only and never reflects an API
// write, so reading it here would report stale hashes indefinitely.
func (s *ClassificationAPIServer) resolveConfigSourceAndRuntimeHash(paths configPersistencePaths) (sourceHash, runtimeHash string, kubernetesTarget bool, err error) {
	if target, ok := configwriter.ConfigMapTargetFromEnv(); ok {
		writer, writerErr := resolvedConfigMapWriter()
		if writerErr != nil {
			return "", "", true, fmt.Errorf("config write target is declared but no Kubernetes client is available: %w", writerErr)
		}
		ctx, cancel := context.WithTimeout(context.Background(), configMapWriteTimeout)
		defer cancel()
		data, found, readErr := writer.Read(ctx, target)
		if readErr != nil {
			return "", "", true, fmt.Errorf("failed to read config ConfigMap: %w", readErr)
		}
		if !found {
			return "", "", true, nil
		}
		hash := configDocumentETagHash(data)
		return hash, hash, true, nil
	}

	data, err := os.ReadFile(paths.sourcePath)
	if err != nil {
		return "", "", false, fmt.Errorf("failed to read config: %w", err)
	}
	runtimeHash, err = configFileHash(paths.runtimePath)
	if err != nil {
		return "", "", false, fmt.Errorf("failed to read runtime config: %w", err)
	}
	return configDocumentETagHash(data), runtimeHash, false, nil
}
