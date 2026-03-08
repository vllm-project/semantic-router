package configlifecycle

import (
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"gopkg.in/yaml.v3"
)

// MaxBackups is the maximum number of config backups to keep.
const MaxBackups = 10

var deployMu sync.Mutex

type DeployRequest struct {
	YAML string
	DSL  string
}

type DeployPreview struct {
	Current string
	Preview string
}

type DeployResult struct {
	Version string
	Message string
}

type ConfigVersion struct {
	Version   string
	Timestamp string
	Source    string
	Filename  string
}

func (s *Service) DeployPreview(req DeployRequest) (DeployPreview, error) {
	yamlBytes, err := parseDeployYAML(req.YAML)
	if err != nil {
		return DeployPreview{}, err
	}

	currentData, err := s.currentConfigYAML()
	if err != nil {
		if !os.IsNotExist(err) {
			return DeployPreview{}, err
		}
		currentData = []byte("# No existing config\n")
	}

	previewBytes := yamlBytes
	if len(currentData) > 0 {
		previewBytes = mergeYAMLWithExisting(currentData, yamlBytes)
	}

	return DeployPreview{
		Current: CanonicalizeYAMLForDiff(currentData),
		Preview: CanonicalizeYAMLForDiff(previewBytes),
	}, nil
}

func (s *Service) Deploy(req DeployRequest) (DeployResult, error) {
	if s.hasRevisionStore() {
		return s.deployWithRevisionWorkflow(req)
	}
	return s.deployDirect(req)
}

func (s *Service) deployWithRevisionWorkflow(req DeployRequest) (DeployResult, error) {
	yamlBytes, err := parseDeployYAML(req.YAML)
	if err != nil {
		return DeployResult{}, err
	}

	existingData, readErr := s.currentConfigYAML()
	if readErr != nil && !os.IsNotExist(readErr) {
		return DeployResult{}, readErr
	}
	if len(existingData) > 0 {
		yamlBytes = mergeYAMLWithExisting(existingData, yamlBytes)
	}

	result, err := s.runCompatibilityRevisionWorkflow(yamlBytes, compatibilityWorkflowOptions{
		source:         "compat_config_deploy",
		summary:        "Applied config deploy via compatibility API",
		triggerSource:  "deploy_api",
		auditAction:    "config.deploy",
		successMessage: "Config deployed successfully. Router and Envoy have been updated.",
		metadata: map[string]interface{}{
			"compat_operation": "deploy",
			"dsl_present":      strings.TrimSpace(req.DSL) != "",
		},
	})
	if err != nil {
		return DeployResult{}, err
	}
	if err := s.archiveDSL(req.DSL); err != nil {
		log.Printf("Warning: failed to archive DSL source: %v", err)
	}
	CleanupBackups(s.backupDir())
	return DeployResult{
		Version: result.Version,
		Message: result.Message,
	}, nil
}

func (s *Service) deployDirect(req DeployRequest) (DeployResult, error) {
	if !deployMu.TryLock() {
		return DeployResult{}, &Error{
			StatusCode: 409,
			Code:       "deploy_in_progress",
			Message:    "Another deploy operation is in progress. Please try again.",
		}
	}
	defer deployMu.Unlock()

	yamlBytes, err := parseDeployYAML(req.YAML)
	if err != nil {
		return DeployResult{}, err
	}

	existingData, readErr := os.ReadFile(s.ConfigPath)
	if readErr == nil && len(existingData) > 0 {
		yamlBytes = mergeYAMLWithExisting(existingData, yamlBytes)
	}

	version := time.Now().Format("20060102-150405")
	if err := s.backupConfigData(existingData, version); err != nil {
		log.Printf("Warning: failed to create backup: %v", err)
	}
	if err := s.archiveDSL(req.DSL); err != nil {
		log.Printf("Warning: failed to archive DSL source: %v", err)
	}
	if err := writeConfigAtomically(s.ConfigPath, yamlBytes); err != nil {
		return DeployResult{}, fmt.Errorf("failed to write config: %w", err)
	}

	log.Printf("[Deploy] Config written to %s: version=%s, size=%d bytes", s.ConfigPath, version, len(yamlBytes))
	if err := s.propagateConfigToRuntime(); err != nil {
		if restoreErr := s.restorePreviousRuntimeConfig(existingData); restoreErr != nil {
			return DeployResult{}, fmt.Errorf("failed to apply deployed config to runtime: %w; failed to restore previous config: %w", err, restoreErr)
		}
		return DeployResult{}, fmt.Errorf("failed to apply deployed config to runtime: %w; previous config restored", err)
	}

	CleanupBackups(s.backupDir())
	s.recordSuccessfulCompatibilityChange(yamlBytes, revisionPersistenceOptions{
		source:         "compat_config_deploy",
		summary:        "Applied config deploy via compatibility API",
		action:         "config.deploy",
		triggerSource:  "deploy_api",
		revisionStatus: "active",
		previousStatus: "superseded",
		deployStatus:   "succeeded",
		message:        "Config deployed successfully. Router and Envoy have been updated.",
		metadata: map[string]interface{}{
			"operation":      "deploy",
			"deploy_version": version,
			"dsl_present":    strings.TrimSpace(req.DSL) != "",
		},
	})
	return DeployResult{
		Version: version,
		Message: "Config deployed successfully. Router and Envoy have been updated.",
	}, nil
}

func (s *Service) ListVersions() ([]ConfigVersion, error) {
	entries, err := os.ReadDir(s.backupDir())
	if err != nil {
		return []ConfigVersion{}, nil
	}

	versions := []ConfigVersion{}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasPrefix(entry.Name(), "config.") || !strings.HasSuffix(entry.Name(), ".yaml") {
			continue
		}
		versionStr := strings.TrimSuffix(strings.TrimPrefix(entry.Name(), "config."), ".yaml")
		timestamp := versionStr
		if parsedTime, err := time.Parse("20060102-150405", versionStr); err == nil {
			timestamp = parsedTime.Format("2006-01-02 15:04:05")
		}
		versions = append(versions, ConfigVersion{
			Version:   versionStr,
			Timestamp: timestamp,
			Source:    "dsl",
			Filename:  entry.Name(),
		})
	}

	sort.Slice(versions, func(i, j int) bool {
		return versions[i].Version > versions[j].Version
	})
	return versions, nil
}

func parseDeployYAML(rawYAML string) ([]byte, error) {
	yamlBytes := []byte(rawYAML)
	var yamlMap interface{}
	if err := yaml.Unmarshal(yamlBytes, &yamlMap); err != nil {
		return nil, &Error{
			StatusCode: 400,
			Code:       "yaml_parse_error",
			Message:    fmt.Sprintf("Invalid YAML syntax: %v", err),
		}
	}
	return yamlBytes, nil
}

func mergeYAMLWithExisting(existingData, yamlBytes []byte) []byte {
	existingMap := make(map[string]interface{})
	if err := yaml.Unmarshal(existingData, &existingMap); err != nil {
		return yamlBytes
	}
	newMap := make(map[string]interface{})
	if err := yaml.Unmarshal(yamlBytes, &newMap); err != nil {
		return yamlBytes
	}
	merged := DeepMerge(existingMap, newMap)
	mergedYAML, err := yaml.Marshal(merged)
	if err != nil {
		return yamlBytes
	}
	return mergedYAML
}

func (s *Service) backupDir() string {
	return filepath.Join(s.ConfigDir, ".vllm-sr", "config-backups")
}

func (s *Service) backupConfigData(existingData []byte, version string) error {
	if len(existingData) == 0 {
		return nil
	}
	backupDir := s.backupDir()
	if err := os.MkdirAll(backupDir, 0o755); err != nil {
		return err
	}
	backupFile := filepath.Join(backupDir, fmt.Sprintf("config.%s.yaml", version))
	if err := os.WriteFile(backupFile, existingData, 0o644); err != nil {
		return err
	}
	log.Printf("[Deploy] Config backup created: %s", backupFile)
	return nil
}

func (s *Service) archiveDSL(dsl string) error {
	if strings.TrimSpace(dsl) == "" {
		return nil
	}
	dslDir := filepath.Join(s.ConfigDir, ".vllm-sr")
	if err := os.MkdirAll(dslDir, 0o755); err != nil {
		return err
	}
	dslFile := filepath.Join(dslDir, "config.dsl")
	return os.WriteFile(dslFile, []byte(dsl), 0o644)
}

// CleanupBackups removes old backups beyond MaxBackups.
func CleanupBackups(backupDir string) {
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
	if len(backups) <= MaxBackups {
		return
	}

	sort.Slice(backups, func(i, j int) bool {
		return backups[i].Name() < backups[j].Name()
	})
	for i := 0; i < len(backups)-MaxBackups; i++ {
		path := filepath.Join(backupDir, backups[i].Name())
		if err := os.Remove(path); err != nil {
			log.Printf("Warning: failed to remove old backup %s: %v", path, err)
		} else {
			log.Printf("Removed old backup: %s", backups[i].Name())
		}
	}
}
