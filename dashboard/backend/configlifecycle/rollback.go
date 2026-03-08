package configlifecycle

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"gopkg.in/yaml.v3"

	"github.com/vllm-project/semantic-router/dashboard/backend/console"
)

func (s *Service) Rollback(version string) (DeployResult, error) {
	if s.hasRevisionStore() {
		return s.rollbackWithRevisionWorkflow(version)
	}
	return s.rollbackDirect(version)
}

func (s *Service) rollbackWithRevisionWorkflow(version string) (DeployResult, error) {
	backupData, err := s.loadRollbackBackup(version)
	if err != nil {
		return DeployResult{}, err
	}

	result, err := s.runCompatibilityRevisionWorkflow(backupData, compatibilityWorkflowOptions{
		source:         "compat_config_rollback",
		summary:        "Rolled back config via compatibility API",
		triggerSource:  "rollback_api",
		auditAction:    "config.rollback",
		successMessage: fmt.Sprintf("Rolled back to version %s. Router and Envoy have been updated.", version),
		metadata: map[string]interface{}{
			"compat_operation": "rollback",
			"rollback_version": version,
		},
		activation: activationOptions{
			previousActiveStatus: console.ConfigRevisionStatusRolledBack,
			deployStatus:         console.DeployEventStatusRolledBack,
		},
	})
	if err != nil {
		return DeployResult{}, err
	}
	CleanupBackups(s.backupDir())
	return DeployResult{
		Version: version,
		Message: result.Message,
	}, nil
}

func (s *Service) rollbackDirect(version string) (DeployResult, error) {
	if !deployMu.TryLock() {
		return DeployResult{}, &Error{
			StatusCode: 409,
			Code:       "deploy_in_progress",
			Message:    "Another config operation is in progress.",
		}
	}
	defer deployMu.Unlock()

	backupData, err := s.loadRollbackBackup(version)
	if err != nil {
		return DeployResult{}, err
	}

	existingData, readErr := os.ReadFile(s.ConfigPath)
	if readErr == nil && len(existingData) > 0 {
		_ = s.backupConfigData(existingData, time.Now().Format("20060102-150405"))
	}
	if err := writeConfigAtomically(s.ConfigPath, backupData); err != nil {
		return DeployResult{}, fmt.Errorf("failed to write config: %w", err)
	}

	if err := s.propagateConfigToRuntime(); err != nil {
		if restoreErr := s.restorePreviousRuntimeConfig(existingData); restoreErr != nil {
			return DeployResult{}, fmt.Errorf("failed to apply rolled back config to runtime: %w; failed to restore previous config: %w", err, restoreErr)
		}
		return DeployResult{}, fmt.Errorf("failed to apply rolled back config to runtime: %w; previous config restored", err)
	}

	s.recordSuccessfulCompatibilityChange(backupData, revisionPersistenceOptions{
		source:         "compat_config_rollback",
		summary:        "Rolled back config via compatibility API",
		action:         "config.rollback",
		triggerSource:  "rollback_api",
		revisionStatus: "active",
		previousStatus: "rolled_back",
		deployStatus:   "rolled_back",
		message:        fmt.Sprintf("Rolled back to version %s. Router and Envoy have been updated.", version),
		metadata: map[string]interface{}{
			"operation":        "rollback",
			"rollback_version": version,
		},
	})
	return DeployResult{
		Version: version,
		Message: fmt.Sprintf("Rolled back to version %s. Router and Envoy have been updated.", version),
	}, nil
}

func (s *Service) loadRollbackBackup(version string) ([]byte, error) {
	backupFile := filepath.Join(s.backupDir(), fmt.Sprintf("config.%s.yaml", version))
	backupData, err := os.ReadFile(backupFile)
	if err != nil {
		return nil, &Error{
			StatusCode: 404,
			Code:       "version_not_found",
			Message:    fmt.Sprintf("Backup version %s not found", version),
		}
	}
	var yamlCheck interface{}
	if err := yaml.Unmarshal(backupData, &yamlCheck); err != nil {
		return nil, &Error{
			StatusCode: 400,
			Code:       "backup_invalid",
			Message:    fmt.Sprintf("Backup config has invalid YAML: %v", err),
		}
	}
	return backupData, nil
}
