package handlers

import (
	"log"
	"os"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/dashboard/backend/configprojection"
	routerconfig "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// ConfigProjectionDrift exposes active projection health for status surfaces.
type ConfigProjectionDrift struct {
	Status        string    `json:"status"`
	ActiveVersion string    `json:"active_version,omitempty"`
	LastError     string    `json:"last_error,omitempty"`
	UpdatedAt     time.Time `json:"updated_at,omitempty"`
}

func currentConfigProjectionDrift() *ConfigProjectionDrift {
	if configProjectionStore == nil {
		return nil
	}

	projection, err := configProjectionStore.GetActiveProjection()
	if err != nil {
		return &ConfigProjectionDrift{
			Status:    configprojection.StatusFailed,
			LastError: err.Error(),
		}
	}

	drift := &ConfigProjectionDrift{
		Status:        projection.Status,
		ActiveVersion: projection.ActiveVersion,
		LastError:     projection.LastError,
		UpdatedAt:     projection.UpdatedAt,
	}
	return drift
}

func readCanonicalConfigPreferProjection(configPath string) (*routerconfig.CanonicalConfig, string, error) {
	if yamlBytes, source, ok := activeProjectionYAML(); ok {
		cfg, err := decodeYAMLTaggedBytes[routerconfig.CanonicalConfig](yamlBytes)
		if err != nil {
			log.Printf("Warning: active config projection decode failed, falling back to file: %v", err)
		} else {
			return &cfg, source, nil
		}
	}

	configData, err := readCanonicalConfigFile(configPath)
	if err != nil {
		return nil, "", err
	}
	return configData, "file", nil
}

func readConfigYAMLPreferProjection(configPath string) ([]byte, string, error) {
	if yamlBytes, source, ok := activeProjectionYAML(); ok {
		return yamlBytes, source, nil
	}

	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, "", err
	}
	return data, "file", nil
}

func activeProjectionYAML() ([]byte, string, bool) {
	if configProjectionStore == nil {
		return nil, "", false
	}

	projection, err := configProjectionStore.GetActiveProjection()
	if err != nil {
		log.Printf("Warning: failed to read active config projection: %v", err)
		return nil, "", false
	}
	if projection.Status != configprojection.StatusOK {
		log.Printf(
			"Warning: active config projection status=%s, falling back to canonical file",
			projection.Status,
		)
		return nil, "", false
	}
	if projection.Deployment == nil || len(strings.TrimSpace(projection.Deployment.YAMLSnapshot)) == 0 {
		log.Printf("Warning: active config projection has no YAML snapshot, falling back to canonical file")
		return nil, "", false
	}

	return []byte(projection.Deployment.YAMLSnapshot), "projection", true
}

func listConfigVersionsPreferProjection(configPath string) ([]ConfigVersion, string, error) {
	if configProjectionStore == nil {
		versions, err := listConfigVersions(configPath)
		return versions, "file", err
	}

	deployments, err := configProjectionStore.ListDeployments()
	if err != nil {
		log.Printf("Warning: failed to list config projection deployments, falling back to backup dir: %v", err)
		versions, listErr := listConfigVersions(configPath)
		return versions, "file", listErr
	}
	if len(deployments) == 0 {
		versions, listErr := listConfigVersions(configPath)
		return versions, "file", listErr
	}

	versions := make([]ConfigVersion, 0, len(deployments))
	seen := make(map[string]struct{}, len(deployments))
	for _, deployment := range deployments {
		versions = append(versions, deploymentSummaryToConfigVersion(deployment))
		seen[deployment.Version] = struct{}{}
	}

	localVersions, localErr := listConfigVersions(configPath)
	if localErr == nil {
		for _, local := range localVersions {
			if _, ok := seen[local.Version]; ok {
				continue
			}
			versions = append(versions, local)
		}
	}

	return versions, "projection", nil
}

func readRollbackYAML(configDir, version string) ([]byte, error) {
	backupData, err := readConfigBackup(configDir, version)
	if err == nil {
		return backupData, nil
	}

	if configProjectionStore == nil {
		return nil, err
	}

	deployment, getErr := configProjectionStore.GetDeployment(version)
	if getErr != nil {
		return nil, err
	}
	if len(strings.TrimSpace(deployment.YAMLSnapshot)) == 0 {
		return nil, err
	}

	return []byte(deployment.YAMLSnapshot), nil
}

func deploymentSummaryToConfigVersion(deployment configprojection.DeploymentSummary) ConfigVersion {
	return ConfigVersion{
		Version:   deployment.Version,
		Timestamp: deployment.CreatedAt.Format("2006-01-02 15:04:05"),
		Source:    deployment.Source,
		Filename:  "config." + deployment.Version + ".yaml",
	}
}
