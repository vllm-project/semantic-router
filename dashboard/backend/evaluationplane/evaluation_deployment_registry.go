package evaluationplane

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	evaluationDeploymentRegistryVersion = "evaluation-deployments.v1"
	evaluationDeploymentRegistryFile    = "registry.json"
	maxEvaluationDeploymentRegistrySize = 1 << 20
	maxEvaluationDeploymentIDLength     = 48
	maxEvaluationDeploymentNameLength   = 96
	maxEvaluationDeploymentDescription  = 512
)

var (
	deploymentIDPattern   = regexp.MustCompile(`^[a-z0-9][a-z0-9._-]{0,47}$`)
	deploymentNamePattern = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9 ._-]{0,95}$`)
)

// DeploymentTargetSnapshot is the private server-owned identity of one
// deployment-scoped Mixture target. Only its label and logical Mixture are
// projected into the authenticated catalog; origins and config paths never are.
type DeploymentTargetSnapshot struct {
	TargetID       string
	DeploymentID   string
	DeploymentName string
	Description    string
	RouterAPIURL   string
	EnvoyURL       string
	ConfigDigest   string
	Mixture        MixtureTargetSnapshot
}

type evaluationDeploymentRegistry struct {
	SchemaVersion string                           `json:"schema_version"`
	Deployments   []evaluationDeploymentDefinition `json:"deployments"`
}

type evaluationDeploymentDefinition struct {
	ID           string `json:"id"`
	Name         string `json:"name"`
	Description  string `json:"description,omitempty"`
	ConfigFile   string `json:"config_file"`
	RouterOrigin string `json:"router_origin"`
	EnvoyOrigin  string `json:"envoy_origin"`
}

// LoadEvaluationDeploymentRegistry freezes every configured deployment from a
// read-only registry directory. The registry is deliberately data-only: it can
// select config bytes and canonical origins, but it cannot declare credentials,
// ledger endpoints, worker commands, or executable adapters.
func LoadEvaluationDeploymentRegistry(root, runtimeRevision string) ([]DeploymentTargetSnapshot, error) {
	root = strings.TrimSpace(root)
	if root == "" {
		return nil, nil
	}
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return nil, fmt.Errorf("resolve evaluation deployments directory: %w", err)
	}
	registryRoot, err := openDeploymentRegistryRoot(absRoot)
	if err != nil {
		return nil, fmt.Errorf("validate evaluation deployments directory: %w", err)
	}
	defer registryRoot.Close()
	registryBytes, err := registryRoot.ReadFile(evaluationDeploymentRegistryFile, maxEvaluationDeploymentRegistrySize)
	if err != nil {
		return nil, fmt.Errorf("read evaluation deployment registry: %w", err)
	}
	registry, err := decodeEvaluationDeploymentRegistry(registryBytes)
	if err != nil {
		return nil, err
	}

	seenDeployments := make(map[string]struct{}, len(registry.Deployments))
	seenTargets := make(map[string]struct{})
	result := make([]DeploymentTargetSnapshot, 0, len(registry.Deployments))
	for index, deployment := range registry.Deployments {
		if err := validateEvaluationDeploymentDefinition(deployment); err != nil {
			return nil, fmt.Errorf("evaluation deployment %d: %w", index, err)
		}
		if _, duplicate := seenDeployments[deployment.ID]; duplicate {
			return nil, fmt.Errorf("duplicate evaluation deployment id %q", deployment.ID)
		}
		seenDeployments[deployment.ID] = struct{}{}
		configBytes, readErr := registryRoot.ReadFile(deployment.ConfigFile, maxStructuredArtifactBytes)
		if readErr != nil {
			return nil, fmt.Errorf("read evaluation deployment %q config: %w", deployment.ID, readErr)
		}
		snapshot, snapshotErr := ModelArmSnapshotFromYAML(configBytes, runtimeRevision)
		if snapshotErr != nil {
			return nil, fmt.Errorf("load evaluation deployment %q config: %w", deployment.ID, snapshotErr)
		}
		if len(snapshot.Mixtures) == 0 {
			return nil, fmt.Errorf("evaluation deployment %q config exposes no Mixture-of-Models target", deployment.ID)
		}
		for _, mixture := range snapshot.Mixtures {
			targetID := deploymentTargetID(deployment.ID, mixture.Mixture.ID)
			if !portableIDPattern.MatchString(targetID) {
				return nil, fmt.Errorf("evaluation deployment %q produced a non-portable target id", deployment.ID)
			}
			if _, duplicate := seenTargets[targetID]; duplicate {
				return nil, fmt.Errorf("duplicate evaluation deployment target id %q", targetID)
			}
			seenTargets[targetID] = struct{}{}
			result = append(result, DeploymentTargetSnapshot{
				TargetID: targetID, DeploymentID: deployment.ID,
				DeploymentName: deployment.Name, Description: deployment.Description,
				RouterAPIURL: deployment.RouterOrigin, EnvoyURL: deployment.EnvoyOrigin,
				ConfigDigest: snapshot.ConfigDigest, Mixture: mixture,
			})
		}
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("evaluation deployment registry produced no targets")
	}
	return result, nil
}

func decodeEvaluationDeploymentRegistry(data []byte) (evaluationDeploymentRegistry, error) {
	if err := rejectDuplicateJSONKeys(data); err != nil {
		return evaluationDeploymentRegistry{}, fmt.Errorf("decode evaluation deployment registry: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var registry evaluationDeploymentRegistry
	if err := decoder.Decode(&registry); err != nil {
		return evaluationDeploymentRegistry{}, fmt.Errorf("decode evaluation deployment registry: %w", err)
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return evaluationDeploymentRegistry{}, fmt.Errorf("decode evaluation deployment registry: trailing JSON data")
	}
	if registry.SchemaVersion != evaluationDeploymentRegistryVersion {
		return evaluationDeploymentRegistry{}, fmt.Errorf(
			"evaluation deployment registry schema_version must be %q",
			evaluationDeploymentRegistryVersion,
		)
	}
	if len(registry.Deployments) == 0 {
		return evaluationDeploymentRegistry{}, fmt.Errorf("evaluation deployment registry requires at least one deployment")
	}
	return registry, nil
}

func validateEvaluationDeploymentDefinition(deployment evaluationDeploymentDefinition) error {
	if !deploymentIDPattern.MatchString(deployment.ID) || len(deployment.ID) > maxEvaluationDeploymentIDLength {
		return fmt.Errorf("id must be a lowercase portable identifier of at most %d characters", maxEvaluationDeploymentIDLength)
	}
	if deployment.Name != strings.TrimSpace(deployment.Name) ||
		!deploymentNamePattern.MatchString(deployment.Name) ||
		len(deployment.Name) > maxEvaluationDeploymentNameLength {
		return fmt.Errorf("name must be a portable 1-%d character display label", maxEvaluationDeploymentNameLength)
	}
	if deployment.Description != strings.TrimSpace(deployment.Description) ||
		len(deployment.Description) > maxEvaluationDeploymentDescription {
		return fmt.Errorf("description must be at most %d trimmed characters", maxEvaluationDeploymentDescription)
	}
	if err := validateRelativeDeploymentConfigPath(deployment.ConfigFile); err != nil {
		return fmt.Errorf("config_file: %w", err)
	}
	if deployment.RouterOrigin == "" {
		return fmt.Errorf("router_origin is required")
	}
	if err := validateServerOrigin(deployment.RouterOrigin); err != nil {
		return fmt.Errorf("router_origin: %w", err)
	}
	if deployment.EnvoyOrigin == "" {
		return fmt.Errorf("envoy_origin is required")
	}
	if err := validateServerOrigin(deployment.EnvoyOrigin); err != nil {
		return fmt.Errorf("envoy_origin: %w", err)
	}
	return nil
}

func validateRelativeDeploymentConfigPath(path string) error {
	if path == "" || path != strings.TrimSpace(path) || filepath.IsAbs(path) || strings.Contains(path, `\`) {
		return fmt.Errorf("must be a trimmed relative POSIX path")
	}
	if filepath.Clean(path) != path || path == "." || strings.HasPrefix(path, "../") {
		return fmt.Errorf("must stay inside the deployment registry directory")
	}
	for _, component := range strings.Split(path, "/") {
		if component == "" || component == "." || component == ".." {
			return fmt.Errorf("must not contain empty or traversal components")
		}
	}
	return nil
}

func deploymentTargetID(deploymentID, mixtureID string) string {
	return deploymentID + "--" + mixtureID
}
