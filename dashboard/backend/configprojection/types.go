package configprojection

import (
	"encoding/json"
	"time"
)

// Source labels persisted deployment provenance.
const (
	SourceDSL      = "dsl"
	SourceManual   = "manual"
	SourceRollback = "rollback"
)

// Status values for the active projection row.
const (
	StatusOK     = "ok"
	StatusStale  = "stale"
	StatusFailed = "failed"
)

// RefreshInput carries canonical config bytes used to rebuild projection state.
type RefreshInput struct {
	Version     string
	Source      string
	YAMLBytes   []byte
	DSLSnapshot string
}

// ValidationSummary records parse/validate outcome for a deployment snapshot.
type ValidationSummary struct {
	Status      string   `json:"status"`
	Diagnostics []string `json:"diagnostics,omitempty"`
}

// DeploymentSummary is the list-view record for deployment history APIs.
type DeploymentSummary struct {
	Version   string    `json:"version"`
	Source    string    `json:"source"`
	CreatedAt time.Time `json:"created_at"`
	YAMLHash  string    `json:"yaml_hash"`
}

// DeploymentRecord is the full persisted deployment projection.
type DeploymentRecord struct {
	Version     string            `json:"version"`
	Source      string            `json:"source"`
	CreatedAt   time.Time         `json:"created_at"`
	DSLSnapshot string            `json:"dsl_snapshot,omitempty"`
	YAMLHash    string            `json:"yaml_hash"`
	Validation  ValidationSummary `json:"validation"`
	Models      json.RawMessage   `json:"models"`
	Signals     json.RawMessage   `json:"signals"`
	Decisions   json.RawMessage   `json:"decisions"`
	Plugins     json.RawMessage   `json:"plugins"`
	Projections json.RawMessage   `json:"projections"`
}

// ActiveProjectionStatus exposes drift metadata for the active projection.
type ActiveProjectionStatus struct {
	ActiveVersion string    `json:"active_version"`
	Status        string    `json:"status"`
	LastError     string    `json:"last_error,omitempty"`
	UpdatedAt     time.Time `json:"updated_at"`
}

// ActiveProjectionResponse is returned by the active-projection API.
type ActiveProjectionResponse struct {
	ActiveProjectionStatus
	Deployment *DeploymentRecord `json:"deployment,omitempty"`
}
