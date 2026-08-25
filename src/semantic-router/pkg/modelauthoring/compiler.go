// Package modelauthoring defines the provider-neutral boundary between a
// human Model connection and the immutable backend consumed by routing.
package modelauthoring

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// Connection is the complete human-authored physical Model connection. The
// selected Provider Integration owns protocol defaults and compiles declared
// provider fields. Transport contains only provider-neutral, non-secret wire
// overrides that can be validated after that compilation.
type Connection struct {
	Name             string             `yaml:"name,omitempty" json:"name,omitempty"`
	Provider         string             `yaml:"provider" json:"provider"`
	Interface        string             `yaml:"interface,omitempty" json:"interface,omitempty"`
	Endpoint         string             `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`
	Model            string             `yaml:"model" json:"model"`
	Credential       string             `yaml:"credential,omitempty" json:"credential,omitempty"`
	Weight           string             `yaml:"weight,omitempty" json:"weight,omitempty"`
	ConnectionFields map[string]any     `yaml:"connection_fields,omitempty" json:"connectionFields,omitempty"`
	Transport        TransportOverrides `yaml:"transport,omitempty" json:"transport,omitempty"`
}

// TransportOverrides are applied to the immutable connection emitted by a
// Provider Integration. Credential headers are forbidden here and are always
// materialized by the credential adapter at request time.
type TransportOverrides struct {
	Path    string            `yaml:"path,omitempty" json:"path,omitempty"`
	Headers map[string]string `yaml:"headers,omitempty" json:"headers,omitempty"`
}

// CompileRequest adds compiler-owned identity to one human connection.
type CompileRequest struct {
	BackendID  string
	Connection Connection
}

// CompileResult is provider-neutral immutable routing state. CatalogRevision
// pins the exact Integration snapshot used for compilation.
type CompileResult struct {
	CatalogRevision string
	Backend         routingsnapshot.Backend
}

// ConnectionCompiler resolves Provider Integration data outside the config
// package. Implementations must be deterministic and side-effect free; they
// must never load or return secret material.
type ConnectionCompiler interface {
	CompileConnection(context.Context, CompileRequest) (CompileResult, error)
}
