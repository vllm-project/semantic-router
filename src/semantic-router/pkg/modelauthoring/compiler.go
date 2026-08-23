// Package modelauthoring defines the provider-neutral boundary between a
// human Model connection and the immutable backend consumed by routing.
package modelauthoring

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routingsnapshot"
)

// Connection is the complete human-authored physical Model connection. The
// selected Provider Integration owns all protocol, path, header, and fixed
// endpoint defaults that are intentionally absent here.
type Connection struct {
	Provider   string `yaml:"provider" json:"provider"`
	Interface  string `yaml:"interface,omitempty" json:"interface,omitempty"`
	Endpoint   string `yaml:"endpoint,omitempty" json:"endpoint,omitempty"`
	Model      string `yaml:"model" json:"model"`
	Credential string `yaml:"credential,omitempty" json:"credential,omitempty"`
	Weight     string `yaml:"weight,omitempty" json:"weight,omitempty"`
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
