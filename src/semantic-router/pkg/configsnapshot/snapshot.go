package configsnapshot

import (
	"crypto/sha256"
	"encoding/hex"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Source is where an update came from.
type Source string

const (
	// SourceStartup is the configuration the Router started with.
	SourceStartup Source = "startup"
	// SourceFile is a change of the watched configuration file.
	SourceFile Source = "file"
	// SourceAPI is a change made through the management API.
	SourceAPI Source = "api"
	// SourceKubernetes is a configuration built from Kubernetes resources.
	SourceKubernetes Source = "kubernetes"
	// SourceRollback reactivates a recorded version.
	SourceRollback Source = "rollback"
)

// Origin attributes an update to its source and, when known, to who made it.
type Origin struct {
	Source Source `json:"source"`
	// Principal identifies who made the change through the management API.
	Principal string `json:"principal,omitempty"`
	// RequestID is the management request that made the change.
	RequestID string `json:"request_id,omitempty"`
	// RollbackOf is the version a rollback reactivates.
	RollbackOf uint64 `json:"rollback_of,omitempty"`
}

// Snapshot is one immutable, versioned configuration: the document it was
// compiled from, the configuration model, and its typed resources. Nothing in
// a snapshot changes after it is created; callers must not modify what its
// accessors return.
type Snapshot struct {
	version   uint64
	hash      string
	origin    Origin
	document  []byte
	config    *config.RouterConfig
	resources *Resources
	createdAt time.Time
	parts     parts
}

// Version orders the snapshots a Router has activated: each activation takes
// the next number, a rollback included.
func (s *Snapshot) Version() uint64 { return s.version }

// Hash identifies the source document. It is the SHA-256 of the document's
// bytes, which the management API also serves as its ETag.
func (s *Snapshot) Hash() string { return s.hash }

// Origin tells where the snapshot's update came from.
func (s *Snapshot) Origin() Origin { return s.origin }

// Document is the source document, or nil when the source had none.
func (s *Snapshot) Document() []byte { return s.document }

// Config is the configuration model compiled into the snapshot.
func (s *Snapshot) Config() *config.RouterConfig { return s.config }

// Resources are the snapshot's typed resources.
func (s *Snapshot) Resources() *Resources { return s.resources }

// CreatedAt is when the snapshot was compiled.
func (s *Snapshot) CreatedAt() time.Time { return s.createdAt }

// documentHash returns the identity of an update's document: the hash the
// loader recorded when it parsed it, else the hash of the bytes.
func documentHash(cfg *config.RouterConfig, document []byte) string {
	if cfg != nil && cfg.DocumentHash != "" {
		return cfg.DocumentHash
	}
	if document == nil {
		return ""
	}
	return documentDigest(document)
}

func documentDigest(document []byte) string {
	sum := sha256.Sum256(document)
	return hex.EncodeToString(sum[:])
}
