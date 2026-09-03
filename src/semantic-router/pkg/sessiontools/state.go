// Package sessiontools implements the state model and storage for
// session-scoped sticky tool-set selection (issue #3347). This package owns
// selection-state algorithms, storage, bounds, and CAS; pkg/extproc stays
// the request-phase orchestrator and adapter (see PL-0042).
//
// State persisted here is identity-only: tool names and bounded fingerprint
// metadata, never full llmprotocol.Tool values, descriptions, JSON Schemas,
// arguments, results, prompts, authorization decisions, credentials, or raw
// session/principal identifiers.
package sessiontools

import (
	"encoding/json"
	"fmt"
	"time"
)

// SchemaVersion is the current State schema version. A stored State whose
// SchemaVersion differs is a miss, not a partially-trusted value — callers
// must delete it and select fresh rather than attempt migration.
const SchemaVersion uint16 = 1

// State is the persistent envelope for one session's sticky tool-set
// selection. Wire-neutral: safe to encode as JSON for either store backend.
type State struct {
	SchemaVersion uint16 `json:"schema_version"`
	// Revision increments only on a successful atomic update (see
	// Store.CompareAndSwap); it is the CAS linearization point.
	Revision uint64 `json:"revision"`
	// PolicyFingerprint, CatalogFingerprint, and CapabilityFingerprint are
	// canonical fingerprints (see pkg/tools/fingerprint.go) of the
	// selection policy, tool catalog, and model/wire capability set that
	// produced this state. A mismatch against the current request's
	// fingerprints means the stored state must be revalidated or
	// invalidated before reuse — this package does not itself compare
	// fingerprints against a live request; that belongs to the manager
	// that owns the merge/invalidation decision (a later task).
	PolicyFingerprint     string `json:"policy_fingerprint"`
	CatalogFingerprint    string `json:"catalog_fingerprint"`
	CapabilityFingerprint string `json:"capability_fingerprint"`
	// StrategyID records which relevance strategy produced the last
	// relevance-driven addition, for observability only.
	StrategyID string      `json:"strategy_id,omitempty"`
	Tools      []ToolState `json:"tools"`
	CreatedAt  time.Time   `json:"created_at"`
	LastSeenAt time.Time   `json:"last_seen_at"`
	ExpiresAt  time.Time   `json:"expires_at"`
}

// ToolState is one retained tool identity within a session's sticky set.
// Identity and bounded metadata only — never the tool's description, JSON
// Schema, or any other content requiring re-authorization to reconstruct.
type ToolState struct {
	Name                  string `json:"name"`
	DefinitionFingerprint string `json:"definition_fingerprint"`
	// Pinned marks a tool observed in an assistant tool call. Monotonic
	// until expiry/invalidation — pinning is never revoked by ordinary
	// bounded growth (see the deterministic merge algorithm, a later task).
	Pinned bool `json:"pinned,omitempty"`
	// FirstSeenTurn is the turn index at which this tool first entered the
	// session's sticky set, used to break ties deterministically during
	// bounded growth and eviction.
	FirstSeenTurn int `json:"first_seen_turn"`
}

// Validate reports whether s is a well-formed State that may be trusted and
// reused. maxTools and maxStateBytes should come from
// config.ToolSessionStoreConfig's Effective* helpers — this package does
// not read router config directly, keeping the state model independent of
// the config package.
//
// A malformed value here is always a miss (delete and select fresh), never
// a partially-trusted value — see PL-0042's Operating Rules.
func (s State) Validate(maxTools int, maxStateBytes int) error {
	if s.SchemaVersion != SchemaVersion {
		return fmt.Errorf("sessiontools: unsupported schema_version %d (want %d)", s.SchemaVersion, SchemaVersion)
	}
	if err := s.validateTimestamps(); err != nil {
		return err
	}
	if err := s.validateTools(maxTools); err != nil {
		return err
	}
	return s.validateEncodedSize(maxStateBytes)
}

func (s State) validateTimestamps() error {
	if s.CreatedAt.IsZero() || s.LastSeenAt.IsZero() || s.ExpiresAt.IsZero() {
		return fmt.Errorf("sessiontools: created_at, last_seen_at, and expires_at must all be set")
	}
	if s.CreatedAt.After(s.LastSeenAt) {
		return fmt.Errorf("sessiontools: created_at must not be after last_seen_at")
	}
	if s.LastSeenAt.After(s.ExpiresAt) {
		return fmt.Errorf("sessiontools: last_seen_at must not be after expires_at")
	}
	return nil
}

func (s State) validateTools(maxTools int) error {
	if maxTools > 0 && len(s.Tools) > maxTools {
		return fmt.Errorf("sessiontools: %d tools exceeds the bound of %d", len(s.Tools), maxTools)
	}
	seen := make(map[string]struct{}, len(s.Tools))
	for i := range s.Tools {
		tool := &s.Tools[i]
		if tool.Name == "" {
			return fmt.Errorf("sessiontools: tool at index %d has an empty name", i)
		}
		if tool.DefinitionFingerprint == "" {
			return fmt.Errorf("sessiontools: tool %q has an empty definition_fingerprint", tool.Name)
		}
		if tool.FirstSeenTurn < 0 {
			return fmt.Errorf("sessiontools: tool %q has a negative first_seen_turn", tool.Name)
		}
		if _, duplicate := seen[tool.Name]; duplicate {
			return fmt.Errorf("sessiontools: duplicate tool name %q", tool.Name)
		}
		seen[tool.Name] = struct{}{}
	}
	return nil
}

func (s State) validateEncodedSize(maxStateBytes int) error {
	if maxStateBytes <= 0 {
		return nil
	}
	encoded, err := json.Marshal(s)
	if err != nil {
		return fmt.Errorf("sessiontools: failed to encode state for size validation: %w", err)
	}
	if len(encoded) > maxStateBytes {
		return fmt.Errorf("sessiontools: encoded state is %d bytes, exceeds the bound of %d", len(encoded), maxStateBytes)
	}
	return nil
}

// Clone returns a deep copy of s. Every store and manager boundary in this
// package clones on read and on write so no caller ever receives a pointer
// into another caller's mutable state (see PL-0042's local-store rules).
func (s State) Clone() State {
	cloned := s
	if s.Tools != nil {
		cloned.Tools = append([]ToolState(nil), s.Tools...)
	}
	return cloned
}
