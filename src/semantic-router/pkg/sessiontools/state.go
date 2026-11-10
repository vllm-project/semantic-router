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
// SchemaVersion differs is a miss, not a partially-trusted value. The manager
// conditionally replaces it using its observed CAS token, never deletion by
// key. Version 2 adds Turn; older state is deliberately not migrated.
const SchemaVersion uint16 = 2

// State is the persistent envelope for one session's sticky tool-set
// selection. Wire-neutral: safe to encode as JSON for either store backend.
type State struct {
	SchemaVersion uint16 `json:"schema_version"`
	// Revision changes only on a successful atomic update (see
	// Store.CompareAndSwap); it is the CAS linearization point. Treat it as
	// an opaque token, not a per-key update count: a conforming Store
	// implementation must guarantee that a revision is never reused at a
	// given key across that key's full lifetime, including after it
	// expires and is later recreated — a fresh incarnation must never be
	// able to collide with a value a caller captured from an earlier,
	// since-reclaimed incarnation of the same key. (MemoryStore satisfies
	// this with a store-wide monotonic counter, not a per-key one; see
	// store_memory.go's nextRevision.)
	Revision uint64 `json:"revision"`
	// Turn counts successful planner updates within this continuity scope,
	// independently of Revision. FirstSeenTurn values refer to this counter.
	// A reset starts at 1; zero is unset, and MaxSelectionTurn exhaustion is an error.
	Turn uint64 `json:"turn"`
	// PolicyFingerprint, CatalogFingerprint, and CapabilityFingerprint are
	// canonical fingerprints (see pkg/tools/fingerprint.go) of the
	// selection policy, tool catalog, and model/wire capability set that
	// produced this state. A mismatch against the current request's
	// fingerprints means the stored state must be revalidated or
	// invalidated before reuse. Manager compares the complete fingerprints
	// supplied by its caller; it does not derive runtime authorization.
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
	// bounded growth. A policy disabling pinning must change its fingerprint.
	Pinned bool `json:"pinned,omitempty"`
	// FirstSeenTurn is the turn index at which this tool first entered the
	// session's sticky set, used to break ties deterministically during
	// bounded growth and eviction.
	FirstSeenTurn int `json:"first_seen_turn"`
}

// Validate reports whether s is a well-formed State that may be trusted and
// reused. maxTools and maxStateBytes should come from
// the effective sticky policy and global store byte bound respectively.
//
// Malformed state is never partially trusted. Replacing a stale snapshot
// still requires its observed revision, even when validation fails.
func (s State) Validate(maxTools int, maxStateBytes int) error {
	if s.SchemaVersion != SchemaVersion {
		return fmt.Errorf("sessiontools: unsupported schema_version %d (want %d)", s.SchemaVersion, SchemaVersion)
	}
	if s.Turn == 0 || s.Turn > MaxSelectionTurn {
		return fmt.Errorf("sessiontools: turn must be in 1..%d", MaxSelectionTurn)
	}
	if !validFingerprint(s.PolicyFingerprint) || !validFingerprint(s.CatalogFingerprint) ||
		!validFingerprint(s.CapabilityFingerprint) || len(s.StrategyID) > 128 {
		return fmt.Errorf("sessiontools: missing or malformed fingerprint/strategy metadata")
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
		identity, err := normalizeIdentity(ToolIdentity{tool.Name, tool.DefinitionFingerprint})
		if err != nil || identity.Name != tool.Name {
			return fmt.Errorf("sessiontools: tool at index %d has a malformed identity", i)
		}
		if tool.FirstSeenTurn < 1 {
			return fmt.Errorf("sessiontools: tool at index %d has an unset or negative first_seen_turn", i)
		}
		if uint64(tool.FirstSeenTurn) > s.Turn {
			return fmt.Errorf("sessiontools: first_seen_turn exceeds state turn")
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
	size, err := s.encodedSize()
	if err != nil {
		return err
	}
	if size > maxStateBytes {
		return fmt.Errorf("%w: %d bytes exceeds %d", ErrStateTooLarge, size, maxStateBytes)
	}
	return nil
}

func (s State) encodedSize() (int, error) {
	encoded, err := json.Marshal(s)
	if err != nil {
		return 0, fmt.Errorf("sessiontools: failed to encode state: %w", err)
	}
	return len(encoded), nil
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
