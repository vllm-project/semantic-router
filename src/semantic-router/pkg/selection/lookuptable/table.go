/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package lookuptable provides persisted lookup tables for session-aware routing
// decisions. It replaces hardcoded constants with data-driven values sourced from
// router replay records, routing history, and operator configuration.
//
// Three table types are supported:
//   - quality_gap: quality difference between models for a task family
//   - handoff_penalty: cost of switching models mid-session
//   - remaining_turn_prior: expected remaining turns per intent/domain
//
// Keys are deterministic strings of the form:
//
//	quality_gap::<task_family>::<current_model>::<candidate_model>
//	handoff_penalty::<from_model>::<to_model>
//	remaining_turn_prior::<intent_or_domain>
package lookuptable

import (
	"fmt"
	"strings"
	"time"
)

const keySep = "::"

// TableType identifies which lookup table a key belongs to.
type TableType string

const (
	// TableQualityGap stores quality-gap estimates between model pairs per task family.
	TableQualityGap TableType = "quality_gap"

	// TableHandoffPenalty stores the cost (latency / quality degradation) of
	// switching from one model to another within a session.
	TableHandoffPenalty TableType = "handoff_penalty"

	// TableRemainingTurnPrior stores the expected number of remaining turns for
	// a given intent or domain.
	TableRemainingTurnPrior TableType = "remaining_turn_prior"
)

// Key is a structured representation of a lookup key. Call String() to obtain
// the canonical wire format used for storage.
type Key struct {
	Table TableType

	// Quality-gap fields
	TaskFamily     string
	CurrentModel   string
	CandidateModel string

	// Handoff-penalty fields (reuse CurrentModel as FromModel, CandidateModel as ToModel)

	// Remaining-turn-prior fields
	IntentOrDomain string
}

// QualityGapKey returns a Key for the quality_gap table.
func QualityGapKey(taskFamily, currentModel, candidateModel string) Key {
	return Key{
		Table:          TableQualityGap,
		TaskFamily:     taskFamily,
		CurrentModel:   currentModel,
		CandidateModel: candidateModel,
	}
}

// HandoffPenaltyKey returns a Key for the handoff_penalty table.
func HandoffPenaltyKey(fromModel, toModel string) Key {
	return Key{
		Table:          TableHandoffPenalty,
		CurrentModel:   fromModel,
		CandidateModel: toModel,
	}
}

// RemainingTurnPriorKey returns a Key for the remaining_turn_prior table.
func RemainingTurnPriorKey(intentOrDomain string) Key {
	return Key{
		Table:          TableRemainingTurnPrior,
		IntentOrDomain: intentOrDomain,
	}
}

// String returns the canonical key string used for persistence and wire transfer.
//
//	quality_gap::<task_family>::<current_model>::<candidate_model>
//	handoff_penalty::<from_model>::<to_model>
//	remaining_turn_prior::<intent_or_domain>
func (k Key) String() string {
	switch k.Table {
	case TableQualityGap:
		return strings.Join([]string{string(k.Table), k.TaskFamily, k.CurrentModel, k.CandidateModel}, keySep)
	case TableHandoffPenalty:
		return strings.Join([]string{string(k.Table), k.CurrentModel, k.CandidateModel}, keySep)
	case TableRemainingTurnPrior:
		return strings.Join([]string{string(k.Table), k.IntentOrDomain}, keySep)
	default:
		return string(k.Table)
	}
}

// ParseKey parses a canonical key string into a Key struct.
func ParseKey(s string) (Key, error) {
	parts := strings.Split(s, keySep)
	if len(parts) < 2 {
		return Key{}, fmt.Errorf("lookuptable: invalid key %q: expected at least 2 segments", s)
	}

	table := TableType(parts[0])
	switch table {
	case TableQualityGap:
		if len(parts) != 4 {
			return Key{}, fmt.Errorf("lookuptable: quality_gap key requires 4 segments, got %d in %q", len(parts), s)
		}
		return QualityGapKey(parts[1], parts[2], parts[3]), nil

	case TableHandoffPenalty:
		if len(parts) != 3 {
			return Key{}, fmt.Errorf("lookuptable: handoff_penalty key requires 3 segments, got %d in %q", len(parts), s)
		}
		return HandoffPenaltyKey(parts[1], parts[2]), nil

	case TableRemainingTurnPrior:
		if len(parts) != 2 {
			return Key{}, fmt.Errorf("lookuptable: remaining_turn_prior key requires 2 segments, got %d in %q", len(parts), s)
		}
		return RemainingTurnPriorKey(parts[1]), nil

	default:
		return Key{}, fmt.Errorf("lookuptable: unknown table type %q in key %q", table, s)
	}
}

// Entry holds a single lookup value together with provenance metadata.
type Entry struct {
	// Value is the looked-up float64 (e.g. quality-gap score, penalty, turn count).
	Value float64 `json:"value"`

	// Source describes how this entry was produced:
	// "replay_derived", "config_override", or "manual".
	Source string `json:"source"`

	// UpdatedAt records when this entry was last written.
	UpdatedAt time.Time `json:"updated_at"`

	// SampleCount is the number of replay records that contributed to this
	// entry's value (0 for manually set entries).
	SampleCount int `json:"sample_count,omitempty"`

	// AggregationWindow describes the time span of data used to derive this
	// entry (e.g. "7d", "30d"). Empty for config overrides or manual entries.
	AggregationWindow string `json:"aggregation_window,omitempty"`
}

const (
	// SourceReplayDerived indicates the entry was computed from router replay records.
	SourceReplayDerived = "replay_derived"

	// SourceConfigOverride indicates the entry was set via YAML configuration.
	SourceConfigOverride = "config_override"

	// SourceManual indicates the entry was set manually (e.g. via API or test code).
	SourceManual = "manual"
)

// LookupTable is the read-only interface used by selectors at request time.
// Implementations must be safe for concurrent reads.
type LookupTable interface {
	// Get resolves a single key. Returns (entry, true) if found, or (Entry{}, false).
	Get(key Key) (Entry, bool)

	// QualityGap is a convenience accessor for TableQualityGap.
	// Returns (value, true) if a matching entry exists.
	QualityGap(taskFamily, currentModel, candidateModel string) (float64, bool)

	// HandoffPenalty is a convenience accessor for TableHandoffPenalty.
	// Returns (value, true) if a matching entry exists.
	HandoffPenalty(fromModel, toModel string) (float64, bool)

	// RemainingTurnPrior is a convenience accessor for TableRemainingTurnPrior.
	// Returns (value, true) if a matching entry exists.
	RemainingTurnPrior(intentOrDomain string) (float64, bool)
}

// LookupTableStorage extends LookupTable with write and persistence operations.
// Implementations must be safe for concurrent use.
type LookupTableStorage interface {
	LookupTable

	// Set writes or overwrites a single entry.
	Set(key Key, entry Entry) error

	// All returns a snapshot of all entries keyed by their canonical string.
	// The returned map is a copy; callers may mutate it freely.
	All() map[string]Entry

	// Load loads persisted state from the backing store (no-op for memory).
	Load() error

	// Save persists current state to the backing store (no-op for memory).
	Save() error

	// Close releases any resources held by the storage backend.
	Close() error
}
