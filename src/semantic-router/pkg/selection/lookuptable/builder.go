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

package lookuptable

import (
	"fmt"
	"sort"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

// emaAlpha is the EMA smoothing factor, matching session telemetry (alpha = 0.2).
const emaAlpha = 0.2

// sessionWindowDuration is the maximum gap between consecutive records in the
// same Decision group that are still considered part of the same session.
// Records separated by more than this duration start a new pseudo-session.
const sessionWindowDuration = 30 * time.Minute

// Builder derives lookup table entries from router replay records.
//
// It is batch-oriented: call PopulateFromRecords with a slice of historical
// records and it writes derived entries to the underlying LookupTableStorage.
// It does not perform online / real-time updates.
type Builder struct {
	storage LookupTableStorage
}

// NewBuilder creates a Builder that writes derived entries into storage.
func NewBuilder(storage LookupTableStorage) *Builder {
	return &Builder{storage: storage}
}

// PopulateFromRecords scans the given replay records and derives lookup table
// entries for all three table types.
//
// Derivation logic:
//
//   - quality_gap: group records by Category (task family). For each category,
//     compute the average ConfidenceScore per SelectedModel. The quality gap from
//     model A to model B is avgScore(B) - avgScore(A). Only pairs where both
//     models have at least one record are emitted.
//
//   - handoff_penalty: detect model switches between consecutive records within
//     the same pseudo-session. Sessions are formed by grouping records that share
//     the same Decision name and whose timestamps are within sessionWindowDuration
//     (30 min) of each other. For each (fromModel → toModel) switch, accumulate
//     cost delta samples and aggregate using EMA (alpha = 0.2). Records that lack
//     ActualCost on either side are skipped to avoid biasing the EMA toward zero.
//
//   - remaining_turn_prior: for each Category, estimate the average total turns
//     per session using the same Decision+time-window grouping. The prior is the
//     average session length across all sessions containing that category.
//
// Config-override entries (Source == SourceConfigOverride) already present in
// storage are not overwritten.
func (b *Builder) PopulateFromRecords(records []store.Record) error {
	if len(records) == 0 {
		return nil
	}

	window := aggregationWindow(records)
	batch := make(map[Key]Entry)

	b.deriveQualityGaps(records, batch, window)
	b.deriveHandoffPenalties(records, batch, window)
	b.deriveRemainingTurnPriors(records, batch, window)

	// Respect existing config overrides: skip keys already set that way.
	for k := range batch {
		if existing, ok := b.storage.Get(k); ok && existing.Source == SourceConfigOverride {
			delete(batch, k)
		}
	}

	for k, v := range batch {
		if err := b.storage.Set(k, v); err != nil {
			return err
		}
	}

	logging.ComponentEvent("selection", "lookuptable_populated", map[string]interface{}{
		"record_count": len(records),
		"entry_count":  len(batch),
	})
	return nil
}

// deriveQualityGaps populates quality_gap entries from per-category model scores.
func (b *Builder) deriveQualityGaps(records []store.Record, batch map[Key]Entry, window string) {
	// category → model → (sum of confidence scores, count)
	type scoreAcc struct {
		sum float64
		n   int
	}
	acc := make(map[string]map[string]*scoreAcc) // category → model → acc

	for _, r := range records {
		if r.Category == "" || r.SelectedModel == "" || r.ConfidenceScore == 0 {
			continue
		}
		if _, ok := acc[r.Category]; !ok {
			acc[r.Category] = make(map[string]*scoreAcc)
		}
		if _, ok := acc[r.Category][r.SelectedModel]; !ok {
			acc[r.Category][r.SelectedModel] = &scoreAcc{}
		}
		acc[r.Category][r.SelectedModel].sum += r.ConfidenceScore
		acc[r.Category][r.SelectedModel].n++
	}

	for category, models := range acc {
		// Compute averages.
		avgs := make(map[string]float64, len(models))
		counts := make(map[string]int, len(models))
		for model, a := range models {
			avgs[model] = a.sum / float64(a.n)
			counts[model] = a.n
		}

		// Emit quality gap for every ordered pair (current, candidate).
		modelList := sortedKeys(avgs)
		for _, current := range modelList {
			for _, candidate := range modelList {
				if current == candidate {
					continue
				}
				gap := avgs[candidate] - avgs[current]
				key := QualityGapKey(category, current, candidate)
				batch[key] = Entry{
					Value:             gap,
					Source:            SourceReplayDerived,
					UpdatedAt:         time.Now(),
					SampleCount:       counts[current] + counts[candidate],
					AggregationWindow: window,
				}
			}
		}
	}
}

// groupIntoSessions groups records into sessions using persisted SessionID when
// available, and falls back to Decision+time-window heuristics for older replay
// records that lack session metadata.
func groupIntoSessions(records []store.Record) [][]store.Record {
	bySession := make(map[string][]store.Record)
	byDecision := make(map[string][]store.Record)
	for _, r := range records {
		if r.SessionID != "" {
			bySession[r.SessionID] = append(bySession[r.SessionID], r)
			continue
		}
		byDecision[r.Decision] = append(byDecision[r.Decision], r)
	}

	var sessions [][]store.Record
	for _, recs := range bySession {
		sort.Slice(recs, func(i, j int) bool {
			return recs[i].Timestamp.Before(recs[j].Timestamp)
		})
		sessions = append(sessions, recs)
	}

	for _, recs := range byDecision {
		sort.Slice(recs, func(i, j int) bool {
			return recs[i].Timestamp.Before(recs[j].Timestamp)
		})

		start := 0
		for i := 1; i < len(recs); i++ {
			if recs[i].Timestamp.Sub(recs[i-1].Timestamp) > sessionWindowDuration {
				sessions = append(sessions, recs[start:i])
				start = i
			}
		}
		sessions = append(sessions, recs[start:])
	}
	return sessions
}

// deriveHandoffPenalties detects model switches within pseudo-sessions and
// computes the associated cost delta via EMA.
func (b *Builder) deriveHandoffPenalties(records []store.Record, batch map[Key]Entry, window string) {
	sessions := groupIntoSessions(records)

	// (from, to) → EMA value + sample count
	type switchAcc struct {
		ema float64
		n   int
	}
	acc := make(map[[2]string]*switchAcc)

	for _, recs := range sessions {
		for i := 1; i < len(recs); i++ {
			prev, cur := recs[i-1], recs[i]
			if prev.SelectedModel == "" || cur.SelectedModel == "" {
				continue
			}
			if prev.SelectedModel == cur.SelectedModel {
				continue
			}
			// Model switch detected. Skip when cost data is missing on either
			// side to avoid biasing the EMA toward zero for unknown costs.
			delta, ok := costDelta(prev, cur)
			if !ok {
				continue
			}
			pair := [2]string{prev.SelectedModel, cur.SelectedModel}
			if _, ok := acc[pair]; !ok {
				acc[pair] = &switchAcc{ema: delta}
			} else {
				acc[pair].ema = emaAlpha*delta + (1-emaAlpha)*acc[pair].ema
			}
			acc[pair].n++
		}
	}

	for pair, a := range acc {
		key := HandoffPenaltyKey(pair[0], pair[1])
		batch[key] = Entry{
			Value:             a.ema,
			Source:            SourceReplayDerived,
			UpdatedAt:         time.Now(),
			SampleCount:       a.n,
			AggregationWindow: window,
		}
	}
}

// deriveRemainingTurnPriors estimates the expected number of remaining turns
// for each category and emits remaining_turn_prior entries.
//
// The value stored is a first-turn prior approximation: avgSessionLength - 1.
// This represents "how many turns are expected to remain after the first turn
// in a session of this category". A policy reading this value at turn N should
// subtract (N-1) to get the remaining-turns estimate for that turn.
//
// Example: if coding sessions average 5 turns, the stored value is 4.0,
// meaning that at turn 1 the router expects 4 more turns to follow.
func (b *Builder) deriveRemainingTurnPriors(records []store.Record, batch map[Key]Entry, window string) {
	sessions := groupIntoSessions(records)

	// category → list of session lengths (in turns)
	categoryLengths := make(map[string][]int)
	for _, recs := range sessions {
		// Collect unique categories that appear in this session.
		seen := make(map[string]bool)
		for _, r := range recs {
			if r.Category != "" {
				seen[r.Category] = true
			}
		}
		for cat := range seen {
			categoryLengths[cat] = append(categoryLengths[cat], len(recs))
		}
	}

	for category, lengths := range categoryLengths {
		if len(lengths) == 0 {
			continue
		}
		sum := 0
		for _, l := range lengths {
			sum += l
		}
		avgLen := float64(sum) / float64(len(lengths))
		// Subtract 1 to convert average total length into a first-turn
		// remaining-turns prior (clamp to 0 to avoid negative values for
		// single-turn sessions).
		prior := avgLen - 1.0
		if prior < 0 {
			prior = 0
		}
		key := RemainingTurnPriorKey(category)
		batch[key] = Entry{
			Value:             prior,
			Source:            SourceReplayDerived,
			UpdatedAt:         time.Now(),
			SampleCount:       len(lengths),
			AggregationWindow: window,
		}
	}
}

// costDelta computes the cost difference between two records.
// Returns (delta, true) when both records have cost data, or (0, false) when
// either record is missing cost data. Callers should skip the (0, false) case
// to avoid biasing the EMA toward zero for unknown costs.
func costDelta(prev, cur store.Record) (float64, bool) {
	if prev.ActualCost == nil || cur.ActualCost == nil {
		return 0, false
	}
	return *cur.ActualCost - *prev.ActualCost, true
}

// aggregationWindow computes a human-readable duration string (e.g. "7d", "3h")
// representing the time span between the earliest and latest record timestamps.
// Returns "" if fewer than 2 records have non-zero timestamps.
func aggregationWindow(records []store.Record) string {
	var earliest, latest time.Time
	for _, r := range records {
		if r.Timestamp.IsZero() {
			continue
		}
		if earliest.IsZero() || r.Timestamp.Before(earliest) {
			earliest = r.Timestamp
		}
		if latest.IsZero() || r.Timestamp.After(latest) {
			latest = r.Timestamp
		}
	}
	if earliest.IsZero() || latest.IsZero() {
		return ""
	}
	d := latest.Sub(earliest)
	if d < time.Minute {
		return ""
	}
	if d >= 24*time.Hour {
		days := int(d.Hours() / 24)
		return fmt.Sprintf("%dd", days)
	}
	return fmt.Sprintf("%dh", int(d.Hours()))
}

// sortedKeys returns the sorted keys of a string→float64 map.
func sortedKeys(m map[string]float64) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
