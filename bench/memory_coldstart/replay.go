package main

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strings"
	"time"

	openai "github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
)

// The memory package reads this switch but does not export its name.
const deterministicEmbeddingsEnv = "VLLM_SR_DETERMINISTIC_EMBEDDINGS"

const replayUserID = "coldstart-user"

type phase string

const (
	phaseNoMemory  phase = "no_memory"
	phaseFirstSeen phase = "first_seen"
	phaseRecurring phase = "recurring"
	phaseStale     phase = "stale_or_conflicting"
)

var phaseOrder = []phase{phaseNoMemory, phaseFirstSeen, phaseRecurring, phaseStale}

type turn struct {
	User      string
	Assistant string
}

// A memory answers the probe when its lowercased content contains any Expect
// fragment. Stale fragments name facts the user has since corrected.
type probe struct {
	Phase  phase
	Query  string
	Expect []string
	Stale  []string
}

// Probes open the session, before its turns are stored, the way a user's first
// questions in a new conversation can only be answered from earlier sessions.
type session struct {
	Day    int
	Probes []probe
	Turns  []turn
}

type replayOptions struct {
	Threshold float32
	Limit     int
}

type retrievedMemory struct {
	Score   float32 `json:"score"`
	Content string  `json:"content"`
}

type probeResult struct {
	Day        int               `json:"day"`
	Phase      phase             `json:"phase"`
	Query      string            `json:"query"`
	Expect     []string          `json:"expect,omitempty"`
	Superseded []string          `json:"superseded,omitempty"`
	Hit        bool              `json:"hit"`
	Right      bool              `json:"right"`
	Top1       bool              `json:"top1"`
	Stale      bool              `json:"stale"`
	Retrieved  []retrievedMemory `json:"retrieved"`
}

type tally struct {
	Probes     int `json:"probes"`
	Hit        int `json:"hit"`
	Right      int `json:"right"`
	Top1       int `json:"top1"`
	Stale      int `json:"stale"`
	Ungrounded int `json:"ungrounded"`
}

func (t *tally) add(result probeResult) {
	t.Probes++
	if result.Hit {
		t.Hit++
	}
	if result.Right {
		t.Right++
	} else {
		t.Ungrounded++
	}
	if result.Top1 {
		t.Top1++
	}
	if result.Stale {
		t.Stale++
	}
}

type phaseSummary struct {
	Phase phase `json:"phase"`
	tally
}

type sessionSummary struct {
	Day      int `json:"day"`
	Memories int `json:"memories"`
	tally
}

type report struct {
	Embeddings string           `json:"embeddings"`
	Threshold  float32          `json:"threshold"`
	Limit      int              `json:"limit"`
	Memories   int              `json:"memories"`
	Phases     []phaseSummary   `json:"phases"`
	Sessions   []sessionSummary `json:"sessions"`
	Probes     []probeResult    `json:"probes"`
}

// Embedding memory.Store directly would name the field Store, which clashes
// with the Store method below.
type backingStore = memory.Store

// sessionClock dates each write to its replayed session, so recency decay sees
// a month of history rather than one written in milliseconds. The one-minute
// step keeps a session's turns in order.
type sessionClock struct {
	backingStore
	now time.Time
}

func (c *sessionClock) Store(ctx context.Context, m *memory.Memory) error {
	c.now = c.now.Add(time.Minute)
	m.CreatedAt = c.now
	return c.backingStore.Store(ctx, m)
}

func replay(ctx context.Context, sessions []session, opts replayOptions) (report, error) {
	if len(sessions) == 0 {
		return report{}, errors.New("scenario has no sessions")
	}
	embedding := memory.EmbeddingConfig{Model: memory.EmbeddingModelBERT}
	fingerprint, deterministic := memory.DeterministicEmbeddingFingerprint(embedding)
	if !deterministic {
		return report{}, fmt.Errorf("the replay runs on deterministic embeddings; set %s=1", deterministicEmbeddingsEnv)
	}
	store := &sessionClock{backingStore: memory.NewInMemoryStoreWithConfig(embedding)}
	chunks := memory.NewMemoryChunkStore(store)
	filter := memory.NewMemoryFilter(config.MemoryReflectionConfig{}, nil)

	rep := report{Embeddings: fingerprint, Threshold: opts.Threshold, Limit: opts.Limit}
	phases := make(map[phase]*tally, len(phaseOrder))
	lastDay := sessions[len(sessions)-1].Day
	start := time.Now()
	for _, s := range sessions {
		store.now = start.AddDate(0, 0, s.Day-lastDay)
		memories, err := countMemories(ctx, store)
		if err != nil {
			return report{}, err
		}
		summary := sessionSummary{Day: s.Day, Memories: memories}
		for _, p := range s.Probes {
			result, err := runProbe(ctx, store, filter, p, s.Day, opts)
			if err != nil {
				return report{}, err
			}
			summary.add(result)
			if phases[p.Phase] == nil {
				phases[p.Phase] = &tally{}
			}
			phases[p.Phase].add(result)
			rep.Probes = append(rep.Probes, result)
		}
		rep.Sessions = append(rep.Sessions, summary)
		if err := storeTurns(ctx, chunks, fmt.Sprintf("day-%d", s.Day), s.Turns); err != nil {
			return report{}, err
		}
	}
	for _, name := range phaseOrder {
		if counts, ok := phases[name]; ok {
			rep.Phases = append(rep.Phases, phaseSummary{Phase: name, tally: *counts})
		}
	}
	memories, err := countMemories(ctx, store)
	if err != nil {
		return report{}, err
	}
	rep.Memories = memories
	return rep, nil
}

func runProbe(
	ctx context.Context,
	store memory.Store,
	filter memory.MemoryFilter,
	p probe,
	day int,
	opts replayOptions,
) (probeResult, error) {
	found, err := store.Retrieve(ctx, memory.RetrieveOptions{
		Query:     p.Query,
		UserID:    replayUserID,
		Limit:     opts.Limit,
		Threshold: opts.Threshold,
	})
	if err != nil {
		return probeResult{}, fmt.Errorf("retrieve %q: %w", p.Query, err)
	}
	return grade(p, day, filter.Filter(found)), nil
}

func countMemories(ctx context.Context, store memory.Store) (int, error) {
	listed, err := store.List(ctx, memory.ListOptions{UserID: replayUserID})
	if err != nil {
		return 0, fmt.Errorf("list memories: %w", err)
	}
	return listed.Total, nil
}

func storeTurns(ctx context.Context, chunks *memory.MemoryExtractor, sessionID string, turns []turn) error {
	var history []openai.ChatCompletionMessageParamUnion
	for _, t := range turns {
		if _, err := chunks.ProcessResponseWithHistoryCount(ctx, sessionID, replayUserID, t.User, t.Assistant, history); err != nil {
			return fmt.Errorf("store turn %q: %w", t.User, err)
		}
		history = append(history,
			memory.SDKMessageForRole("user", t.User),
			memory.SDKMessageForRole("assistant", t.Assistant),
		)
	}
	return nil
}

func grade(p probe, day int, found []*memory.RetrieveResult) probeResult {
	result := probeResult{
		Day:        day,
		Phase:      p.Phase,
		Query:      p.Query,
		Expect:     p.Expect,
		Superseded: p.Stale,
		Hit:        len(found) > 0,
		Retrieved:  make([]retrievedMemory, 0, len(found)),
	}
	for rank, r := range found {
		content := strings.ToLower(r.Memory.Content)
		if containsAny(content, p.Expect) {
			result.Right = true
			result.Top1 = result.Top1 || rank == 0
		}
		result.Stale = result.Stale || containsAny(content, p.Stale)
		result.Retrieved = append(result.Retrieved, retrievedMemory{Score: r.Score, Content: r.Memory.Content})
	}
	return result
}

func containsAny(content string, fragments []string) bool {
	return slices.ContainsFunc(fragments, func(fragment string) bool {
		return strings.Contains(content, fragment)
	})
}
