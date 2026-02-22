package memory

import (
	"context"
	"fmt"
	"sort"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	// DefaultRelationThreshold is the minimum cosine similarity between two memories
	// for an automatic relation link to be created.
	DefaultRelationThreshold float32 = 0.85

	// DefaultMaxRelationsPerMemory caps the total number of relations per memory.
	DefaultMaxRelationsPerMemory = 10
)

// AutoLinkOptions configures automatic relation creation when a memory is stored.
type AutoLinkOptions struct {
	// Threshold is the minimum cosine similarity to create a link (default: 0.85).
	Threshold float32

	// MaxRelations caps how many relations a single memory can have (default: 10).
	MaxRelations int

	// CandidateMultiplier controls how many extra candidates to fetch relative to MaxRelations
	// to account for self-matches and low-scoring results (default: 2).
	CandidateMultiplier int

	// Reason is the reason string stored on created relations. If empty, no reason is stored.
	Reason string

	// CrossGroup allows linking memories across group boundaries when both are
	// visible to each other (group or public visibility).
	CrossGroup bool
}

// ApplyDefaults fills zero-valued fields.
func (o *AutoLinkOptions) ApplyDefaults() {
	if o.Threshold <= 0 {
		o.Threshold = DefaultRelationThreshold
	}
	if o.MaxRelations <= 0 {
		o.MaxRelations = DefaultMaxRelationsPerMemory
	}
	if o.CandidateMultiplier <= 0 {
		o.CandidateMultiplier = 2
	}
}

// AutoLinkNewMemory finds existing memories similar to the newly stored memory
// and creates bidirectional relations. Returns the number of links created and any error.
func AutoLinkNewMemory(ctx context.Context, hs HierarchicalStore, newMem *Memory, opts AutoLinkOptions) (int, error) {
	opts.ApplyDefaults()

	if newMem.IsCategory {
		return 0, nil
	}

	searchOpts := RetrieveOptions{
		Query:     newMem.Content,
		UserID:    newMem.UserID,
		Limit:     opts.MaxRelations * opts.CandidateMultiplier,
		Threshold: opts.Threshold,
	}

	candidates, err := hs.Retrieve(ctx, searchOpts)
	if err != nil {
		return 0, fmt.Errorf("retrieval for auto-linking memory %s: %w", newMem.ID, err)
	}

	type scored struct {
		id    string
		score float32
	}
	var links []scored

	for _, c := range candidates {
		if c.Memory.ID == newMem.ID {
			continue
		}
		if c.Score < opts.Threshold {
			continue
		}
		if len(c.Memory.RelatedIDs) >= opts.MaxRelations {
			continue
		}
		links = append(links, scored{id: c.Memory.ID, score: c.Score})
	}

	sort.Slice(links, func(i, j int) bool { return links[i].score > links[j].score })
	if len(links) > opts.MaxRelations {
		links = links[:opts.MaxRelations]
	}

	now := time.Now()
	var firstErr error
	created := 0
	for _, link := range links {
		rel := MemoryRelation{
			FromID:    newMem.ID,
			ToID:      link.id,
			Reason:    opts.Reason,
			Strength:  link.score,
			CreatedAt: now,
		}
		if err := hs.StoreRelation(ctx, rel); err != nil {
			logging.Warnf("AutoLinkNewMemory: failed to store relation %s->%s: %v", newMem.ID, link.id, err)
			if firstErr == nil {
				firstErr = err
			}
			continue
		}
		created++
	}

	if created > 0 {
		logging.Debugf("AutoLinkNewMemory: created %d relations for memory %s", created, newMem.ID)
	}

	return created, firstErr
}
