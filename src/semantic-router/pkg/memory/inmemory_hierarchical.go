package memory

import (
	"context"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var _ HierarchicalStore = (*InMemoryStore)(nil)

// InMemoryHierarchicalConfig holds tunable parameters for in-memory hierarchical search.
type InMemoryHierarchicalConfig struct {
	// MaxCategoriesPerDepth caps how many categories are explored at each depth level (default: 10).
	MaxCategoriesPerDepth int

	// CandidateThresholdFactor is multiplied with the user threshold for pre-filtering
	// candidates and category-level gating (default: 0.7).
	CandidateThresholdFactor float32

	// AbstractFallbackMaxLen is the max length used when truncating content as a fallback
	// for a missing abstract on related memories (default: 150).
	AbstractFallbackMaxLen int
}

// ApplyDefaults fills zero-valued fields.
func (c *InMemoryHierarchicalConfig) ApplyDefaults() {
	if c.MaxCategoriesPerDepth <= 0 {
		c.MaxCategoriesPerDepth = 10
	}
	if c.CandidateThresholdFactor <= 0 {
		c.CandidateThresholdFactor = 0.7
	}
	if c.AbstractFallbackMaxLen <= 0 {
		c.AbstractFallbackMaxLen = 150
	}
}

// HierarchicalRetrieve implements the HierarchicalStore interface for InMemoryStore.
func (s *InMemoryStore) HierarchicalRetrieve(ctx context.Context, opts HierarchicalRetrieveOptions) ([]*RetrieveResult, error) {
	return s.HierarchicalRetrieveWithConfig(ctx, opts, InMemoryHierarchicalConfig{})
}

// HierarchicalRetrieveWithConfig is the configurable variant.
func (s *InMemoryStore) HierarchicalRetrieveWithConfig(ctx context.Context, opts HierarchicalRetrieveOptions, cfg InMemoryHierarchicalConfig) ([]*RetrieveResult, error) {
	if !s.enabled {
		return nil, nil
	}
	opts.ApplyDefaults()
	cfg.ApplyDefaults()

	s.mu.RLock()
	defer s.mu.RUnlock()

	queryEmbedding, err := GenerateEmbedding(opts.Query, s.embeddingConfig)
	if err != nil {
		return nil, err
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultHierarchicalLimit
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = DefaultMemoryConfig().DefaultSimilarityThreshold
	}

	var candidates []*Memory
	for _, mem := range s.memories {
		if !s.passesAccessFilter(mem, opts) {
			continue
		}
		candidates = append(candidates, mem)
	}

	if len(candidates) == 0 {
		return []*RetrieveResult{}, nil
	}

	// Build hybrid scorer if hybrid config is provided.
	var hybridScorer *MemHybridScorer
	if opts.Hybrid != nil {
		docs := make(map[string]string, len(candidates))
		for _, mem := range candidates {
			docs[mem.ID] = mem.Content
		}
		hybridScorer = BuildMemHybridScorer(docs, opts.Hybrid)
	}

	// scoreMemory returns the fused hybrid score or pure cosine depending on config.
	scoreMemory := func(mem *Memory) float32 {
		cosine := cosineSimilarity(queryEmbedding, mem.Embedding)
		if hybridScorer == nil {
			return cosine
		}
		return hybridScorer.FusedScore(mem.ID, cosine, opts.Query)
	}

	type scoredMem struct {
		mem   *Memory
		score float32
	}
	var categories []scoredMem
	var leaves []scoredMem

	candidateThreshold := threshold * cfg.CandidateThresholdFactor
	for _, mem := range candidates {
		sim := scoreMemory(mem)
		if sim < candidateThreshold {
			continue
		}
		sm := scoredMem{mem: mem, score: sim}
		if mem.IsCategory {
			categories = append(categories, sm)
		} else {
			leaves = append(leaves, sm)
		}
	}

	if len(categories) == 0 {
		var results []*RetrieveResult
		for _, l := range leaves {
			if l.score >= threshold {
				results = append(results, &RetrieveResult{Memory: l.mem, Score: l.score})
			}
		}
		sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
		if len(results) > limit {
			results = results[:limit]
		}
		s.populateRelations(results, opts, cfg)
		return results, nil
	}

	sort.Slice(categories, func(i, j int) bool { return categories[i].score > categories[j].score })

	alpha := opts.ScorePropAlpha
	collected := make(map[string]*RetrieveResult)

	for _, l := range leaves {
		if l.score >= threshold {
			collected[l.mem.ID] = &RetrieveResult{Memory: l.mem, Score: l.score}
		}
	}

	maxCategories := cfg.MaxCategoriesPerDepth
	if len(categories) < maxCategories {
		maxCategories = len(categories)
	}

	for depth := 0; depth < opts.MaxDepth; depth++ {
		newCategories := make([]scoredMem, 0)

		for ci := 0; ci < maxCategories && ci < len(categories); ci++ {
			cat := categories[ci]
			if cat.score < candidateThreshold {
				continue
			}

			for _, child := range candidates {
				if child.ParentID != cat.mem.ID {
					continue
				}
				if _, already := collected[child.ID]; already {
					continue
				}

				childSim := scoreMemory(child)
				propagated := PropagateScore(childSim, cat.score, alpha)

				if child.IsCategory {
					newCategories = append(newCategories, scoredMem{mem: child, score: propagated})
				} else if propagated >= threshold {
					collected[child.ID] = &RetrieveResult{Memory: child, Score: propagated}
				}
			}
		}

		if len(newCategories) == 0 {
			break
		}
		categories = newCategories
		sort.Slice(categories, func(i, j int) bool { return categories[i].score > categories[j].score })
		maxCategories = cfg.MaxCategoriesPerDepth
		if len(categories) < maxCategories {
			maxCategories = len(categories)
		}
	}

	results := make([]*RetrieveResult, 0, len(collected))
	for _, r := range collected {
		results = append(results, r)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > limit {
		results = results[:limit]
	}

	s.populateRelations(results, opts, cfg)

	logging.Debugf("InMemoryStore.HierarchicalRetrieve: returning %d results", len(results))
	return results, nil
}

// passesAccessFilter checks if a memory is accessible given the retrieval options.
func (s *InMemoryStore) passesAccessFilter(mem *Memory, opts HierarchicalRetrieveOptions) bool {
	if mem.UserID == opts.UserID {
		return true
	}

	if opts.IncludeGroupLevel && len(opts.GroupIDs) > 0 {
		if mem.Visibility == VisibilityGroup || mem.Visibility == VisibilityPublic {
			for _, gid := range opts.GroupIDs {
				if mem.GroupID == gid {
					return true
				}
			}
		}
	}

	if opts.IncludeGroupLevel && mem.Visibility == VisibilityPublic {
		return true
	}

	return false
}

// StoreRelation persists a relation in-memory by updating the RelatedIDs on both endpoints.
func (s *InMemoryStore) StoreRelation(ctx context.Context, rel MemoryRelation) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if from, ok := s.memories[rel.FromID]; ok {
		if !containsString(from.RelatedIDs, rel.ToID) {
			from.RelatedIDs = append(from.RelatedIDs, rel.ToID)
		}
	}
	if to, ok := s.memories[rel.ToID]; ok {
		if !containsString(to.RelatedIDs, rel.FromID) {
			to.RelatedIDs = append(to.RelatedIDs, rel.FromID)
		}
	}

	if s.relations == nil {
		s.relations = make(map[string][]MemoryRelation)
	}
	s.relations[rel.FromID] = append(s.relations[rel.FromID], rel)

	reverse := MemoryRelation{
		FromID:    rel.ToID,
		ToID:      rel.FromID,
		Reason:    rel.Reason,
		Strength:  rel.Strength,
		CreatedAt: rel.CreatedAt,
	}
	s.relations[rel.ToID] = append(s.relations[rel.ToID], reverse)

	return nil
}

// GetRelations returns relations originating from the given memory ID.
func (s *InMemoryStore) GetRelations(ctx context.Context, memoryID string, limit int) ([]MemoryRelation, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.relations == nil {
		return nil, nil
	}

	rels := s.relations[memoryID]
	if limit > 0 && len(rels) > limit {
		rels = rels[:limit]
	}
	return rels, nil
}

func (s *InMemoryStore) populateRelations(results []*RetrieveResult, opts HierarchicalRetrieveOptions, cfg InMemoryHierarchicalConfig) {
	if !opts.EnableRelations || s.relations == nil {
		return
	}

	maxRels := opts.MaxRelationsPerHit
	if maxRels <= 0 {
		maxRels = 5
	}

	for _, r := range results {
		rels := s.relations[r.Memory.ID]
		if len(rels) == 0 {
			continue
		}
		if len(rels) > maxRels {
			rels = rels[:maxRels]
		}

		for _, rel := range rels {
			related := &RelatedMemory{
				ID:     rel.ToID,
				Reason: rel.Reason,
				Score:  rel.Strength,
			}
			if mem, ok := s.memories[rel.ToID]; ok {
				related.Abstract = mem.Abstract
				if related.Abstract == "" && len(mem.Content) > 0 {
					maxLen := cfg.AbstractFallbackMaxLen
					if len(mem.Content) > maxLen {
						related.Abstract = mem.Content[:maxLen] + "..."
					} else {
						related.Abstract = mem.Content
					}
				}
			}
			r.Related = append(r.Related, related)
		}
	}
}

func containsString(slice []string, s string) bool {
	for _, v := range slice {
		if v == s {
			return true
		}
	}
	return false
}
