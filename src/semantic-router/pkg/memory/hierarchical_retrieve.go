package memory

import (
	"container/heap"
	"context"
	"fmt"
	"sort"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

type hierarchicalCandidate struct {
	memoryID string
	score    float32
	depth    int
}

type candidateHeap []hierarchicalCandidate

func (h candidateHeap) Len() int            { return len(h) }
func (h candidateHeap) Less(i, j int) bool  { return h[i].score > h[j].score }
func (h candidateHeap) Swap(i, j int)       { h[i], h[j] = h[j], h[i] }
func (h *candidateHeap) Push(x interface{}) { *h = append(*h, x.(hierarchicalCandidate)) }
func (h *candidateHeap) Pop() interface{} {
	old := *h
	n := len(old)
	item := old[n-1]
	*h = old[:n-1]
	return item
}

// HierarchicalSearchConfig holds tunable parameters for the generic hierarchical search.
type HierarchicalSearchConfig struct {
	// MaxConvergenceRounds stops the drill-down loop when the top-K set is unchanged
	// for this many consecutive rounds (default: 3).
	MaxConvergenceRounds int

	// CategorySearchTopK controls how many candidates are fetched in Phase 1 (default: 20).
	CategorySearchTopK int

	// Phase1ThresholdFactor is multiplied with the user threshold to relax filtering in
	// Phase 1 so more category candidates are found (default: 0.8).
	Phase1ThresholdFactor float32

	// ChildThresholdFactor is multiplied with the user threshold for child searches (default: 0.7).
	ChildThresholdFactor float32

	// CategorySeedFactor is multiplied with the user threshold to decide which categories
	// are seeded into the priority queue (default: 0.8).
	CategorySeedFactor float32
}

// ApplyDefaults fills zero-valued fields.
func (c *HierarchicalSearchConfig) ApplyDefaults() {
	if c.MaxConvergenceRounds <= 0 {
		c.MaxConvergenceRounds = 3
	}
	if c.CategorySearchTopK <= 0 {
		c.CategorySearchTopK = 20
	}
	if c.Phase1ThresholdFactor <= 0 {
		c.Phase1ThresholdFactor = 0.8
	}
	if c.ChildThresholdFactor <= 0 {
		c.ChildThresholdFactor = 0.7
	}
	if c.CategorySeedFactor <= 0 {
		c.CategorySeedFactor = 0.8
	}
}

// DefaultHierarchicalSearchConfig returns the default configuration.
func DefaultHierarchicalSearchConfig() HierarchicalSearchConfig {
	cfg := HierarchicalSearchConfig{}
	cfg.ApplyDefaults()
	return cfg
}

// HierarchicalRetrieveFromStore performs two-phase hierarchical retrieval on any Store.
// Phase 1: Broad search, partition results into categories and leaves.
// Phase 2: Drill down from top categories into children with score propagation.
//
// This is the generic fallback for stores that don't implement HierarchicalStore natively.
func HierarchicalRetrieveFromStore(
	ctx context.Context,
	store Store,
	opts HierarchicalRetrieveOptions,
) ([]*RetrieveResult, error) {
	return HierarchicalRetrieveFromStoreWithConfig(ctx, store, opts, DefaultHierarchicalSearchConfig())
}

// HierarchicalRetrieveFromStoreWithConfig is the configurable variant.
func HierarchicalRetrieveFromStoreWithConfig(
	ctx context.Context,
	store Store,
	opts HierarchicalRetrieveOptions,
	cfg HierarchicalSearchConfig,
) ([]*RetrieveResult, error) {
	opts.ApplyDefaults()
	cfg.ApplyDefaults()

	if opts.Query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if opts.UserID == "" && !opts.IncludeGroupLevel {
		return nil, fmt.Errorf("user id or group ids required")
	}

	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultHierarchicalLimit
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = DefaultMemoryConfig().DefaultSimilarityThreshold
	}

	phase1Limit := cfg.CategorySearchTopK
	if phase1Limit < limit*4 {
		phase1Limit = limit * 4
	}

	phase1Opts := RetrieveOptions{
		Query:     opts.Query,
		UserID:    opts.UserID,
		ProjectID: opts.ProjectID,
		Types:     opts.Types,
		Limit:     phase1Limit,
		Threshold: threshold * cfg.Phase1ThresholdFactor,
	}

	allResults, err := store.Retrieve(ctx, phase1Opts)
	if err != nil {
		return nil, fmt.Errorf("phase 1 retrieval failed: %w", err)
	}

	if len(allResults) == 0 {
		return []*RetrieveResult{}, nil
	}

	// Build hybrid scorer if hybrid config is provided.
	// Re-scores the cosine-based results from store.Retrieve with BM25 + n-gram.
	var hybridScorer *MemHybridScorer
	if opts.Hybrid != nil {
		docs := make(map[string]string, len(allResults))
		for _, r := range allResults {
			docs[r.Memory.ID] = r.Memory.Content
		}
		hybridScorer = BuildMemHybridScorer(docs, opts.Hybrid)
	}

	// rerankScore applies hybrid fusion to a single result if enabled.
	rerankScore := func(r *RetrieveResult) float32 {
		if hybridScorer == nil {
			return r.Score
		}
		return hybridScorer.FusedScore(r.Memory.ID, r.Score, opts.Query)
	}

	// Apply hybrid reranking to phase 1 results.
	for _, r := range allResults {
		r.Score = rerankScore(r)
	}

	var categories []*RetrieveResult
	var leaves []*RetrieveResult
	for _, r := range allResults {
		if r.Memory.IsCategory {
			categories = append(categories, r)
		} else {
			leaves = append(leaves, r)
		}
	}

	logging.Debugf("HierarchicalRetrieve: Phase 1 found %d categories, %d leaves", len(categories), len(leaves))

	if len(categories) == 0 {
		filtered := filterByThreshold(leaves, threshold)
		if len(filtered) > limit {
			filtered = filtered[:limit]
		}
		return filtered, nil
	}

	alpha := opts.ScorePropAlpha
	pq := &candidateHeap{}
	heap.Init(pq)

	visited := make(map[string]bool)
	collected := make([]*RetrieveResult, 0, limit*2)

	for _, cat := range categories {
		if cat.Score >= threshold*cfg.CategorySeedFactor {
			heap.Push(pq, hierarchicalCandidate{
				memoryID: cat.Memory.ID,
				score:    cat.Score,
				depth:    0,
			})
		}
	}

	for _, leaf := range leaves {
		if leaf.Score >= threshold {
			collected = append(collected, leaf)
			visited[leaf.Memory.ID] = true
		}
	}

	prevTopKIDs := ""
	convergenceRounds := 0

	for pq.Len() > 0 && convergenceRounds < cfg.MaxConvergenceRounds {
		top := heap.Pop(pq).(hierarchicalCandidate)
		if visited[top.memoryID] {
			continue
		}
		visited[top.memoryID] = true

		if top.depth >= opts.MaxDepth {
			continue
		}

		childOpts := RetrieveOptions{
			Query:     opts.Query,
			UserID:    opts.UserID,
			ProjectID: opts.ProjectID,
			Types:     opts.Types,
			Limit:     phase1Limit,
			Threshold: threshold * cfg.ChildThresholdFactor,
		}

		childResults, childErr := store.Retrieve(ctx, childOpts)
		if childErr != nil {
			return nil, fmt.Errorf("child search for category %s failed: %w", top.memoryID, childErr)
		}

		// Build hybrid scorer for child results if needed.
		var childScorer *MemHybridScorer
		if opts.Hybrid != nil {
			childDocs := make(map[string]string, len(childResults))
			for _, cr := range childResults {
				childDocs[cr.Memory.ID] = cr.Memory.Content
			}
			childScorer = BuildMemHybridScorer(childDocs, opts.Hybrid)
		}

		for _, child := range childResults {
			if visited[child.Memory.ID] {
				continue
			}
			if child.Memory.ParentID != top.memoryID {
				continue
			}

			childScore := child.Score
			if childScorer != nil {
				childScore = childScorer.FusedScore(child.Memory.ID, child.Score, opts.Query)
			}
			propagated := PropagateScore(childScore, top.score, alpha)

			if child.Memory.IsCategory {
				heap.Push(pq, hierarchicalCandidate{
					memoryID: child.Memory.ID,
					score:    propagated,
					depth:    top.depth + 1,
				})
			} else if propagated >= threshold {
				child.Score = propagated
				collected = append(collected, child)
				visited[child.Memory.ID] = true
			}
		}

		currentTopKIDs := topKSignature(collected, limit)
		if currentTopKIDs == prevTopKIDs {
			convergenceRounds++
		} else {
			convergenceRounds = 0
			prevTopKIDs = currentTopKIDs
		}
	}

	sort.Slice(collected, func(i, j int) bool {
		return collected[i].Score > collected[j].Score
	})

	if len(collected) > limit {
		collected = collected[:limit]
	}

	// Graph expansion: follow RelatedIDs to discover cross-category memories.
	if opts.FollowLinks {
		embCfg := EmbeddingConfig{Model: EmbeddingModelBERT}
		if opts.LinkEmbeddingConfig != nil {
			embCfg = *opts.LinkEmbeddingConfig
		}
		qEmb, embErr := GenerateEmbedding(opts.Query, embCfg)
		if embErr != nil {
			return nil, fmt.Errorf("expandViaLinks: failed to generate query embedding: %w", embErr)
		}
		collected = expandViaLinks(ctx, store, collected, opts, qEmb)
	}

	logging.Debugf("HierarchicalRetrieve: returning %d results after %d convergence rounds",
		len(collected), convergenceRounds)

	return collected, nil
}

func filterByThreshold(results []*RetrieveResult, threshold float32) []*RetrieveResult {
	filtered := make([]*RetrieveResult, 0, len(results))
	for _, r := range results {
		if r.Score >= threshold {
			filtered = append(filtered, r)
		}
	}
	return filtered
}

func topKSignature(results []*RetrieveResult, k int) string {
	sorted := make([]*RetrieveResult, len(results))
	copy(sorted, results)
	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i].Score > sorted[j].Score
	})
	if len(sorted) > k {
		sorted = sorted[:k]
	}
	sig := ""
	for _, r := range sorted {
		sig += r.Memory.ID + ","
	}
	return sig
}

// BuildGroupFilter constructs a Milvus filter expression that includes
// the user's own memories and group-visible memories from the given groups.
func BuildGroupFilter(userID string, groupIDs []string, includeGroup bool) string {
	if !includeGroup || len(groupIDs) == 0 {
		return fmt.Sprintf("user_id == \"%s\"", userID)
	}

	groupList := ""
	for i, gid := range groupIDs {
		if i > 0 {
			groupList += ", "
		}
		groupList += fmt.Sprintf("\"%s\"", gid)
	}

	return fmt.Sprintf(
		"(user_id == \"%s\") || (group_id in [%s] && visibility in [\"group\", \"public\"])",
		userID, groupList,
	)
}

// BuildCategoryFilter adds is_category filtering on top of an existing filter.
func BuildCategoryFilter(baseFilter string, categoryOnly bool) string {
	if categoryOnly {
		return fmt.Sprintf("(%s) && is_category == true", baseFilter)
	}
	return baseFilter
}

// PropagateScore blends a child score with its parent score.
//
//	result = alpha * childScore + (1 - alpha) * parentScore
func PropagateScore(childScore, parentScore, alpha float32) float32 {
	return alpha*childScore + (1-alpha)*parentScore
}

// expandViaLinks performs graph expansion on the collected results by following
// RelatedIDs. For each result, linked memories are fetched and scored against
// the query using embedding cosine similarity (not hybrid BM25/n-gram).
//
// Hybrid scoring is deliberately NOT applied to linked memories. Link expansion
// discovers cross-domain associations where the linked memory shares zero
// vocabulary with the query — exactly the scenario BM25/n-gram would penalize.
// The referrer's score (which may include hybrid fusion from the main pipeline)
// propagates through linkDecay as the primary relevance signal.
//
// This is what distinguishes RelatedIDs from both pure tree traversal (which
// cannot cross subtree boundaries) and hybrid search (which requires keyword
// or semantic overlap). Neither approach can discover a Finance memory linked
// to a DevOps memory when the query is about Kubernetes.
//
// Research context: GraphRAG systems show +15–23% recall improvement from
// cross-document link traversal (HLG, ICLR 2025; Practical GraphRAG, ACL 2025).
func expandViaLinks(
	ctx context.Context,
	store Store,
	collected []*RetrieveResult,
	opts HierarchicalRetrieveOptions,
	queryEmbedding []float32,
) []*RetrieveResult {
	if !opts.FollowLinks || len(collected) == 0 {
		return collected
	}

	if len(queryEmbedding) == 0 {
		logging.Warnf("expandViaLinks: no query embedding provided, skipping link expansion")
		return collected
	}

	maxDepth := opts.MaxLinkDepth
	if maxDepth <= 0 {
		maxDepth = DefaultMaxLinkDepth
	}

	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = DefaultMemoryConfig().DefaultSimilarityThreshold
	}
	limit := opts.Limit
	if limit <= 0 {
		limit = DefaultHierarchicalLimit
	}

	seen := make(map[string]bool, len(collected))
	for _, r := range collected {
		seen[r.Memory.ID] = true
	}

	frontier := make([]*RetrieveResult, len(collected))
	copy(frontier, collected)

	for hop := 0; hop < maxDepth; hop++ {
		var nextFrontier []*RetrieveResult

		type linkRef struct {
			mem      *Memory
			referrer *RetrieveResult
		}
		var refs []linkRef

		for _, r := range frontier {
			for _, linkedID := range r.Memory.RelatedIDs {
				if seen[linkedID] {
					continue
				}
				seen[linkedID] = true

				linked, err := store.Get(ctx, linkedID)
				if err != nil || linked == nil || linked.IsCategory {
					continue
				}
				refs = append(refs, linkRef{mem: linked, referrer: r})
			}
		}

		if len(refs) == 0 {
			break
		}

		for _, ref := range refs {
			// Cosine similarity only — not hybrid. Link expansion deliberately
			// skips BM25/n-gram because linked memories are discovered by
			// association, not by keyword match. Applying keyword scoring to
			// cross-domain links penalizes the exact scenario links exist to
			// support (e.g., a DevOps memory linked to a Finance memory shares
			// zero vocabulary). The referrer's score (which may include hybrid
			// fusion) already propagates through linkDecay.
			directScore := cosineSimilarity(queryEmbedding, ref.mem.Embedding)

			// Link-propagated score: the referring memory's relevance is the
			// primary signal (the link itself is evidence of association).
			// Direct similarity provides a bonus but is not required — this is
			// the key difference from pure semantic search.
			const linkDecay float32 = 0.8
			const directBonus float32 = 0.2
			blended := ref.referrer.Score*linkDecay + directScore*directBonus

			if blended >= threshold {
				result := &RetrieveResult{Memory: ref.mem, Score: blended}
				collected = append(collected, result)
				nextFrontier = append(nextFrontier, result)
			}
		}

		if len(nextFrontier) == 0 {
			break
		}
		frontier = nextFrontier
	}

	sort.Slice(collected, func(i, j int) bool {
		return collected[i].Score > collected[j].Score
	})

	if len(collected) > limit {
		collected = collected[:limit]
	}

	logging.Debugf("expandViaLinks: %d results after %d hop(s) of link expansion", len(collected), maxDepth)
	return collected
}
