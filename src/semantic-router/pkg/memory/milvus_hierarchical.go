package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"sort"
	"time"

	"github.com/milvus-io/milvus-sdk-go/v2/client"
	"github.com/milvus-io/milvus-sdk-go/v2/entity"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var _ HierarchicalStore = (*MilvusStore)(nil)

// MilvusHierarchicalConfig holds tunable parameters for Milvus hierarchical search.
type MilvusHierarchicalConfig struct {
	// HNSWSearchEf is the ef parameter for HNSW index search (default: 64).
	HNSWSearchEf int

	// MaxCategoriesPerDepth caps how many categories are explored at each depth (default: 10).
	MaxCategoriesPerDepth int

	// CategorySearchTopK is the number of category candidates to fetch in Phase 1 (default: 20).
	CategorySearchTopK int

	// MinFlatTopK is the minimum number of flat candidates to fetch (default: 20).
	MinFlatTopK int

	// MinChildTopK is the minimum number of child candidates per category drill-down (default: 20).
	MinChildTopK int

	// Phase1ThresholdFactor is multiplied with the user threshold for category filtering (default: 0.8).
	Phase1ThresholdFactor float32

	// ChildThresholdFactor is multiplied with the user threshold for category gating (default: 0.7).
	ChildThresholdFactor float32

	// FlatThresholdFactor is multiplied with the user threshold for flat candidate pre-filter (default: 0.7).
	FlatThresholdFactor float32
}

// ApplyDefaults fills zero-valued fields.
func (c *MilvusHierarchicalConfig) ApplyDefaults() {
	if c.HNSWSearchEf <= 0 {
		c.HNSWSearchEf = 64
	}
	if c.MaxCategoriesPerDepth <= 0 {
		c.MaxCategoriesPerDepth = 10
	}
	if c.CategorySearchTopK <= 0 {
		c.CategorySearchTopK = 20
	}
	if c.MinFlatTopK <= 0 {
		c.MinFlatTopK = 20
	}
	if c.MinChildTopK <= 0 {
		c.MinChildTopK = 20
	}
	if c.Phase1ThresholdFactor <= 0 {
		c.Phase1ThresholdFactor = 0.8
	}
	if c.ChildThresholdFactor <= 0 {
		c.ChildThresholdFactor = 0.7
	}
	if c.FlatThresholdFactor <= 0 {
		c.FlatThresholdFactor = 0.7
	}
}

// HierarchicalRetrieve performs group-aware, two-phase hierarchical retrieval in Milvus.
func (m *MilvusStore) HierarchicalRetrieve(ctx context.Context, opts HierarchicalRetrieveOptions) ([]*RetrieveResult, error) {
	return m.HierarchicalRetrieveWithConfig(ctx, opts, MilvusHierarchicalConfig{})
}

// HierarchicalRetrieveWithConfig is the configurable variant.
func (m *MilvusStore) HierarchicalRetrieveWithConfig(ctx context.Context, opts HierarchicalRetrieveOptions, cfg MilvusHierarchicalConfig) ([]*RetrieveResult, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	opts.ApplyDefaults()
	cfg.ApplyDefaults()

	limit := opts.Limit
	if limit <= 0 {
		limit = m.config.DefaultRetrievalLimit
	}
	threshold := opts.Threshold
	if threshold <= 0 {
		threshold = m.config.DefaultSimilarityThreshold
	}

	if opts.Query == "" {
		return nil, fmt.Errorf("query is required")
	}
	if opts.UserID == "" && !opts.IncludeGroupLevel {
		return nil, fmt.Errorf("user id or group ids required")
	}

	logging.Debugf("MilvusStore.HierarchicalRetrieve: query='%s', user_id='%s', groups=%v, limit=%d",
		truncateForLog(opts.Query, 60), opts.UserID, opts.GroupIDs, limit)

	queryEmbedding, err := GenerateEmbedding(opts.Query, m.embeddingConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to generate embedding: %w", err)
	}

	baseFilter := BuildGroupFilter(opts.UserID, opts.GroupIDs, opts.IncludeGroupLevel)

	if len(opts.Types) > 0 {
		typeFilter := "("
		for i, memType := range opts.Types {
			if i > 0 {
				typeFilter += " || "
			}
			typeFilter += fmt.Sprintf("memory_type == \"%s\"", string(memType))
		}
		typeFilter += ")"
		baseFilter = fmt.Sprintf("(%s) && %s", baseFilter, typeFilter)
	}

	searchParam, err := entity.NewIndexHNSWSearchParam(cfg.HNSWSearchEf)
	if err != nil {
		return nil, fmt.Errorf("failed to create search parameters: %w", err)
	}

	outputFields := []string{
		"id", "content", "memory_type", "metadata",
		"group_id", "parent_id", "is_category", "visibility", "abstract",
	}

	// Phase 1: Search category nodes only.
	categoryFilter := BuildCategoryFilter(baseFilter, true)
	phase1TopK := cfg.CategorySearchTopK
	if phase1TopK < limit*4 {
		phase1TopK = limit * 4
	}

	categoryResults, catErr := m.milvusSearch(ctx, categoryFilter, queryEmbedding, outputFields, phase1TopK, searchParam)
	if catErr != nil {
		return nil, fmt.Errorf("category search failed: %w", catErr)
	}

	// Also do a flat search (no category filter) to capture leaf memories.
	flatTopK := limit * 4
	if flatTopK < cfg.MinFlatTopK {
		flatTopK = cfg.MinFlatTopK
	}
	flatResults, flatErr := m.milvusSearch(ctx, baseFilter, queryEmbedding, outputFields, flatTopK, searchParam)
	if flatErr != nil {
		return nil, fmt.Errorf("flat search failed: %w", flatErr)
	}

	categories := m.parseHierarchicalResults(categoryResults, threshold*cfg.Phase1ThresholdFactor)
	allFlat := m.parseHierarchicalResults(flatResults, threshold*cfg.FlatThresholdFactor)

	logging.Debugf("MilvusStore.HierarchicalRetrieve: Phase 1 found %d categories, %d flat candidates",
		len(categories), len(allFlat))

	if len(categories) == 0 {
		results := filterByThreshold(allFlat, threshold)
		sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
		if len(results) > limit {
			results = results[:limit]
		}
		return results, nil
	}

	// Phase 2: Drill down.
	alpha := opts.ScorePropAlpha
	collected := make(map[string]*RetrieveResult)

	for _, r := range allFlat {
		if !r.Memory.IsCategory && r.Score >= threshold {
			collected[r.Memory.ID] = r
		}
	}

	sort.Slice(categories, func(i, j int) bool { return categories[i].Score > categories[j].Score })

	maxCats := cfg.MaxCategoriesPerDepth
	if len(categories) < maxCats {
		maxCats = len(categories)
	}

	for depth := 0; depth < opts.MaxDepth; depth++ {
		nextCategories := make([]*RetrieveResult, 0)

		for ci := 0; ci < maxCats; ci++ {
			cat := categories[ci]
			if cat.Score < threshold*cfg.ChildThresholdFactor {
				continue
			}

			childFilter := fmt.Sprintf("(%s) && parent_id == \"%s\"", baseFilter, cat.Memory.ID)
			childTopK := limit * 3
			if childTopK < cfg.MinChildTopK {
				childTopK = cfg.MinChildTopK
			}

			childSearchResults, childErr := m.milvusSearch(ctx, childFilter, queryEmbedding, outputFields, childTopK, searchParam)
			if childErr != nil {
				return nil, fmt.Errorf("child search for category %s failed: %w", cat.Memory.ID, childErr)
			}

			children := m.parseHierarchicalResults(childSearchResults, 0)
			for _, child := range children {
				if _, exists := collected[child.Memory.ID]; exists {
					continue
				}

				propagated := PropagateScore(child.Score, cat.Score, alpha)

				if child.Memory.IsCategory {
					child.Score = propagated
					nextCategories = append(nextCategories, child)
				} else if propagated >= threshold {
					child.Score = propagated
					collected[child.Memory.ID] = child
				}
			}
		}

		if len(nextCategories) == 0 {
			break
		}
		categories = nextCategories
		sort.Slice(categories, func(i, j int) bool { return categories[i].Score > categories[j].Score })
		maxCats = cfg.MaxCategoriesPerDepth
		if len(categories) < maxCats {
			maxCats = len(categories)
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

	if opts.EnableRelations && len(results) > 0 {
		m.populateRelationsFromMetadata(results, opts.MaxRelationsPerHit)
	}

	if len(results) > 0 {
		ids := make([]string, len(results))
		for i, r := range results {
			ids[i] = r.Memory.ID
		}
		go m.recordRetrievalBatch(ids)
	}

	logging.Debugf("MilvusStore.HierarchicalRetrieve: returning %d results", len(results))
	return results, nil
}

func (m *MilvusStore) milvusSearch(
	ctx context.Context,
	filterExpr string,
	embedding []float32,
	outputFields []string,
	topK int,
	searchParam entity.SearchParam,
) ([]client.SearchResult, error) {
	var searchResult []client.SearchResult
	err := m.retryWithBackoff(ctx, func() error {
		var retryErr error
		searchResult, retryErr = m.client.Search(
			ctx,
			m.collectionName,
			[]string{},
			filterExpr,
			outputFields,
			[]entity.Vector{entity.FloatVector(embedding)},
			"embedding",
			entity.COSINE,
			topK,
			searchParam,
		)
		return retryErr
	})
	return searchResult, err
}

func (m *MilvusStore) parseHierarchicalResults(searchResult []client.SearchResult, minScore float32) []*RetrieveResult {
	if len(searchResult) == 0 || searchResult[0].ResultCount == 0 {
		return nil
	}

	scores := searchResult[0].Scores
	fields := searchResult[0].Fields

	fieldMap := make(map[string]int)
	for i, f := range fields {
		fieldMap[f.Name()] = i
	}

	results := make([]*RetrieveResult, 0, len(scores))
	for i := 0; i < len(scores); i++ {
		if scores[i] < minScore {
			continue
		}

		mem := &Memory{}

		if idx, ok := fieldMap["id"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				mem.ID, _ = col.ValueByIdx(i)
			}
		}
		if idx, ok := fieldMap["content"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				mem.Content, _ = col.ValueByIdx(i)
			}
		}
		if idx, ok := fieldMap["memory_type"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				val, _ := col.ValueByIdx(i)
				mem.Type = MemoryType(val)
			}
		}
		if idx, ok := fieldMap["group_id"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				mem.GroupID, _ = col.ValueByIdx(i)
			}
		}
		if idx, ok := fieldMap["parent_id"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				mem.ParentID, _ = col.ValueByIdx(i)
			}
		}
		if idx, ok := fieldMap["is_category"]; ok {
			if col, ok := fields[idx].(*entity.ColumnBool); ok && col.Len() > i {
				mem.IsCategory, _ = col.ValueByIdx(i)
			}
		}
		if idx, ok := fieldMap["visibility"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				val, _ := col.ValueByIdx(i)
				mem.Visibility = MemoryVisibility(val)
			}
		}
		if idx, ok := fieldMap["abstract"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				mem.Abstract, _ = col.ValueByIdx(i)
			}
		}

		if idx, ok := fieldMap["metadata"]; ok {
			if col, ok := fields[idx].(*entity.ColumnVarChar); ok && col.Len() > i {
				metaStr, _ := col.ValueByIdx(i)
				if metaStr != "" {
					var meta map[string]interface{}
					if json.Unmarshal([]byte(metaStr), &meta) == nil {
						if uid, ok := meta["user_id"].(string); ok {
							mem.UserID = uid
						}
						if pid, ok := meta["project_id"].(string); ok {
							mem.ProjectID = pid
						}
						if src, ok := meta["source"].(string); ok {
							mem.Source = src
						}
						if imp, ok := meta["importance"].(float64); ok {
							mem.Importance = float32(imp)
						}
						if rels, ok := meta["related_ids"].([]interface{}); ok {
							for _, r := range rels {
								if rid, ok := r.(string); ok {
									mem.RelatedIDs = append(mem.RelatedIDs, rid)
								}
							}
						}
						if overview, ok := meta["overview"].(string); ok {
							mem.Overview = overview
						}
					}
				}
			}
		}

		if mem.ID == "" {
			continue
		}

		results = append(results, &RetrieveResult{
			Memory: mem,
			Score:  scores[i],
		})
	}

	return results
}

// StoreRelation persists a memory relation by storing it in the metadata of both endpoints.
// Returns an error if either endpoint update fails.
func (m *MilvusStore) StoreRelation(ctx context.Context, rel MemoryRelation) error {
	if !m.enabled {
		return fmt.Errorf("milvus store is not enabled")
	}

	fromErr := m.appendRelatedID(ctx, rel.FromID, rel.ToID)
	toErr := m.appendRelatedID(ctx, rel.ToID, rel.FromID)

	if fromErr != nil && toErr != nil {
		return fmt.Errorf("failed to update both endpoints: from=%s: %w, to=%s: %w", rel.FromID, fromErr, rel.ToID, toErr)
	}
	if fromErr != nil {
		return fmt.Errorf("failed to update from=%s: %w", rel.FromID, fromErr)
	}
	if toErr != nil {
		return fmt.Errorf("failed to update to=%s: %w", rel.ToID, toErr)
	}
	return nil
}

func (m *MilvusStore) appendRelatedID(ctx context.Context, memoryID, targetID string) error {
	mem, err := m.Get(ctx, memoryID)
	if err != nil {
		return err
	}

	for _, rid := range mem.RelatedIDs {
		if rid == targetID {
			return nil
		}
	}

	mem.RelatedIDs = append(mem.RelatedIDs, targetID)
	mem.UpdatedAt = time.Now()

	return m.upsert(ctx, mem)
}

// GetRelations returns relations originating from the given memory ID
// by reading RelatedIDs from the memory's metadata.
func (m *MilvusStore) GetRelations(ctx context.Context, memoryID string, limit int) ([]MemoryRelation, error) {
	if !m.enabled {
		return nil, fmt.Errorf("milvus store is not enabled")
	}

	mem, err := m.Get(ctx, memoryID)
	if err != nil {
		return nil, err
	}

	if len(mem.RelatedIDs) == 0 {
		return nil, nil
	}

	rels := make([]MemoryRelation, 0, len(mem.RelatedIDs))
	for _, rid := range mem.RelatedIDs {
		rels = append(rels, MemoryRelation{
			FromID: memoryID,
			ToID:   rid,
		})
		if limit > 0 && len(rels) >= limit {
			break
		}
	}

	return rels, nil
}

func (m *MilvusStore) populateRelationsFromMetadata(results []*RetrieveResult, maxPerHit int) {
	if maxPerHit <= 0 {
		maxPerHit = 5
	}

	for _, r := range results {
		if len(r.Memory.RelatedIDs) == 0 {
			continue
		}

		count := len(r.Memory.RelatedIDs)
		if count > maxPerHit {
			count = maxPerHit
		}

		for _, rid := range r.Memory.RelatedIDs[:count] {
			related := &RelatedMemory{ID: rid}
			r.Related = append(r.Related, related)
		}
	}
}
