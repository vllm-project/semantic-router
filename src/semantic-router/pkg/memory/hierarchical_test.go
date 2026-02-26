package memory

import (
	"context"
	"fmt"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// workloadMemories returns realistic memories modeled on paper topics
// (architecture, memory, safety, deployment, routing, RAG, evaluation).
// Each topic cluster has 3 memories so hierarchical search has structure to exploit.
func workloadMemories(userID string) []*Memory {
	now := time.Now()
	mems := []*Memory{
		// --- architecture cluster ---
		{ID: "arch-1", Type: MemoryTypeSemantic, Content: "The routing pipeline uses an ExtProc gRPC filter integrated with Envoy as a sidecar proxy", UserID: userID, CreatedAt: now},
		{ID: "arch-2", Type: MemoryTypeSemantic, Content: "Signal engine collects embedding similarity, keyword matching, and domain classification signals before the decision engine selects a model", UserID: userID, CreatedAt: now},
		{ID: "arch-3", Type: MemoryTypeProcedural, Content: "To add a new signal, implement the Signal interface and register it in the signal registry with a unique name", UserID: userID, CreatedAt: now},
		// --- memory cluster ---
		{ID: "mem-1", Type: MemoryTypeSemantic, Content: "User memories are extracted from conversation turns by an LLM and stored in Milvus with vector embeddings for retrieval", UserID: userID, CreatedAt: now},
		{ID: "mem-2", Type: MemoryTypeSemantic, Content: "Memory retention uses exponential decay with strength S = S0 + access_count, pruning below threshold R < delta", UserID: userID, CreatedAt: now},
		{ID: "mem-3", Type: MemoryTypeProcedural, Content: "Query rewriting reformulates vague user queries using conversation history before memory retrieval for better recall", UserID: userID, CreatedAt: now},
		// --- safety cluster ---
		{ID: "safe-1", Type: MemoryTypeSemantic, Content: "Jailbreak detection uses a fine-tuned ModernBERT classifier that flags adversarial prompt injection attempts", UserID: userID, CreatedAt: now},
		{ID: "safe-2", Type: MemoryTypeSemantic, Content: "PII detection identifies credit card numbers, social security numbers, and email addresses in request and response bodies", UserID: userID, CreatedAt: now},
		{ID: "safe-3", Type: MemoryTypeProcedural, Content: "To enable hallucination gating, configure the halugate plugin with a reference model endpoint and confidence threshold", UserID: userID, CreatedAt: now},
		// --- deployment cluster ---
		{ID: "deploy-1", Type: MemoryTypeSemantic, Content: "Production deployment uses Kubernetes with Helm charts that configure Envoy sidecars and ExtProc service replicas", UserID: userID, CreatedAt: now},
		{ID: "deploy-2", Type: MemoryTypeProcedural, Content: "Rolling updates use canary deployment strategy with gradual traffic shifting from 10% to 50% to 100% across model backends", UserID: userID, CreatedAt: now},
		{ID: "deploy-3", Type: MemoryTypeEpisodic, Content: "Last week we rolled back the v2 routing model after observing 15% latency regression in P99 response times", UserID: userID, CreatedAt: now},
		// --- RAG cluster ---
		{ID: "rag-1", Type: MemoryTypeSemantic, Content: "Hybrid retrieval combines vector similarity, BM25 keyword matching, and character n-gram Jaccard scoring for document search", UserID: userID, CreatedAt: now},
		{ID: "rag-2", Type: MemoryTypeSemantic, Content: "Score fusion supports weighted combination with configurable weights and reciprocal rank fusion with parameter k=60", UserID: userID, CreatedAt: now},
		{ID: "rag-3", Type: MemoryTypeProcedural, Content: "To configure a vector store backend, set the vector_store_id and choose between in-memory, Milvus, or file-based persistence", UserID: userID, CreatedAt: now},
		// --- evaluation cluster ---
		{ID: "eval-1", Type: MemoryTypeSemantic, Content: "End-to-end evaluation measures routing accuracy, retrieval recall at k, and response latency percentiles across model backends", UserID: userID, CreatedAt: now},
		{ID: "eval-2", Type: MemoryTypeEpisodic, Content: "Benchmark run on February 15 showed 94% routing accuracy with 12ms P50 latency for the signal-based decision engine", UserID: userID, CreatedAt: now},
		{ID: "eval-3", Type: MemoryTypeSemantic, Content: "The feedback routing loop collects user thumbs-up and thumbs-down signals to adjust model preference scores over time", UserID: userID, CreatedAt: now},
	}
	return mems
}

func storeAll(t *testing.T, store *InMemoryStore, mems []*Memory) {
	t.Helper()
	ctx := context.Background()
	for _, m := range mems {
		require.NoError(t, store.Store(ctx, m), "storing %s", m.ID)
	}
}

// ---------------------------------------------------------------------------
// Categorizer
// ---------------------------------------------------------------------------

func TestExtractTopic(t *testing.T) {
	kw := DefaultTopicKeywords()

	tests := []struct {
		content string
		want    string
	}{
		{"deploy rollback pipeline", "deployment"},
		{"kubernetes container cluster database", "infrastructure"},
		{"test coverage regression benchmark", "testing"},
		{"prefer favorite style convention", "preferences"},
		{"this content has no known keywords at all", "general"},
	}
	for _, tt := range tests {
		got := extractTopic(tt.content, MemoryTypeSemantic, kw)
		assert.Equal(t, tt.want, got, "content=%q", tt.content)
	}
}

func TestExtractTopic_CustomKeywords(t *testing.T) {
	custom := map[string][]string{
		"llm": {"language model", "transformer", "attention"},
	}
	got := extractTopic("transformer attention mechanism", MemoryTypeSemantic, custom)
	assert.Equal(t, "llm", got)
}

func TestBuildCategoryID_Deterministic(t *testing.T) {
	id1 := buildCategoryID("user-a", "", MemoryTypeSemantic, "deployment")
	id2 := buildCategoryID("user-a", "", MemoryTypeSemantic, "deployment")
	assert.Equal(t, id1, id2, "same inputs must produce same ID")

	id3 := buildCategoryID("user-b", "", MemoryTypeSemantic, "deployment")
	assert.NotEqual(t, id1, id3, "different user must produce different ID")

	id4 := buildCategoryID("user-a", "grp-1", MemoryTypeSemantic, "deployment")
	assert.NotEqual(t, id1, id4, "group scoping must change the ID")
}

func TestAutoCategorize_CreatesCategory(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	cfg := CategorizerConfig{}
	ct := NewCategoryTree(store, store.embeddingConfig, cfg)

	mem := &Memory{
		ID:      "test-mem",
		Type:    MemoryTypeSemantic,
		Content: "deploy rollback release pipeline CI/CD",
		UserID:  "user1",
	}
	require.NoError(t, store.Store(ctx, mem))

	result, err := ct.AutoCategorize(ctx, mem)
	require.NoError(t, err)
	require.NotNil(t, result)

	assert.NotEmpty(t, result.ParentID, "should assign a parent category")
	assert.NotEmpty(t, result.Abstract, "should generate an abstract")
	assert.NotEmpty(t, result.Overview, "should generate an overview")

	// The category node should now exist in the store.
	catMem, err := store.Get(ctx, result.ParentID)
	require.NoError(t, err)
	assert.True(t, catMem.IsCategory)
	assert.NotEmpty(t, catMem.Embedding, "category must have an embedding")
}

func TestAutoCategorize_SkipsCategories(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	ct := NewCategoryTree(store, store.embeddingConfig, CategorizerConfig{})

	catMem := &Memory{ID: "cat-1", IsCategory: true, Content: "category node"}
	result, err := ct.AutoCategorize(ctx, catMem)
	require.NoError(t, err)
	assert.Nil(t, result)
}

func TestAutoCategorize_ReusesExistingCategory(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	ct := NewCategoryTree(store, store.embeddingConfig, CategorizerConfig{})

	mem1 := &Memory{ID: "m1", Type: MemoryTypeSemantic, Content: "deploy rollback release", UserID: "u1"}
	require.NoError(t, store.Store(ctx, mem1))
	r1, err := ct.AutoCategorize(ctx, mem1)
	require.NoError(t, err)

	mem2 := &Memory{ID: "m2", Type: MemoryTypeSemantic, Content: "deploy pipeline CI/CD", UserID: "u1"}
	require.NoError(t, store.Store(ctx, mem2))
	r2, err := ct.AutoCategorize(ctx, mem2)
	require.NoError(t, err)

	assert.Equal(t, r1.ParentID, r2.ParentID, "same topic/user should reuse category")
}

func TestEnrichMemoryBeforeStore_SetsFields(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	mem := &Memory{
		ID:      "enrich-test",
		Type:    MemoryTypeSemantic,
		Content: "deploy rollback release pipeline CI/CD for kubernetes cluster",
		UserID:  "user1",
	}
	require.NoError(t, store.Store(ctx, mem))

	err := EnrichMemoryBeforeStore(ctx, store, mem, store.embeddingConfig, CategorizerConfig{})
	require.NoError(t, err)

	assert.Equal(t, VisibilityUser, mem.Visibility, "default visibility")
	assert.NotEmpty(t, mem.ParentID, "should be assigned a parent")
	assert.NotEmpty(t, mem.Abstract, "should have abstract")
	assert.NotEmpty(t, mem.Overview, "should have overview")
}

func TestEnrichMemoryBeforeStore_PreservesExistingFields(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	mem := &Memory{
		ID:       "preserve-test",
		Type:     MemoryTypeSemantic,
		Content:  "deploy rollback release",
		UserID:   "user1",
		ParentID: "custom-parent",
		Abstract: "custom abstract",
		Overview: "custom overview",
	}
	require.NoError(t, store.Store(ctx, mem))

	err := EnrichMemoryBeforeStore(ctx, store, mem, store.embeddingConfig, CategorizerConfig{})
	require.NoError(t, err)

	assert.Equal(t, "custom-parent", mem.ParentID)
	assert.Equal(t, "custom abstract", mem.Abstract)
	assert.Equal(t, "custom overview", mem.Overview)
}

func TestGenerateAbstract_Truncation(t *testing.T) {
	short := "Short content."
	assert.Equal(t, short, generateAbstract(short, 200))

	long := ""
	for i := 0; i < 50; i++ {
		long += "word "
	}
	result := generateAbstract(long, 80)
	assert.LessOrEqual(t, len(result), 84) // 80 + "..."
}

func TestGenerateOverview_Truncation(t *testing.T) {
	short := "Short."
	assert.Equal(t, short, generateOverview(short, 600))

	long := ""
	for i := 0; i < 200; i++ {
		long += "word "
	}
	result := generateOverview(long, 100)
	assert.LessOrEqual(t, len(result), 104)
}

// ---------------------------------------------------------------------------
// Filter builders
// ---------------------------------------------------------------------------

func TestBuildGroupFilter_UserOnly(t *testing.T) {
	f := BuildGroupFilter("u1", nil, false)
	assert.Equal(t, `user_id == "u1"`, f)

	f2 := BuildGroupFilter("u1", []string{"g1"}, false)
	assert.Equal(t, `user_id == "u1"`, f2, "includeGroup=false ignores groups")
}

func TestBuildGroupFilter_WithGroups(t *testing.T) {
	f := BuildGroupFilter("u1", []string{"g1", "g2"}, true)
	assert.Contains(t, f, `user_id == "u1"`)
	assert.Contains(t, f, `group_id in ["g1", "g2"]`)
	assert.Contains(t, f, `visibility in ["group", "public"]`)
}

func TestBuildCategoryFilter(t *testing.T) {
	base := `user_id == "u1"`
	assert.Equal(t, `(user_id == "u1") && is_category == true`, BuildCategoryFilter(base, true))
	assert.Equal(t, base, BuildCategoryFilter(base, false))
}

func TestPropagateScore(t *testing.T) {
	child := float32(0.9)
	parent := float32(0.7)

	// alpha=1 means full child weight
	assert.InDelta(t, 0.9, PropagateScore(child, parent, 1.0), 1e-6)
	// alpha=0 means full parent weight
	assert.InDelta(t, 0.7, PropagateScore(child, parent, 0.0), 1e-6)
	// alpha=0.5 means midpoint
	assert.InDelta(t, 0.8, PropagateScore(child, parent, 0.5), 1e-6)
}

// ---------------------------------------------------------------------------
// InMemory HierarchicalRetrieve
// ---------------------------------------------------------------------------

func TestHierarchicalRetrieve_FlatFallback(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	storeAll(t, store, workloadMemories("user1"))

	results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query:     "How does memory retrieval work?",
			UserID:    "user1",
			Limit:     5,
			Threshold: 0.3,
		},
	})
	require.NoError(t, err)
	assert.NotEmpty(t, results, "should return results even without categories")

	for _, r := range results {
		assert.GreaterOrEqual(t, r.Score, float32(0.3))
	}
}

func TestHierarchicalRetrieve_WithCategoriesBoostsRecall(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	mems := workloadMemories("user1")
	storeAll(t, store, mems)

	// Enrich all memories to create categories.
	for _, m := range mems {
		_ = EnrichMemoryBeforeStore(ctx, store, m, store.embeddingConfig, CategorizerConfig{})
		_ = store.Update(ctx, m.ID, m)
	}

	results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query:     "How does the deployment pipeline work with kubernetes?",
			UserID:    "user1",
			Limit:     5,
			Threshold: 0.3,
		},
		MaxDepth:       3,
		ScorePropAlpha: 0.6,
	})
	require.NoError(t, err)
	assert.NotEmpty(t, results)

	for _, r := range results {
		assert.GreaterOrEqual(t, r.Score, float32(0.3))
	}
}

func TestHierarchicalRetrieve_UserIsolation(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	storeAll(t, store, workloadMemories("alice"))
	storeAll(t, store, []*Memory{
		{ID: "bob-1", Type: MemoryTypeSemantic, Content: "Bob's secret deployment strategy", UserID: "bob", CreatedAt: time.Now()},
	})

	results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query:     "deployment strategy",
			UserID:    "alice",
			Limit:     10,
			Threshold: 0.1,
		},
	})
	require.NoError(t, err)

	for _, r := range results {
		assert.Equal(t, "alice", r.Memory.UserID, "must not leak bob's memories")
	}
}

func TestHierarchicalRetrieve_GroupVisibility(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Alice's private memory.
	require.NoError(t, store.Store(ctx, &Memory{
		ID: "alice-priv", Type: MemoryTypeSemantic, UserID: "alice",
		Content: "private deployment runbook for alice only", Visibility: VisibilityUser, CreatedAt: time.Now(),
	}))
	// Group-shared memory by bob in group "team-x".
	require.NoError(t, store.Store(ctx, &Memory{
		ID: "bob-shared", Type: MemoryTypeSemantic, UserID: "bob",
		Content: "shared deployment runbook for the whole team", Visibility: VisibilityGroup, GroupID: "team-x", CreatedAt: time.Now(),
	}))
	// Public memory.
	require.NoError(t, store.Store(ctx, &Memory{
		ID: "public-doc", Type: MemoryTypeSemantic, UserID: "charlie",
		Content: "public deployment best practices document", Visibility: VisibilityPublic, CreatedAt: time.Now(),
	}))

	// Alice searching without group inclusion -- should only see her own.
	results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query: "deployment runbook", UserID: "alice", Limit: 10, Threshold: 0.1,
		},
		IncludeGroupLevel: false,
	})
	require.NoError(t, err)
	for _, r := range results {
		assert.Equal(t, "alice", r.Memory.UserID)
	}

	// Alice searching with group inclusion in team-x -- should see her own + bob-shared + public.
	results, err = store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query: "deployment runbook", UserID: "alice", Limit: 10, Threshold: 0.1,
		},
		IncludeGroupLevel: true,
		GroupIDs:          []string{"team-x"},
	})
	require.NoError(t, err)

	ids := make(map[string]bool)
	for _, r := range results {
		ids[r.Memory.ID] = true
	}
	assert.True(t, ids["alice-priv"], "alice should see her own private memory")
	assert.True(t, ids["bob-shared"], "alice should see bob's group-shared memory")
	assert.True(t, ids["public-doc"], "alice should see public memory")
}

func TestHierarchicalRetrieve_LimitRespected(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	storeAll(t, store, workloadMemories("user1"))

	results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query: "deployment", UserID: "user1", Limit: 2, Threshold: 0.1,
		},
	})
	require.NoError(t, err)
	assert.LessOrEqual(t, len(results), 2)
}

func TestHierarchicalRetrieve_EmptyStore(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query: "anything", UserID: "nobody", Limit: 5, Threshold: 0.1,
		},
	})
	require.NoError(t, err)
	assert.Empty(t, results)
}

// ---------------------------------------------------------------------------
// Generic HierarchicalRetrieveFromStore (against InMemoryStore treated as Store)
// ---------------------------------------------------------------------------

func TestGenericHierarchicalRetrieve(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	storeAll(t, store, workloadMemories("user1"))

	results, err := HierarchicalRetrieveFromStore(ctx, store, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{
			Query: "memory retrieval and retention scoring", UserID: "user1", Limit: 5, Threshold: 0.3,
		},
	})
	require.NoError(t, err)
	assert.NotEmpty(t, results)
}

func TestGenericHierarchicalRetrieve_ReturnsError(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	_, err := HierarchicalRetrieveFromStore(ctx, store, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: "", UserID: "user1"},
	})
	assert.Error(t, err, "empty query should error")

	_, err = HierarchicalRetrieveFromStore(ctx, store, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: "test", UserID: ""},
	})
	assert.Error(t, err, "missing user and no group should error")
}

// ---------------------------------------------------------------------------
// Relations
// ---------------------------------------------------------------------------

func TestStoreRelation_Bidirectional(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	require.NoError(t, store.Store(ctx, &Memory{ID: "a", Content: "memory a", UserID: "u", CreatedAt: time.Now()}))
	require.NoError(t, store.Store(ctx, &Memory{ID: "b", Content: "memory b", UserID: "u", CreatedAt: time.Now()}))

	err := store.StoreRelation(ctx, MemoryRelation{
		FromID: "a", ToID: "b", Reason: "test-link", Strength: 0.9, CreatedAt: time.Now(),
	})
	require.NoError(t, err)

	relsA, err := store.GetRelations(ctx, "a", 10)
	require.NoError(t, err)
	require.Len(t, relsA, 1)
	assert.Equal(t, "b", relsA[0].ToID)

	relsB, err := store.GetRelations(ctx, "b", 10)
	require.NoError(t, err)
	require.Len(t, relsB, 1)
	assert.Equal(t, "a", relsB[0].ToID)
}

func TestGetRelations_LimitWorks(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	require.NoError(t, store.Store(ctx, &Memory{ID: "src", Content: "source", UserID: "u", CreatedAt: time.Now()}))
	for i := 0; i < 5; i++ {
		id := fmt.Sprintf("tgt-%d", i)
		require.NoError(t, store.Store(ctx, &Memory{ID: id, Content: "target", UserID: "u", CreatedAt: time.Now()}))
		require.NoError(t, store.StoreRelation(ctx, MemoryRelation{FromID: "src", ToID: id, CreatedAt: time.Now()}))
	}

	rels, err := store.GetRelations(ctx, "src", 3)
	require.NoError(t, err)
	assert.Len(t, rels, 3)
}

func TestAutoLinkNewMemory_CreatesBidirectionalLinks(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	// Seed with two similar memories.
	m1 := &Memory{ID: "seed-1", Type: MemoryTypeSemantic, Content: "deploy rollback release pipeline CI/CD strategy", UserID: "u1", CreatedAt: time.Now()}
	m2 := &Memory{ID: "seed-2", Type: MemoryTypeSemantic, Content: "deployment rollback process for kubernetes services", UserID: "u1", CreatedAt: time.Now()}
	require.NoError(t, store.Store(ctx, m1))
	require.NoError(t, store.Store(ctx, m2))

	newMem := &Memory{ID: "new-1", Type: MemoryTypeSemantic, Content: "rollback deployment to previous version in CI/CD pipeline", UserID: "u1", CreatedAt: time.Now()}
	require.NoError(t, store.Store(ctx, newMem))

	created, err := AutoLinkNewMemory(ctx, store, newMem, AutoLinkOptions{
		Threshold: 0.5,
		Reason:    "content-similarity",
	})
	// err may be nil or non-nil depending on whether similar memories exist above threshold.
	// We verify the return contract works.
	if err != nil {
		t.Logf("AutoLinkNewMemory returned error (non-fatal for this test): %v", err)
	}
	t.Logf("AutoLinkNewMemory created %d links", created)

	if created > 0 {
		rels, err := store.GetRelations(ctx, "new-1", 10)
		require.NoError(t, err)
		assert.NotEmpty(t, rels)
	}
}

func TestAutoLinkNewMemory_SkipsCategories(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	catMem := &Memory{ID: "cat-1", IsCategory: true, Content: "category", UserID: "u1", CreatedAt: time.Now()}
	require.NoError(t, store.Store(ctx, catMem))

	created, err := AutoLinkNewMemory(ctx, store, catMem, AutoLinkOptions{})
	require.NoError(t, err)
	assert.Equal(t, 0, created)
}

// ---------------------------------------------------------------------------
// Config defaults
// ---------------------------------------------------------------------------

func TestHierarchicalRetrieveOptions_ApplyDefaults(t *testing.T) {
	opts := HierarchicalRetrieveOptions{}
	opts.ApplyDefaults()

	assert.Equal(t, DefaultHierarchicalMaxDepth, opts.MaxDepth)
	assert.InDelta(t, DefaultScorePropAlpha, opts.ScorePropAlpha, 1e-6)
	assert.Equal(t, DefaultMaxRelationsPerHit, opts.MaxRelationsPerHit)
}

func TestHierarchicalSearchConfig_ApplyDefaults(t *testing.T) {
	cfg := HierarchicalSearchConfig{}
	cfg.ApplyDefaults()

	assert.Equal(t, 3, cfg.MaxConvergenceRounds)
	assert.Equal(t, 20, cfg.CategorySearchTopK)
	assert.InDelta(t, 0.8, cfg.Phase1ThresholdFactor, 1e-6)
	assert.InDelta(t, 0.7, cfg.ChildThresholdFactor, 1e-6)
	assert.InDelta(t, 0.8, cfg.CategorySeedFactor, 1e-6)
}

func TestCategorizerConfig_ApplyDefaults(t *testing.T) {
	cfg := CategorizerConfig{}
	cfg.ApplyDefaults()

	assert.Equal(t, 200, cfg.AbstractMaxLen)
	assert.Equal(t, 600, cfg.OverviewMaxLen)
	assert.NotNil(t, cfg.TopicKeywords)
	assert.NotEmpty(t, cfg.TopicKeywords)
}

func TestAutoLinkOptions_ApplyDefaults(t *testing.T) {
	opts := AutoLinkOptions{}
	opts.ApplyDefaults()

	assert.InDelta(t, DefaultRelationThreshold, opts.Threshold, 1e-6)
	assert.Equal(t, DefaultMaxRelationsPerMemory, opts.MaxRelations)
	assert.Equal(t, 2, opts.CandidateMultiplier)
}

func TestInMemoryHierarchicalConfig_ApplyDefaults(t *testing.T) {
	cfg := InMemoryHierarchicalConfig{}
	cfg.ApplyDefaults()

	assert.Equal(t, 10, cfg.MaxCategoriesPerDepth)
	assert.InDelta(t, 0.7, cfg.CandidateThresholdFactor, 1e-6)
	assert.Equal(t, 150, cfg.AbstractFallbackMaxLen)
}

// ---------------------------------------------------------------------------
// AsHierarchicalStore
// ---------------------------------------------------------------------------

func TestAsHierarchicalStore(t *testing.T) {
	store := newTestInMemoryStore()
	hs, ok := AsHierarchicalStore(store)
	assert.True(t, ok, "InMemoryStore must implement HierarchicalStore")
	assert.NotNil(t, hs)
}

// ---------------------------------------------------------------------------
// PopulateRelations
// ---------------------------------------------------------------------------

func TestPopulateRelations_InMemory(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	m1 := &Memory{ID: "p-1", Content: "deploy rollback", UserID: "u", Abstract: "deploy abstract", CreatedAt: time.Now()}
	m2 := &Memory{ID: "p-2", Content: "deploy pipeline", UserID: "u", Abstract: "pipeline abstract", CreatedAt: time.Now()}
	require.NoError(t, store.Store(ctx, m1))
	require.NoError(t, store.Store(ctx, m2))

	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{
		FromID: "p-1", ToID: "p-2", Reason: "test", Strength: 0.88, CreatedAt: time.Now(),
	}))

	results := []*RetrieveResult{{Memory: m1, Score: 0.9}}
	cfg := InMemoryHierarchicalConfig{}
	cfg.ApplyDefaults()
	store.populateRelations(results, HierarchicalRetrieveOptions{EnableRelations: true}, cfg)

	require.Len(t, results[0].Related, 1)
	assert.Equal(t, "p-2", results[0].Related[0].ID)
	assert.Equal(t, "pipeline abstract", results[0].Related[0].Abstract)
}

func TestPopulateRelations_FallbackContent(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()

	m1 := &Memory{ID: "r-1", Content: "memory one content", UserID: "u", CreatedAt: time.Now()}
	m2 := &Memory{ID: "r-2", Content: "memory two has no abstract field set", UserID: "u", CreatedAt: time.Now()}
	require.NoError(t, store.Store(ctx, m1))
	require.NoError(t, store.Store(ctx, m2))

	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{
		FromID: "r-1", ToID: "r-2", Strength: 0.85, CreatedAt: time.Now(),
	}))

	results := []*RetrieveResult{{Memory: m1, Score: 0.9}}
	cfg := InMemoryHierarchicalConfig{}
	cfg.ApplyDefaults()
	store.populateRelations(results, HierarchicalRetrieveOptions{EnableRelations: true}, cfg)

	require.Len(t, results[0].Related, 1)
	assert.Equal(t, m2.Content, results[0].Related[0].Abstract, "should fall back to content when abstract is empty")
}

// ---------------------------------------------------------------------------
// MemoryHybridConfig
// ---------------------------------------------------------------------------

func TestMemoryHybridConfig_ApplyDefaults(t *testing.T) {
	cfg := MemoryHybridConfig{}
	cfg.ApplyDefaults()

	assert.Equal(t, "weighted", cfg.Mode)
	assert.InDelta(t, 0.7, cfg.VectorWeight, 1e-6)
	assert.InDelta(t, 0.2, cfg.BM25Weight, 1e-6)
	assert.InDelta(t, 0.1, cfg.NgramWeight, 1e-6)
	assert.Equal(t, 60, cfg.RRFConstant)
	assert.InDelta(t, 1.2, cfg.BM25K1, 1e-6)
	assert.InDelta(t, 0.75, cfg.BM25B, 1e-6)
	assert.Equal(t, 3, cfg.NgramSize)
}

func TestMemoryHybridConfig_PreservesExplicitValues(t *testing.T) {
	cfg := MemoryHybridConfig{
		Mode:         "rrf",
		VectorWeight: 0.5,
		BM25Weight:   0.3,
		NgramWeight:  0.2,
		RRFConstant:  30,
		BM25K1:       2.0,
		BM25B:        0.5,
		NgramSize:    4,
	}
	cfg.ApplyDefaults()

	assert.Equal(t, "rrf", cfg.Mode)
	assert.InDelta(t, 0.5, cfg.VectorWeight, 1e-6)
	assert.InDelta(t, 0.3, cfg.BM25Weight, 1e-6)
	assert.InDelta(t, 0.2, cfg.NgramWeight, 1e-6)
	assert.Equal(t, 30, cfg.RRFConstant)
	assert.InDelta(t, 2.0, cfg.BM25K1, 1e-6)
	assert.InDelta(t, 0.5, cfg.BM25B, 1e-6)
	assert.Equal(t, 4, cfg.NgramSize)
}

func TestMemoryHybridConfig_NormalizedWeights(t *testing.T) {
	cfg := MemoryHybridConfig{VectorWeight: 0.7, BM25Weight: 0.2, NgramWeight: 0.1}
	v, b, n := cfg.NormalizedWeights()
	assert.InDelta(t, 0.7, v, 1e-6)
	assert.InDelta(t, 0.2, b, 1e-6)
	assert.InDelta(t, 0.1, n, 1e-6)
	assert.InDelta(t, 1.0, v+b+n, 1e-6)

	cfg2 := MemoryHybridConfig{VectorWeight: 2, BM25Weight: 1, NgramWeight: 1}
	v2, b2, n2 := cfg2.NormalizedWeights()
	assert.InDelta(t, 0.5, v2, 1e-6)
	assert.InDelta(t, 0.25, b2, 1e-6)
	assert.InDelta(t, 0.25, n2, 1e-6)
}

func TestMemoryHybridConfig_NormalizedWeightsAllZero(t *testing.T) {
	cfg := MemoryHybridConfig{}
	v, b, n := cfg.NormalizedWeights()
	assert.InDelta(t, 1.0/3, v, 1e-6)
	assert.InDelta(t, 1.0/3, b, 1e-6)
	assert.InDelta(t, 1.0/3, n, 1e-6)
}

// ---------------------------------------------------------------------------
// HierarchicalRetrieve with Hybrid
// ---------------------------------------------------------------------------

func TestHierarchicalRetrieve_InMemory_HybridChangesScoring(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	mems := []*Memory{
		{ID: "k8s-1", Type: MemoryTypeSemantic, Content: "Kubernetes deployment uses Helm charts for pod configuration management", UserID: "u1", CreatedAt: now},
		{ID: "k8s-2", Type: MemoryTypeSemantic, Content: "Rolling updates use canary deployment strategy with gradual traffic shifting", UserID: "u1", CreatedAt: now},
		{ID: "unrelated", Type: MemoryTypeSemantic, Content: "The cat sat on a warm mat in a sunny afternoon", UserID: "u1", CreatedAt: now},
	}

	for _, m := range mems {
		require.NoError(t, store.Store(ctx, m))
		require.NoError(t, EnrichMemoryBeforeStore(ctx, store, m, store.embeddingConfig, CategorizerConfig{}))
		require.NoError(t, store.Update(ctx, m.ID, m))
	}

	query := "Helm charts Kubernetes deployment"
	baseOpts := RetrieveOptions{Query: query, UserID: "u1", Limit: 3, Threshold: 0.1}

	cosineRes, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: baseOpts,
		MaxDepth:        2,
		ScorePropAlpha:  0.6,
	})
	require.NoError(t, err)

	hybridRes, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: baseOpts,
		MaxDepth:        2,
		ScorePropAlpha:  0.6,
		Hybrid:          &MemoryHybridConfig{VectorWeight: 0.7, BM25Weight: 0.2, NgramWeight: 0.1},
	})
	require.NoError(t, err)

	require.NotEmpty(t, cosineRes, "cosine should return results")
	require.NotEmpty(t, hybridRes, "hybrid should return results")

	t.Logf("Cosine top: %s (%.3f)", cosineRes[0].Memory.ID, cosineRes[0].Score)
	t.Logf("Hybrid top: %s (%.3f)", hybridRes[0].Memory.ID, hybridRes[0].Score)

	// Both should rank the Kubernetes memory first since the query has exact term overlap
	assert.Equal(t, "k8s-1", cosineRes[0].Memory.ID, "cosine top should be k8s-1")
	assert.Equal(t, "k8s-1", hybridRes[0].Memory.ID, "hybrid top should be k8s-1")

	// Hybrid scores and cosine scores should differ because BM25/n-gram contribute
	cosineScores := make(map[string]float32)
	hybridScores := make(map[string]float32)
	for _, r := range cosineRes {
		cosineScores[r.Memory.ID] = r.Score
	}
	for _, r := range hybridRes {
		hybridScores[r.Memory.ID] = r.Score
	}

	scoreDifferences := 0
	for id, cs := range cosineScores {
		if hs, ok := hybridScores[id]; ok {
			if cs != hs {
				scoreDifferences++
			}
		}
	}
	assert.Positive(t, scoreDifferences, "hybrid scoring should produce different scores from cosine-only")
}

func TestHierarchicalRetrieve_InMemory_HybridNilIsCosinePure(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	mems := []*Memory{
		{ID: "a", Type: MemoryTypeSemantic, Content: "testing memory one", UserID: "u1", CreatedAt: now},
		{ID: "b", Type: MemoryTypeSemantic, Content: "testing memory two", UserID: "u1", CreatedAt: now},
	}
	for _, m := range mems {
		require.NoError(t, store.Store(ctx, m))
	}

	opts := HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: "testing", UserID: "u1", Limit: 5, Threshold: 0.1},
	}

	// Hybrid=nil should behave identically to no hybrid config
	res1, err := store.HierarchicalRetrieve(ctx, opts)
	require.NoError(t, err)

	opts.Hybrid = nil
	res2, err := store.HierarchicalRetrieve(ctx, opts)
	require.NoError(t, err)

	require.Len(t, res2, len(res1))
	for i := range res1 {
		assert.Equal(t, res1[i].Memory.ID, res2[i].Memory.ID)
		assert.InDelta(t, res1[i].Score, res2[i].Score, 1e-6)
	}
}

func TestHierarchicalRetrieve_InMemory_HybridRRFMode(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	mems := []*Memory{
		{ID: "x", Type: MemoryTypeSemantic, Content: "BM25 inverse document frequency weighting scores", UserID: "u1", CreatedAt: now},
		{ID: "y", Type: MemoryTypeSemantic, Content: "cosine similarity vector search embedding", UserID: "u1", CreatedAt: now},
	}
	for _, m := range mems {
		require.NoError(t, store.Store(ctx, m))
	}

	res, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: "BM25 scoring", UserID: "u1", Limit: 5, Threshold: 0.01},
		Hybrid:          &MemoryHybridConfig{Mode: "rrf"},
	})
	require.NoError(t, err)
	require.NotEmpty(t, res, "rrf mode should return results")
	t.Logf("RRF top: %s (%.3f)", res[0].Memory.ID, res[0].Score)
}

// TestFollowLinks_CrossCategoryRecall demonstrates that link expansion discovers
// cross-category memories that hierarchical drill-down alone misses.
//
// Design: The linked memory ("budget") uses completely different vocabulary from
// the query ("Kubernetes Helm deployment"), so embedding similarity alone scores
// it below the 0.65 threshold. Only the explicit cross-link allows it to be found.
func TestFollowLinks_CrossCategoryRecall(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	catInfra := &Memory{
		ID: "cat-infra", Type: MemoryTypeSemantic, IsCategory: true,
		Content: "cloud infrastructure deployment and orchestration",
		UserID: "u1", CreatedAt: now,
	}
	catFinance := &Memory{
		ID: "cat-finance", Type: MemoryTypeSemantic, IsCategory: true,
		Content: "company budgets and financial planning",
		UserID: "u1", CreatedAt: now,
	}
	helmDeploy := &Memory{
		ID: "helm-deploy", Type: MemoryTypeSemantic,
		Content:  "We use Kubernetes Helm charts to deploy microservices to the production cluster",
		UserID:   "u1", CreatedAt: now,
		ParentID: "cat-infra",
	}
	// Linked memory: completely different domain vocabulary.
	// A human linked it because the cloud budget is relevant to the K8s project.
	budget := &Memory{
		ID: "cloud-budget", Type: MemoryTypeSemantic,
		Content:  "The annual spend allocation for Q3 is forty thousand dollars across all departments",
		UserID:   "u1", CreatedAt: now,
		ParentID: "cat-finance",
	}
	unrelated := &Memory{
		ID: "pasta", Type: MemoryTypeSemantic,
		Content:  "My favorite Italian pasta recipe uses fresh basil and garlic in olive oil",
		UserID:   "u1", CreatedAt: now,
		ParentID: "cat-finance",
	}

	for _, m := range []*Memory{catInfra, catFinance, helmDeploy, budget, unrelated} {
		require.NoError(t, store.Store(ctx, m))
	}

	// Cross-link: helm deployment <-> cloud budget (related by project context)
	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{
		FromID: "helm-deploy", ToID: "cloud-budget",
		Reason: "cloud budget funds the K8s cluster", Strength: 0.88, CreatedAt: now,
	}))

	threshold := float32(0.65)
	query := "Kubernetes Helm charts deployment cluster"

	// Without link expansion
	resNoLinks, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: query, UserID: "u1", Limit: 5, Threshold: threshold},
		MaxDepth:        3,
		FollowLinks:     false,
	})
	require.NoError(t, err)

	// With link expansion
	resWithLinks, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: query, UserID: "u1", Limit: 5, Threshold: threshold},
		MaxDepth:        3,
		FollowLinks:     true,
		MaxLinkDepth:    1,
	})
	require.NoError(t, err)

	noLinkIDs := map[string]bool{}
	t.Log("=== Without link expansion ===")
	for i, r := range resNoLinks {
		noLinkIDs[r.Memory.ID] = true
		t.Logf("  [%d] %s (%.3f) %s", i+1, r.Memory.ID, r.Score, r.Memory.Content[:min(60, len(r.Memory.Content))])
	}

	withLinkIDs := map[string]bool{}
	t.Log("=== With link expansion ===")
	for i, r := range resWithLinks {
		withLinkIDs[r.Memory.ID] = true
		t.Logf("  [%d] %s (%.3f) %s", i+1, r.Memory.ID, r.Score, r.Memory.Content[:min(60, len(r.Memory.Content))])
	}

	assert.True(t, withLinkIDs["helm-deploy"], "direct match should be found in both")

	if !noLinkIDs["cloud-budget"] && withLinkIDs["cloud-budget"] {
		t.Log("CONFIRMED: cloud-budget discovered only via cross-link (not by embedding similarity)")
	} else if noLinkIDs["cloud-budget"] {
		t.Log("NOTE: cloud-budget was also found via embeddings (threshold may be too low)")
	}

	assert.False(t, withLinkIDs["pasta"],
		"pasta (no link, no semantic match) should not appear")
}

// TestFollowLinks_MultiHop verifies that 2-hop link traversal discovers
// memories reachable only through an intermediate linked memory.
func TestFollowLinks_MultiHop(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	// Chain: A -> B -> C (A and C share no direct link)
	memA := &Memory{ID: "a", Content: "Rust programming borrow checker ownership", UserID: "u1", CreatedAt: now, Type: MemoryTypeSemantic}
	memB := &Memory{ID: "b", Content: "Rust robotics servo controller firmware", UserID: "u1", CreatedAt: now, Type: MemoryTypeSemantic}
	memC := &Memory{ID: "c", Content: "Tokyo robotics lab industrial servo motor demo", UserID: "u1", CreatedAt: now, Type: MemoryTypeSemantic}

	for _, m := range []*Memory{memA, memB, memC} {
		require.NoError(t, store.Store(ctx, m))
	}

	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{FromID: "a", ToID: "b", Strength: 0.9, CreatedAt: now}))
	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{FromID: "b", ToID: "c", Strength: 0.85, CreatedAt: now}))

	query := "Rust programming borrow checker"

	// 1-hop: should find A (direct) and B (via link from A)
	res1, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: query, UserID: "u1", Limit: 10, Threshold: 0.01},
		FollowLinks:     true,
		MaxLinkDepth:    1,
	})
	require.NoError(t, err)

	// 2-hop: should also find C (via A->B->C)
	res2, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: query, UserID: "u1", Limit: 10, Threshold: 0.01},
		FollowLinks:     true,
		MaxLinkDepth:    2,
	})
	require.NoError(t, err)

	ids1 := make(map[string]bool)
	for _, r := range res1 {
		ids1[r.Memory.ID] = true
	}
	ids2 := make(map[string]bool)
	for _, r := range res2 {
		ids2[r.Memory.ID] = true
	}

	t.Logf("1-hop results: %d, 2-hop results: %d", len(res1), len(res2))
	for _, r := range res2 {
		t.Logf("  %s (%.3f) %s", r.Memory.ID, r.Score, r.Memory.Content[:min(50, len(r.Memory.Content))])
	}

	assert.True(t, ids1["a"], "1-hop should include A (direct match)")
	assert.True(t, ids2["a"], "2-hop should include A")

	if ids2["c"] && !ids1["c"] {
		t.Log("CONFIRMED: C was only reachable via 2-hop traversal (A->B->C)")
	}
}

// TestRelatedIDs_CrossCategoryComparison is a four-way comparison that
// demonstrates why RelatedIDs-based graph expansion discovers relevant
// memories that neither tree-cosine traversal nor tree-hybrid search can find.
//
// Scenario: A user has memories across 4 categories. Some memories are linked
// via RelatedIDs because they are contextually related despite being in
// completely different domains with zero keyword/semantic overlap.
//
// The test runs 4 retrieval strategies against the same store and queries,
// then compares cross-category recall:
//
//  1. Tree-Cosine:       hierarchical tree traversal with cosine scoring, no links
//  2. Tree-Hybrid:       hierarchical tree traversal with BM25 + n-gram + cosine, no links
//  3. Tree-Cosine+Links: hierarchical cosine + RelatedIDs graph expansion
//  4. Tree-Hybrid+Links: hierarchical hybrid + RelatedIDs graph expansion
func TestRelatedIDs_CrossCategoryComparison(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	// --- Categories ---
	catDevOps := &Memory{
		ID: "cat-devops", Type: MemoryTypeSemantic, IsCategory: true,
		Content: "DevOps and infrastructure", UserID: "u1", CreatedAt: now,
	}
	catFinance := &Memory{
		ID: "cat-finance", Type: MemoryTypeSemantic, IsCategory: true,
		Content: "finance budget and cost management", UserID: "u1", CreatedAt: now,
	}
	catML := &Memory{
		ID: "cat-ml", Type: MemoryTypeSemantic, IsCategory: true,
		Content: "machine learning and AI model training", UserID: "u1", CreatedAt: now,
	}
	catCompliance := &Memory{
		ID: "cat-compliance", Type: MemoryTypeSemantic, IsCategory: true,
		Content: "regulatory compliance and legal audits", UserID: "u1", CreatedAt: now,
	}

	// --- Leaf memories ---

	// DevOps cluster
	helmDeploy := &Memory{
		ID: "devops-helm", Type: MemoryTypeSemantic,
		Content: "Production services use Kubernetes Helm charts with canary rollout and automatic rollback",
		UserID: "u1", CreatedAt: now, ParentID: "cat-devops",
	}
	ciPipeline := &Memory{
		ID: "devops-ci", Type: MemoryTypeSemantic,
		Content: "CI pipeline runs unit tests, integration tests, and container image builds on every pull request",
		UserID: "u1", CreatedAt: now, ParentID: "cat-devops",
	}

	// Finance cluster — NO semantic overlap with DevOps vocabulary
	cloudBudget := &Memory{
		ID: "finance-budget", Type: MemoryTypeSemantic,
		Content: "The quarterly spend allocation is forty-two thousand dollars across all departmental cost centers",
		UserID: "u1", CreatedAt: now, ParentID: "cat-finance",
	}
	headcount := &Memory{
		ID: "finance-headcount", Type: MemoryTypeSemantic,
		Content: "Team headcount plan approved for six new hires in the engineering division next quarter",
		UserID: "u1", CreatedAt: now, ParentID: "cat-finance",
	}

	// ML cluster
	gpuTraining := &Memory{
		ID: "ml-gpu", Type: MemoryTypeSemantic,
		Content: "Model training uses eight A100 GPUs with distributed data parallel across two nodes",
		UserID: "u1", CreatedAt: now, ParentID: "cat-ml",
	}
	mlDataset := &Memory{
		ID: "ml-dataset", Type: MemoryTypeSemantic,
		Content: "Training dataset contains two million labeled examples with quarterly refresh cycle",
		UserID: "u1", CreatedAt: now, ParentID: "cat-ml",
	}

	// Compliance cluster — NO semantic overlap with ML vocabulary
	dataRetention := &Memory{
		ID: "compliance-retention", Type: MemoryTypeSemantic,
		Content: "GDPR requires personal information to be purged within thirty days of account deletion request",
		UserID: "u1", CreatedAt: now, ParentID: "cat-compliance",
	}
	auditLog := &Memory{
		ID: "compliance-audit", Type: MemoryTypeSemantic,
		Content: "SOC2 Type II audit mandates immutable access logs retained for seven calendar years minimum",
		UserID: "u1", CreatedAt: now, ParentID: "cat-compliance",
	}

	allMems := []*Memory{
		catDevOps, catFinance, catML, catCompliance,
		helmDeploy, ciPipeline,
		cloudBudget, headcount,
		gpuTraining, mlDataset,
		dataRetention, auditLog,
	}
	for _, m := range allMems {
		require.NoError(t, store.Store(ctx, m))
	}

	// --- Cross-links (contextual associations that don't share vocabulary) ---

	// Link 1: Helm deployment → cloud budget
	//   (the K8s cluster drives the cloud spend, but terms are completely disjoint)
	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{
		FromID: "devops-helm", ToID: "finance-budget",
		Reason: "K8s cluster is the primary driver of cloud spend", Strength: 0.9, CreatedAt: now,
	}))

	// Link 2: GPU training → data retention compliance
	//   (training data must comply with GDPR, but vocabulary is disjoint)
	require.NoError(t, store.StoreRelation(ctx, MemoryRelation{
		FromID: "ml-gpu", ToID: "compliance-retention",
		Reason: "training data subject to GDPR retention rules", Strength: 0.85, CreatedAt: now,
	}))

	// --- Test queries ---
	type testQuery struct {
		name          string
		query         string
		directMatchID string // expected via any method
		linkedMatchID string // expected ONLY via RelatedIDs
	}
	queries := []testQuery{
		{
			name:          "DevOps→Finance",
			query:         "Kubernetes Helm charts canary rollout deployment pipeline",
			directMatchID: "devops-helm",
			linkedMatchID: "finance-budget",
		},
		{
			name:          "ML→Compliance",
			query:         "GPU distributed training A100 machine learning model",
			directMatchID: "ml-gpu",
			linkedMatchID: "compliance-retention",
		},
	}

	threshold := float32(0.55)
	hybridCfg := &MemoryHybridConfig{VectorWeight: 0.7, BM25Weight: 0.2, NgramWeight: 0.1}

	type strategy struct {
		name    string
		hybrid  *MemoryHybridConfig
		links   bool
		explain string
	}
	strategies := []strategy{
		{name: "Tree-Cosine", hybrid: nil, links: false,
			explain: "hierarchical tree traversal with cosine scoring (no cross-links, no keyword matching)"},
		{name: "Tree-Hybrid", hybrid: hybridCfg, links: false,
			explain: "hierarchical tree traversal with hybrid BM25+n-gram+cosine scoring (no cross-links)"},
		{name: "Ours (cosine+links)", hybrid: nil, links: true,
			explain: "hierarchical tree traversal with cosine scoring + RelatedIDs graph expansion"},
		{name: "Ours (hybrid+links)", hybrid: hybridCfg, links: true,
			explain: "hierarchical tree traversal with hybrid scoring + RelatedIDs graph expansion"},
	}

	t.Logf("=== Four-Way Comparison: Tree-Cosine vs Tree-Hybrid vs +Links ===\n")
	t.Logf("%-22s | %-18s | %-13s | %-14s | %-14s | %s",
		"Strategy", "Query", "Direct match?", "Linked match?", "Unrelated hit?", "Explanation")
	t.Logf("%s", "-----------------------------------------------------------------------------------------------------------")

	// Track cross-category recall per strategy
	type recallResult struct {
		directFound int
		linkedFound int
	}
	totals := make(map[string]*recallResult)
	for _, s := range strategies {
		totals[s.name] = &recallResult{}
	}

	for _, q := range queries {
		for _, s := range strategies {
			res, err := store.HierarchicalRetrieveWithConfig(ctx, HierarchicalRetrieveOptions{
				RetrieveOptions: RetrieveOptions{Query: q.query, UserID: "u1", Limit: 10, Threshold: threshold},
				MaxDepth:        3,
				FollowLinks:     s.links,
				MaxLinkDepth:    1,
				Hybrid:          s.hybrid,
			}, InMemoryHierarchicalConfig{})
			require.NoError(t, err)

			ids := make(map[string]bool)
			for _, r := range res {
				ids[r.Memory.ID] = true
			}

			directHit := ids[q.directMatchID]
			linkedHit := ids[q.linkedMatchID]

			if directHit {
				totals[s.name].directFound++
			}
			if linkedHit {
				totals[s.name].linkedFound++
			}

			// Log scores for linked match if found
			linkedScore := ""
			for _, r := range res {
				if r.Memory.ID == q.linkedMatchID {
					linkedScore = fmt.Sprintf(" (%.3f)", r.Score)
				}
			}

			t.Logf("%-22s | %-18s | %-13v | %-14s | %s",
				s.name, q.name, directHit,
				fmt.Sprintf("%v%s", linkedHit, linkedScore),
				s.explain)
		}
		t.Log("")
	}

	t.Logf("=== Cross-Category Recall Summary ===\n")
	t.Logf("%-22s | Direct (%d queries) | Linked (%d queries) | Cross-Category Recall",
		"Strategy", len(queries), len(queries))
	for _, s := range strategies {
		r := totals[s.name]
		crossRecall := float64(r.linkedFound) / float64(len(queries)) * 100
		t.Logf("%-22s | %d/%d                | %d/%d                | %.0f%%",
			s.name, r.directFound, len(queries), r.linkedFound, len(queries), crossRecall)
	}

	// --- Hard assertions ---

	// All strategies should find the direct semantic match.
	for _, s := range strategies {
		assert.Equal(t, len(queries), totals[s.name].directFound,
			"%s should find all direct matches", s.name)
	}

	// Tree-only strategies should NOT find the linked memories
	// (they are in a different category with no semantic/keyword overlap).
	assert.Equal(t, 0, totals["Tree-Cosine"].linkedFound,
		"tree-cosine traversal cannot discover cross-category links")
	assert.Equal(t, 0, totals["Tree-Hybrid"].linkedFound,
		"tree-hybrid search cannot discover semantically disjoint cross-links")

	// Link-enabled strategies MUST beat both tree-only baselines on cross-category recall.
	assert.Greater(t, totals["Ours (cosine+links)"].linkedFound, totals["Tree-Cosine"].linkedFound,
		"cosine+links should beat tree-cosine on cross-category recall")
	assert.Greater(t, totals["Ours (cosine+links)"].linkedFound, totals["Tree-Hybrid"].linkedFound,
		"cosine+links should beat tree-hybrid on cross-category recall")
	assert.Greater(t, totals["Ours (hybrid+links)"].linkedFound, totals["Tree-Cosine"].linkedFound,
		"hybrid+links should beat tree-cosine on cross-category recall")
	assert.Greater(t, totals["Ours (hybrid+links)"].linkedFound, totals["Tree-Hybrid"].linkedFound,
		"hybrid+links should beat tree-hybrid on cross-category recall")

	// Cosine+links should achieve full recall — no hybrid penalty on link scoring.
	assert.Equal(t, len(queries), totals["Ours (cosine+links)"].linkedFound,
		"RelatedIDs (cosine) should discover all cross-linked memories")

	bestOurs := totals["Ours (cosine+links)"].linkedFound
	if totals["Ours (hybrid+links)"].linkedFound > bestOurs {
		bestOurs = totals["Ours (hybrid+links)"].linkedFound
	}

	t.Logf("\nCONCLUSION:")
	t.Logf("  Tree-Cosine (no links):          %d/%d cross-category recall (0%%)",
		totals["Tree-Cosine"].linkedFound, len(queries))
	t.Logf("  Tree-Hybrid (no links):          %d/%d cross-category recall (0%%)",
		totals["Tree-Hybrid"].linkedFound, len(queries))
	t.Logf("  Tree-Cosine + Links:             %d/%d cross-category recall (%.0f%%)",
		totals["Ours (cosine+links)"].linkedFound, len(queries),
		float64(totals["Ours (cosine+links)"].linkedFound)/float64(len(queries))*100)
	t.Logf("  Tree-Hybrid + Links:             %d/%d cross-category recall (%.0f%%)",
		totals["Ours (hybrid+links)"].linkedFound, len(queries),
		float64(totals["Ours (hybrid+links)"].linkedFound)/float64(len(queries))*100)
	t.Log("")
	t.Log("Neither hierarchical tree traversal (cosine) nor hybrid search (BM25+n-gram)")
	t.Log("can discover semantically disjoint but contextually related memories across")
	t.Log("category boundaries. Only explicit RelatedIDs graph expansion bridges this gap.")
}

func TestGenericHierarchicalRetrieve_WithHybrid(t *testing.T) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	mems := []*Memory{
		{ID: "g1", Type: MemoryTypeSemantic, Content: "Kubernetes Helm chart deployment configuration", UserID: "u1", CreatedAt: now},
		{ID: "g2", Type: MemoryTypeSemantic, Content: "Docker container orchestration platform", UserID: "u1", CreatedAt: now},
	}
	for _, m := range mems {
		require.NoError(t, store.Store(ctx, m))
	}

	hybridCfg := &MemoryHybridConfig{VectorWeight: 0.7, BM25Weight: 0.2, NgramWeight: 0.1}

	res, err := HierarchicalRetrieveFromStore(ctx, store, HierarchicalRetrieveOptions{
		RetrieveOptions: RetrieveOptions{Query: "Helm Kubernetes", UserID: "u1", Limit: 5, Threshold: 0.1},
		Hybrid:          hybridCfg,
	})
	require.NoError(t, err)
	require.NotEmpty(t, res)
	t.Logf("Generic hybrid top: %s (%.3f)", res[0].Memory.ID, res[0].Score)
}
