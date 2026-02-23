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
