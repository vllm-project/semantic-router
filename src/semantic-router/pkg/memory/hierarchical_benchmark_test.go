package memory

import (
	"context"
	"fmt"
	"testing"
	"time"
)

// paperTopics provides content clusters modeled on the semantic-router paper sections.
// Each cluster has a topic and 5 content variants so benchmarks can scale by
// repeating clusters across users and groups.
var paperTopics = []struct {
	topic    string
	contents []string
}{
	{
		topic: "architecture",
		contents: []string{
			"The routing pipeline uses an ExtProc gRPC filter integrated with Envoy as a sidecar proxy for request interception",
			"Signal engine collects embedding similarity, keyword matching, and domain classification signals before the decision engine",
			"The decision engine evaluates rule-based conditions over collected signals to select the optimal model backend",
			"Plugin architecture allows per-decision activation of memory, RAG, guardrails, and caching plugins",
			"Request processing follows a sequential pipeline: headers, body parsing, signal collection, decision, and model dispatch",
		},
	},
	{
		topic: "memory",
		contents: []string{
			"User memories are extracted from conversation turns by an LLM and stored in Milvus with vector embeddings",
			"Memory retention uses exponential decay model with strength S = S0 + access_count and pruning threshold R < delta",
			"Query rewriting reformulates vague user queries using conversation history before memory retrieval for better recall",
			"Deduplication checks cosine similarity of new memories against existing entries to prevent redundant storage",
			"Retrieval gating uses heuristics to skip memory search for general knowledge questions and simple greetings",
		},
	},
	{
		topic: "safety",
		contents: []string{
			"Jailbreak detection uses a fine-tuned ModernBERT classifier that flags adversarial prompt injection attempts",
			"PII detection identifies credit card numbers, social security numbers, and email addresses in requests and responses",
			"Hallucination gating compares model output against reference knowledge to flag unsupported claims with confidence scores",
			"Content safety classifier categorizes outputs into hazard taxonomies including violence, hate speech, and self-harm",
			"Guardrail plugins can block, redact, or flag content at both request and response stages of the pipeline",
		},
	},
	{
		topic: "deployment",
		contents: []string{
			"Production deployment uses Kubernetes with Helm charts that configure Envoy sidecars and ExtProc service replicas",
			"Rolling updates use canary deployment strategy with gradual traffic shifting from 10% to 50% to 100%",
			"Horizontal pod autoscaling adjusts ExtProc replicas based on gRPC request rate and CPU utilization metrics",
			"Configuration hot-reloading watches for changes in ConfigMaps and reloads routing rules without downtime",
			"Multi-cluster deployment supports geographic routing with latency-based backend selection across regions",
		},
	},
	{
		topic: "rag",
		contents: []string{
			"Hybrid retrieval combines vector similarity, BM25 keyword matching, and character n-gram Jaccard scoring",
			"Score fusion supports weighted combination with configurable weights and reciprocal rank fusion with k=60",
			"Documents are chunked with configurable size and overlap then embedded using the shared embedding model",
			"Vector store backends include in-memory for development, Milvus for production, and file-based for persistence",
			"RAG plugin configuration supports per-decision activation with different vector stores and chunk strategies",
		},
	},
	{
		topic: "evaluation",
		contents: []string{
			"End-to-end evaluation measures routing accuracy, retrieval recall at k, and response latency percentiles",
			"Benchmark suite tests signal collection latency, decision engine throughput, and memory retrieval precision",
			"A/B testing framework compares routing strategies by splitting traffic between candidate decision configurations",
			"Feedback routing loop collects user thumbs-up and thumbs-down signals to adjust model preference scores",
			"Load testing with synthetic workloads measures sustained throughput under concurrent multi-user request patterns",
		},
	},
}

// generateWorkload creates N memories distributed across paper topics.
// Every memory has an embedding (computed by the test-init BERT model).
func generateWorkload(t testing.TB, store *InMemoryStore, userID string, n int) []*Memory {
	t.Helper()
	ctx := context.Background()
	now := time.Now()

	mems := make([]*Memory, 0, n)
	for i := 0; i < n; i++ {
		cluster := paperTopics[i%len(paperTopics)]
		content := cluster.contents[i%len(cluster.contents)]

		mem := &Memory{
			ID:        fmt.Sprintf("%s-%s-%d", userID, cluster.topic, i),
			Type:      MemoryTypeSemantic,
			Content:   content,
			UserID:    userID,
			CreatedAt: now,
		}
		if err := store.Store(ctx, mem); err != nil {
			t.Fatalf("store memory %d: %v", i, err)
		}
		mems = append(mems, mem)
	}
	return mems
}

// generateHierarchicalWorkload creates N memories and enriches them so
// category nodes exist for hierarchical retrieval.
func generateHierarchicalWorkload(t testing.TB, store *InMemoryStore, userID string, n int) []*Memory {
	t.Helper()
	ctx := context.Background()
	mems := generateWorkload(t, store, userID, n)

	for _, m := range mems {
		_ = EnrichMemoryBeforeStore(ctx, store, m, store.embeddingConfig, CategorizerConfig{})
		_ = store.Update(ctx, m.ID, m)
	}
	return mems
}

// Queries drawn from the paper topics. Each targets a different cluster.
var benchQueries = []string{
	"How does the Envoy ExtProc routing pipeline process requests?",
	"How does memory retrieval and retention scoring work?",
	"What jailbreak and PII safety guardrails are available?",
	"How is Kubernetes deployment and canary rollout configured?",
	"How does hybrid RAG search combine vector, BM25, and n-gram scores?",
	"What metrics are measured in the end-to-end evaluation benchmark?",
}

// ---------------------------------------------------------------------------
// Flat retrieval baselines
// ---------------------------------------------------------------------------

func BenchmarkFlatRetrieve_30(b *testing.B)  { benchFlatRetrieve(b, 30) }
func BenchmarkFlatRetrieve_100(b *testing.B) { benchFlatRetrieve(b, 100) }
func BenchmarkFlatRetrieve_300(b *testing.B) { benchFlatRetrieve(b, 300) }

func benchFlatRetrieve(b *testing.B, n int) {
	store := newTestInMemoryStore()
	generateWorkload(b, store, "bench-user", n)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := benchQueries[i%len(benchQueries)]
		results, err := store.Retrieve(ctx, RetrieveOptions{
			Query: q, UserID: "bench-user", Limit: 5, Threshold: 0.3,
		})
		if err != nil {
			b.Fatal(err)
		}
		_ = results
	}
}

// ---------------------------------------------------------------------------
// Hierarchical retrieval (no categories -- measures overhead of the fallback path)
// ---------------------------------------------------------------------------

func BenchmarkHierarchicalRetrieve_NoCat_30(b *testing.B)  { benchHierNoCat(b, 30) }
func BenchmarkHierarchicalRetrieve_NoCat_100(b *testing.B) { benchHierNoCat(b, 100) }
func BenchmarkHierarchicalRetrieve_NoCat_300(b *testing.B) { benchHierNoCat(b, 300) }

func benchHierNoCat(b *testing.B, n int) {
	store := newTestInMemoryStore()
	generateWorkload(b, store, "bench-user", n)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := benchQueries[i%len(benchQueries)]
		results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: RetrieveOptions{
				Query: q, UserID: "bench-user", Limit: 5, Threshold: 0.3,
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		_ = results
	}
}

// ---------------------------------------------------------------------------
// Hierarchical retrieval (with categories -- full two-phase path)
// ---------------------------------------------------------------------------

func BenchmarkHierarchicalRetrieve_WithCat_30(b *testing.B)  { benchHierWithCat(b, 30) }
func BenchmarkHierarchicalRetrieve_WithCat_100(b *testing.B) { benchHierWithCat(b, 100) }
func BenchmarkHierarchicalRetrieve_WithCat_300(b *testing.B) { benchHierWithCat(b, 300) }

func benchHierWithCat(b *testing.B, n int) {
	store := newTestInMemoryStore()
	generateHierarchicalWorkload(b, store, "bench-user", n)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := benchQueries[i%len(benchQueries)]
		results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: RetrieveOptions{
				Query: q, UserID: "bench-user", Limit: 5, Threshold: 0.3,
			},
			MaxDepth:       3,
			ScorePropAlpha: 0.6,
		})
		if err != nil {
			b.Fatal(err)
		}
		_ = results
	}
}

// ---------------------------------------------------------------------------
// Generic fallback (HierarchicalRetrieveFromStore against a flat Store)
// ---------------------------------------------------------------------------

func BenchmarkGenericHierarchicalRetrieve_30(b *testing.B)  { benchGenericHier(b, 30) }
func BenchmarkGenericHierarchicalRetrieve_100(b *testing.B) { benchGenericHier(b, 100) }
func BenchmarkGenericHierarchicalRetrieve_300(b *testing.B) { benchGenericHier(b, 300) }

func benchGenericHier(b *testing.B, n int) {
	store := newTestInMemoryStore()
	generateWorkload(b, store, "bench-user", n)
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := benchQueries[i%len(benchQueries)]
		results, err := HierarchicalRetrieveFromStore(ctx, store, HierarchicalRetrieveOptions{
			RetrieveOptions: RetrieveOptions{
				Query: q, UserID: "bench-user", Limit: 5, Threshold: 0.3,
			},
		})
		if err != nil {
			b.Fatal(err)
		}
		_ = results
	}
}

// ---------------------------------------------------------------------------
// Group-scoped retrieval (multi-user, multi-group)
// ---------------------------------------------------------------------------

func BenchmarkGroupRetrieve_3Groups(b *testing.B) { benchGroupRetrieve(b, 3, 30) }
func BenchmarkGroupRetrieve_5Groups(b *testing.B) { benchGroupRetrieve(b, 5, 30) }

func benchGroupRetrieve(b *testing.B, numGroups, memsPerGroup int) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	groups := make([]string, numGroups)
	for g := 0; g < numGroups; g++ {
		gid := fmt.Sprintf("group-%d", g)
		groups[g] = gid
		for i := 0; i < memsPerGroup; i++ {
			cluster := paperTopics[i%len(paperTopics)]
			mem := &Memory{
				ID:         fmt.Sprintf("%s-mem-%d", gid, i),
				Type:       MemoryTypeSemantic,
				Content:    cluster.contents[i%len(cluster.contents)],
				UserID:     fmt.Sprintf("user-%d", g),
				GroupID:    gid,
				Visibility: VisibilityGroup,
				CreatedAt:  now,
			}
			if err := store.Store(ctx, mem); err != nil {
				b.Fatal(err)
			}
		}
	}

	// Also store requester's own memories.
	generateWorkload(b, store, "requester", 10)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		q := benchQueries[i%len(benchQueries)]
		results, err := store.HierarchicalRetrieve(ctx, HierarchicalRetrieveOptions{
			RetrieveOptions: RetrieveOptions{
				Query: q, UserID: "requester", Limit: 5, Threshold: 0.3,
			},
			IncludeGroupLevel: true,
			GroupIDs:          groups,
		})
		if err != nil {
			b.Fatal(err)
		}
		_ = results
	}
}

// ---------------------------------------------------------------------------
// EnrichMemoryBeforeStore (categorization + summary generation)
// ---------------------------------------------------------------------------

func BenchmarkEnrichMemoryBeforeStore(b *testing.B) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	// Pre-populate store so the categorizer can check for existing categories.
	generateWorkload(b, store, "bench-user", 30)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster := paperTopics[i%len(paperTopics)]
		mem := &Memory{
			ID:        fmt.Sprintf("enrich-%d", i),
			Type:      MemoryTypeSemantic,
			Content:   cluster.contents[i%len(cluster.contents)],
			UserID:    "bench-user",
			CreatedAt: now,
		}
		// Store first so it has an embedding.
		if err := store.Store(ctx, mem); err != nil {
			// ID collision on repeated runs -- ignore.
			continue
		}
		if err := EnrichMemoryBeforeStore(ctx, store, mem, store.embeddingConfig, CategorizerConfig{}); err != nil {
			b.Fatal(err)
		}
	}
}

// ---------------------------------------------------------------------------
// AutoLinkNewMemory
// ---------------------------------------------------------------------------

func BenchmarkAutoLinkNewMemory(b *testing.B) {
	store := newTestInMemoryStore()
	ctx := context.Background()
	now := time.Now()

	generateWorkload(b, store, "bench-user", 60)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cluster := paperTopics[i%len(paperTopics)]
		mem := &Memory{
			ID:        fmt.Sprintf("link-%d", i),
			Type:      MemoryTypeSemantic,
			Content:   cluster.contents[i%len(cluster.contents)],
			UserID:    "bench-user",
			CreatedAt: now,
		}
		if err := store.Store(ctx, mem); err != nil {
			continue
		}
		_, _ = AutoLinkNewMemory(ctx, store, mem, AutoLinkOptions{
			Threshold: 0.5,
			Reason:    "bench",
		})
	}
}

// ---------------------------------------------------------------------------
// Pure compute: score propagation and cosine similarity
// ---------------------------------------------------------------------------

func BenchmarkPropagateScore(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = PropagateScore(0.85, 0.72, 0.6)
	}
}

func BenchmarkCosineSimilarity_384d(b *testing.B) {
	a := make([]float32, 384)
	bb := make([]float32, 384)
	for i := range a {
		a[i] = float32(i) * 0.001
		bb[i] = float32(384-i) * 0.001
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = cosineSimilarity(a, bb)
	}
}

func BenchmarkTopicExtraction(b *testing.B) {
	kw := DefaultTopicKeywords()
	texts := []string{
		"deploy rollback release pipeline CI/CD strategy for kubernetes cluster",
		"memory retrieval retention scoring and exponential decay pruning",
		"jailbreak detection PII redaction content safety classification",
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = extractTopic(texts[i%len(texts)], MemoryTypeSemantic, kw)
	}
}
