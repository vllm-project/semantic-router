package cache

import (
	"fmt"
	"os"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
)

// ContentLength defines different query content sizes
type ContentLength int

const (
	ShortContent  ContentLength = 20  // ~20 words
	MediumContent ContentLength = 50  // ~50 words
	LongContent   ContentLength = 100 // ~100 words
)

func (c ContentLength) String() string {
	switch c {
	case ShortContent:
		return "short"
	case MediumContent:
		return "medium"
	case LongContent:
		return "long"
	default:
		return "unknown"
	}
}

// GenerateQuery generates a query with maximum semantic diversity using hash-based randomization
func generateQuery(length ContentLength, index int) string {
	// Hash the index to get pseudo-random values (deterministic but well-distributed)
	hash := uint64(index) // #nosec G115 -- index is always positive and bounded
	hash *= 2654435761    // Knuth's multiplicative hash

	// Expanded templates for maximum diversity
	templates := []string{
		// Technical how-to questions
		"How to implement %s using %s and %s for %s applications in production environments",
		"What are the best practices for %s when building %s systems with %s constraints",
		"Can you explain the architecture of %s systems that integrate %s and %s components",
		"How do I configure %s to work with %s while ensuring %s compatibility",
		"What is the recommended approach for %s development using %s and %s technologies",

		// Comparison questions
		"Explain the difference between %s and %s in the context of %s development",
		"Compare and contrast %s approaches versus %s methods for %s use cases",
		"What is the performance impact of %s versus %s for %s workloads",
		"Which is better for %s: %s or %s, considering %s requirements",
		"When should I use %s instead of %s for %s scenarios",

		// Debugging/troubleshooting
		"Can you help me debug %s issues related to %s when using %s framework",
		"Why is my %s failing when I integrate %s with %s system",
		"How to troubleshoot %s errors in %s when deploying to %s environment",
		"What causes %s problems in %s architecture with %s configuration",

		// Optimization questions
		"How do I optimize %s for %s while maintaining %s requirements",
		"What are the performance bottlenecks in %s when using %s with %s",
		"How can I improve %s throughput in %s systems running %s",
		"What are common pitfalls when optimizing %s with %s in %s environments",

		// Design/architecture questions
		"How should I design %s to handle %s and support %s functionality",
		"What are the scalability considerations for %s when implementing %s with %s",
		"How to architect %s systems that require %s and %s capabilities",
		"What design patterns work best for %s in %s architectures with %s",
	}

	// Massively expanded topics for semantic diversity
	topics := []string{
		// ML/AI
		"machine learning", "deep learning", "neural networks", "reinforcement learning",
		"computer vision", "NLP", "transformers", "embeddings", "fine-tuning",

		// Infrastructure
		"microservices", "distributed systems", "message queues", "event streaming",
		"container orchestration", "service mesh", "API gateway", "load balancing",
		"database sharding", "data replication", "consensus algorithms", "circuit breakers",

		// Data
		"data pipelines", "ETL", "data warehousing", "real-time analytics",
		"stream processing", "batch processing", "data lakes", "data modeling",

		// Security
		"authentication", "authorization", "encryption", "TLS", "OAuth",
		"API security", "zero trust", "secrets management", "key rotation",

		// Observability
		"monitoring", "logging", "tracing", "metrics", "alerting",
		"observability", "profiling", "debugging", "APM",

		// Performance
		"caching strategies", "rate limiting", "connection pooling", "query optimization",
		"memory management", "garbage collection", "CPU profiling", "I/O optimization",

		// Reliability
		"high availability", "fault tolerance", "disaster recovery", "backups",
		"failover", "redundancy", "chaos engineering", "SLA management",

		// Cloud/DevOps
		"CI/CD", "GitOps", "infrastructure as code", "configuration management",
		"auto-scaling", "serverless", "edge computing", "multi-cloud",

		// Databases
		"SQL databases", "NoSQL", "graph databases", "time series databases",
		"vector databases", "in-memory databases", "database indexing", "query planning",
	}

	// Additional random modifiers for even more diversity
	modifiers := []string{
		"large-scale", "enterprise", "cloud-native", "production-grade",
		"real-time", "distributed", "fault-tolerant", "high-performance",
		"mission-critical", "scalable", "secure", "compliant",
	}

	// Use hash to pseudo-randomly select (but deterministic for same index)
	templateIdx := int(hash % uint64(len(templates))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647                  // LCG for next random

	topic1Idx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647

	topic2Idx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647

	topic3Idx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
	hash = hash * 16807 % 2147483647

	// Build query with selected template and topics
	query := fmt.Sprintf(templates[templateIdx],
		topics[topic1Idx],
		topics[topic2Idx],
		topics[topic3Idx],
		modifiers[int(hash%uint64(len(modifiers)))]) // #nosec G115 -- modulo operation is bounded by array length

	// Add unique identifier to guarantee uniqueness
	query += fmt.Sprintf(" [Request ID: REQ-%d]", index)

	// Add extra context for longer queries
	if length > MediumContent {
		hash = hash * 16807 % 2147483647
		extraTopicIdx := int(hash % uint64(len(topics))) // #nosec G115 -- modulo operation is bounded by array length
		query += fmt.Sprintf(" Also considering %s integration and %s compatibility requirements.",
			topics[extraTopicIdx],
			modifiers[int(hash%uint64(len(modifiers)))]) // #nosec G115 -- modulo operation is bounded by array length
	}

	return query
}

// BenchmarkComprehensive runs comprehensive benchmarks across multiple dimensions
func BenchmarkComprehensive(b *testing.B) {
	// Initialize BERT model
	useCPU := os.Getenv("USE_CPU") != "false" // Default to CPU
	modelName := "sentence-transformers/all-MiniLM-L6-v2"
	if err := candle_binding.InitModel(modelName, useCPU); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	// Determine hardware type
	hardware := "cpu"
	if !useCPU {
		hardware = "gpu"
	}

	// Test configurations
	cacheSizes := []int{100, 500, 1000, 5000}
	contentLengths := []ContentLength{ShortContent, MediumContent, LongContent}
	hnswConfigs := []struct {
		name string
		m    int
		ef   int
	}{
		{"default", 16, 200},
		{"fast", 8, 100},
		{"accurate", 32, 400},
	}

	// Open CSV file for results
	csvFile, err := os.OpenFile(
		"../../benchmark_results/benchmark_data.csv",
		os.O_APPEND|os.O_CREATE|os.O_WRONLY,
		0o644)
	if err != nil {
		b.Logf("Warning: Could not open CSV file: %v", err)
	} else {
		defer csvFile.Close()
	}

	// Run benchmarks
	for _, cacheSize := range cacheSizes {
		for _, contentLen := range contentLengths {
			// Generate test data
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			// Benchmark Linear Search
			b.Run(fmt.Sprintf("%s/Linear/%s/%dEntries", hardware, contentLen.String(), cacheSize), func(b *testing.B) {
				cache := NewInMemoryCache(InMemoryCacheOptions{
					Enabled:             true,
					MaxEntries:          cacheSize * 2,
					SimilarityThreshold: 0.85,
					TTLSeconds:          0,
					UseHNSW:             false,
				})

				// Populate cache
				for i, query := range testQueries {
					reqID := fmt.Sprintf("req%d", i)
					_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
				}

				searchQuery := generateQuery(contentLen, cacheSize/2)
				b.ResetTimer()

				for i := 0; i < b.N; i++ {
					_, _, _ = cache.FindSimilar("test-model", searchQuery)
				}

				b.StopTimer()

				// Write to CSV
				if csvFile != nil {
					nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)

					line := fmt.Sprintf("%s,%s,%d,linear,0,0,%.0f,0,0,%d,1.0\n",
						hardware, contentLen.String(), cacheSize, nsPerOp, b.N)
					if _, err := csvFile.WriteString(line); err != nil {
						b.Logf("Warning: failed to write to CSV: %v", err)
					}
				}
			})

			// Benchmark HNSW with different configurations
			for _, hnswCfg := range hnswConfigs {
				b.Run(fmt.Sprintf("%s/HNSW_%s/%s/%dEntries", hardware, hnswCfg.name, contentLen.String(), cacheSize), func(b *testing.B) {
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						MaxEntries:          cacheSize * 2,
						SimilarityThreshold: 0.85,
						TTLSeconds:          0,
						UseHNSW:             true,
						HNSWM:               hnswCfg.m,
						HNSWEfConstruction:  hnswCfg.ef,
					})

					// Populate cache
					for i, query := range testQueries {
						reqID := fmt.Sprintf("req%d", i)
						_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
					}

					searchQuery := generateQuery(contentLen, cacheSize/2)
					b.ResetTimer()

					for i := 0; i < b.N; i++ {
						_, _, _ = cache.FindSimilar("test-model", searchQuery)
					}

					b.StopTimer()

					// Write to CSV
					if csvFile != nil {
						nsPerOp := float64(b.Elapsed().Nanoseconds()) / float64(b.N)

						line := fmt.Sprintf("%s,%s,%d,hnsw_%s,%d,%d,%.0f,0,0,%d,0.0\n",
							hardware, contentLen.String(), cacheSize, hnswCfg.name,
							hnswCfg.m, hnswCfg.ef, nsPerOp, b.N)
						if _, err := csvFile.WriteString(line); err != nil {
							b.Logf("Warning: failed to write to CSV: %v", err)
						}
					}
				})
			}
		}
	}
}

// BenchmarkIndexConstruction benchmarks HNSW index build time
func BenchmarkIndexConstruction(b *testing.B) {
	if err := candle_binding.InitModel("sentence-transformers/all-MiniLM-L6-v2", true); err != nil {
		b.Skipf("Failed to initialize BERT model: %v", err)
	}

	cacheSizes := []int{100, 500, 1000, 5000}
	contentLengths := []ContentLength{ShortContent, MediumContent, LongContent}

	for _, cacheSize := range cacheSizes {
		for _, contentLen := range contentLengths {
			testQueries := make([]string, cacheSize)
			for i := 0; i < cacheSize; i++ {
				testQueries[i] = generateQuery(contentLen, i)
			}

			b.Run(fmt.Sprintf("BuildIndex/%s/%dEntries", contentLen.String(), cacheSize), func(b *testing.B) {
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					b.StopTimer()
					cache := NewInMemoryCache(InMemoryCacheOptions{
						Enabled:             true,
						MaxEntries:          cacheSize * 2,
						SimilarityThreshold: 0.85,
						TTLSeconds:          0,
						UseHNSW:             true,
						HNSWM:               16,
						HNSWEfConstruction:  200,
					})
					b.StartTimer()

					// Build index by adding entries
					for j, query := range testQueries {
						reqID := fmt.Sprintf("req%d", j)
						_ = cache.AddEntry(reqID, "test-model", query, []byte(query), []byte("response"))
					}
				}
			})
		}
	}
}
