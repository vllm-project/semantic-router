package modelselection

import (
	"bufio"
	"encoding/json"
	"os"
	"sync"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// Trainer for generating embeddings in load tests
var (
	loadTestTrainer         *Trainer
	loadTestCandleAvailable bool
	loadTestTrainerOnce     sync.Once
)

// initLoadTestTrainer initializes the trainer with Candle if available
func initLoadTestTrainer(t *testing.T) *Trainer {
	loadTestTrainerOnce.Do(func() {
		loadTestTrainer = NewTrainer(768)

		// Try to initialize Candle with Qwen3
		qwen3Path := "../../../../models/mom-embedding-pro"
		err := candle_binding.InitEmbeddingModelsBatched(qwen3Path, 32, 10, true)
		if err == nil {
			loadTestCandleAvailable = true
			loadTestTrainer.SetUseCandle(true)
			loadTestTrainer.SetEmbeddingModel("qwen3")
		}
	})

	if loadTestCandleAvailable {
		t.Log("✓ Using Qwen3/Candle embeddings for NEW queries")
	} else {
		t.Log("⚠ Candle not available - using hash-based fallback for NEW queries")
	}

	return loadTestTrainer
}

// TestCase represents a test query with its category
// Uses only query + category from VSR's 14 categories
type TestCase struct {
	Query    string
	Category string // One of the 14 VSR categories
}

// getDefaultModelRefs returns the standard set of model references for testing
func getDefaultModelRefs() []config.ModelRef {
	return []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}
}

// ============================================================================
// Tests for loading pre-trained models
// ============================================================================

func TestLoadPretrainedKNNSelector(t *testing.T) {
	modelsPath := "data/trained_models"
	selector, err := loadPretrainedSelectorFromPath("knn", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	if selector.Name() != "knn" {
		t.Errorf("Expected name 'knn', got '%s'", selector.Name())
	}

	knnSelector := selector.(*KNNSelector)
	count := knnSelector.getTrainingCount()
	if count == 0 {
		t.Error("Expected training records to be loaded, got 0")
	}
	t.Logf("Loaded KNN selector with %d training records", count)
}

func TestLoadPretrainedKMeansSelector(t *testing.T) {
	modelsPath := "data/trained_models"
	selector, err := loadPretrainedSelectorFromPath("kmeans", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	if selector.Name() != "kmeans" {
		t.Errorf("Expected name 'kmeans', got '%s'", selector.Name())
	}

	kmeansSelector := selector.(*KMeansSelector)
	count := kmeansSelector.getTrainingCount()
	if count == 0 {
		t.Error("Expected training records to be loaded, got 0")
	}
	t.Logf("Loaded KMeans selector with %d training records", count)
}

func TestLoadPretrainedMLPSelector(t *testing.T) {
	modelsPath := "data/trained_models"
	selector, err := loadPretrainedSelectorFromPath("mlp", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	if selector.Name() != "mlp" {
		t.Errorf("Expected name 'mlp', got '%s'", selector.Name())
	}

	mlpSelector := selector.(*MLPSelector)
	count := mlpSelector.getTrainingCount()
	if count == 0 {
		t.Error("Expected training records to be loaded, got 0")
	}
	t.Logf("Loaded MLP selector with %d training records", count)
}

func TestLoadPretrainedSVMSelector(t *testing.T) {
	modelsPath := "data/trained_models"
	selector, err := loadPretrainedSelectorFromPath("svm", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	if selector.Name() != "svm" {
		t.Errorf("Expected name 'svm', got '%s'", selector.Name())
	}

	svmSelector := selector.(*SVMSelector)
	count := svmSelector.getTrainingCount()
	if count == 0 {
		t.Error("Expected training records to be loaded, got 0")
	}
	t.Logf("Loaded SVM selector with %d training records", count)
}

func TestLoadPretrainedMFSelector(t *testing.T) {
	modelsPath := "data/trained_models"
	selector, err := loadPretrainedSelectorFromPath("matrix_factorization", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	if selector.Name() != "matrix_factorization" {
		t.Errorf("Expected name 'matrix_factorization', got '%s'", selector.Name())
	}

	mfSelector := selector.(*MatrixFactorizationSelector)
	count := mfSelector.getTrainingCount()
	if count == 0 {
		t.Error("Expected training records to be loaded, got 0")
	}
	t.Logf("Loaded MF selector with %d training records", count)
}

// ============================================================================
// Generalization tests with NEW queries (not in training data)
// Uses query + category format only (VSR's 14 categories)
// ============================================================================

// TestGeneralizationBiology tests model selection on NEW biology queries
func TestGeneralizationBiology(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the process of mitosis and cell division", "biology"},
		{"What are the differences between DNA and RNA?", "biology"},
		{"Describe how photosynthesis works in plants", "biology"},
		{"What is CRISPR gene editing and how does it work?", "biology"},
		{"Explain the human immune system response to pathogens", "biology"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "biology")
}

// TestGeneralizationBusiness tests model selection on NEW business queries
func TestGeneralizationBusiness(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the concept of supply chain management", "business"},
		{"What are the key principles of effective marketing strategies?", "business"},
		{"How do companies calculate return on investment?", "business"},
		{"Describe the differences between B2B and B2C business models", "business"},
		{"What is a SWOT analysis and how is it used?", "business"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "business")
}

// TestGeneralizationChemistry tests model selection on NEW chemistry queries
func TestGeneralizationChemistry(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the periodic table and element classification", "chemistry"},
		{"What is the difference between ionic and covalent bonds?", "chemistry"},
		{"Describe the process of chemical equilibrium", "chemistry"},
		{"How do acids and bases interact in a neutralization reaction?", "chemistry"},
		{"Explain the concept of oxidation and reduction reactions", "chemistry"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "chemistry")
}

// TestGeneralizationComputerScience tests model selection on NEW computer science queries
func TestGeneralizationComputerScience(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Implement a binary search algorithm in Python", "computer science"},
		{"Explain the difference between stack and queue data structures", "computer science"},
		{"Write a function to reverse a linked list", "computer science"},
		{"Describe how hash tables work and their time complexity", "computer science"},
		{"Implement merge sort with detailed comments", "computer science"},
		{"Explain the concept of recursion with an example", "computer science"},
		{"Write a depth-first search algorithm for a graph", "computer science"},
		{"What is the difference between TCP and UDP protocols?", "computer science"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "computer science")
}

// TestGeneralizationEconomics tests model selection on NEW economics queries
func TestGeneralizationEconomics(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the law of supply and demand", "economics"},
		{"What causes inflation and how is it measured?", "economics"},
		{"Describe the difference between fiscal and monetary policy", "economics"},
		{"How does GDP measure economic growth?", "economics"},
		{"Explain the concept of market equilibrium", "economics"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "economics")
}

// TestGeneralizationEngineering tests model selection on NEW engineering queries
func TestGeneralizationEngineering(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the principles of structural engineering", "engineering"},
		{"How do electric motors convert electrical energy to mechanical energy?", "engineering"},
		{"Describe the process of designing a bridge", "engineering"},
		{"What is thermodynamics and its laws?", "engineering"},
		{"Explain fluid dynamics in pipe systems", "engineering"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "engineering")
}

// TestGeneralizationHealth tests model selection on NEW health queries
func TestGeneralizationHealth(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"What are the symptoms and treatments for diabetes?", "health"},
		{"Explain how vaccines work to prevent disease", "health"},
		{"Describe the cardiovascular system and heart function", "health"},
		{"What are the effects of stress on physical health?", "health"},
		{"Explain the importance of nutrition in maintaining health", "health"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "health")
}

// TestGeneralizationHistory tests model selection on NEW history queries
func TestGeneralizationHistory(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Describe the causes and effects of World War I", "history"},
		{"What were the key events of the French Revolution?", "history"},
		{"Explain the rise and fall of the Roman Empire", "history"},
		{"What was the significance of the Industrial Revolution?", "history"},
		{"Describe the events leading to American independence", "history"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "history")
}

// TestGeneralizationLaw tests model selection on NEW law queries
func TestGeneralizationLaw(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the difference between civil and criminal law", "law"},
		{"What are the fundamental principles of contract law?", "law"},
		{"Describe the concept of due process in legal proceedings", "law"},
		{"What is intellectual property and how is it protected?", "law"},
		{"Explain the role of precedent in common law systems", "law"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "law")
}

// TestGeneralizationMath tests model selection on NEW math queries
func TestGeneralizationMath(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Solve the quadratic equation x^2 + 5x + 6 = 0", "math"},
		{"Calculate the definite integral of x^2 from 0 to 3", "math"},
		{"Prove that the square root of 2 is irrational", "math"},
		{"Find the eigenvalues of a 2x2 matrix", "math"},
		{"Explain the fundamental theorem of calculus", "math"},
		{"Solve the system of linear equations: 2x + y = 5, x - y = 1", "math"},
		{"What is the probability of rolling two sixes with two dice?", "math"},
		{"Derive the formula for the sum of an arithmetic series", "math"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "math")
}

// TestGeneralizationOther tests model selection on NEW general/other queries
func TestGeneralizationOther(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Write a creative story about a time traveler", "other"},
		{"What are some tips for improving public speaking skills?", "other"},
		{"Describe how to plan an effective meeting", "other"},
		{"What are the best practices for project management?", "other"},
		{"Give me advice on how to learn a new language", "other"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "other")
}

// TestGeneralizationPhilosophy tests model selection on NEW philosophy queries
func TestGeneralizationPhilosophy(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain Plato's theory of Forms", "philosophy"},
		{"What is the trolley problem and its ethical implications?", "philosophy"},
		{"Describe Descartes' 'I think, therefore I am' argument", "philosophy"},
		{"What is utilitarianism and how does it differ from deontology?", "philosophy"},
		{"Explain the concept of free will versus determinism", "philosophy"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "philosophy")
}

// TestGeneralizationPhysics tests model selection on NEW physics queries
func TestGeneralizationPhysics(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain Newton's three laws of motion", "physics"},
		{"What is the theory of relativity and how does it work?", "physics"},
		{"Describe the wave-particle duality of light", "physics"},
		{"How does quantum entanglement work?", "physics"},
		{"Explain the concept of entropy in thermodynamics", "physics"},
		{"What is the Heisenberg uncertainty principle?", "physics"},
		{"Describe the Standard Model of particle physics", "physics"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "physics")
}

// TestGeneralizationPsychology tests model selection on NEW psychology queries
func TestGeneralizationPsychology(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{"Explain the stages of cognitive development according to Piaget", "psychology"},
		{"What is classical conditioning and how was it discovered?", "psychology"},
		{"Describe the concept of cognitive dissonance", "psychology"},
		{"What are the main theories of personality in psychology?", "psychology"},
		{"Explain the difference between short-term and long-term memory", "psychology"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "psychology")
}

// ============================================================================
// Comprehensive generalization test across all categories
// ============================================================================

// TestGeneralizationAllCategories runs generalization tests across all 14 VSR categories
func TestGeneralizationAllCategories(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	t.Log("=== Generalization Test: All 14 VSR Categories ===")
	t.Log("Testing with NEW queries (not in training data)")
	t.Logf("Embedding mode: %s", map[bool]string{true: "Qwen3/Candle", false: "Hash-based fallback"}[loadTestCandleAvailable])

	// All test cases organized by category
	allTestCases := []TestCase{
		// Biology
		{"Explain the structure of a eukaryotic cell", "biology"},
		{"What is natural selection and how does it drive evolution?", "biology"},

		// Business
		{"What are the key components of a business plan?", "business"},
		{"Explain the concept of competitive advantage", "business"},

		// Chemistry
		{"Describe the structure of an atom", "chemistry"},
		{"What is electronegativity and how does it affect bonding?", "chemistry"},

		// Computer Science
		{"Write a breadth-first search algorithm", "computer science"},
		{"Explain Big O notation and time complexity analysis", "computer science"},
		{"Implement a function to detect cycles in a linked list", "computer science"},

		// Economics
		{"What is opportunity cost in economic decision-making?", "economics"},
		{"Explain the Phillips curve relationship", "economics"},

		// Engineering
		{"Describe the principles of control systems engineering", "engineering"},
		{"What is signal processing and where is it applied?", "engineering"},

		// Health
		{"Explain the nervous system and neurotransmitter function", "health"},
		{"What are the stages of wound healing?", "health"},

		// History
		{"Describe the impact of the printing press on society", "history"},
		{"What were the causes of the Cold War?", "history"},

		// Law
		{"Explain the concept of habeas corpus", "law"},
		{"What are the elements required to prove negligence?", "law"},

		// Math
		{"Find the derivative of sin(x) * e^x", "math"},
		{"Prove that there are infinitely many prime numbers", "math"},
		{"Calculate the determinant of a 3x3 matrix", "math"},

		// Other
		{"How do I improve my time management skills?", "other"},
		{"What are effective strategies for conflict resolution?", "other"},

		// Philosophy
		{"Explain Kant's categorical imperative", "philosophy"},
		{"What is existentialism and who are its main proponents?", "philosophy"},

		// Physics
		{"Explain Maxwell's equations for electromagnetism", "physics"},
		{"What is dark matter and dark energy?", "physics"},

		// Psychology
		{"Explain Maslow's hierarchy of needs", "psychology"},
		{"What is attachment theory in developmental psychology?", "psychology"},
	}

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			categoryModelCounts := make(map[string]map[string]int)
			for _, cat := range VSRCategories {
				categoryModelCounts[cat] = make(map[string]int)
			}

			totalSelections := 0
			successfulSelections := 0

			for _, tc := range allTestCases {
				embedding, err := trainer.GetEmbedding(tc.Query)
				if err != nil {
					t.Errorf("Failed to get embedding for query: %v", err)
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				totalSelections++
				if err != nil {
					t.Logf("  [%s] Selection error: %v", tc.Category, err)
					continue
				}

				if selected != nil {
					successfulSelections++
					categoryModelCounts[tc.Category][selected.Model]++

					queryPreview := tc.Query
					if len(queryPreview) > 50 {
						queryPreview = queryPreview[:50] + "..."
					}
					t.Logf("  [%s] '%s' -> %s", tc.Category, queryPreview, selected.Model)
				}
			}

			t.Logf("\n%s Summary: %d/%d successful selections", alg, successfulSelections, totalSelections)

			// Log per-category distribution
			for cat, models := range categoryModelCounts {
				if total := sumModelCounts(models); total > 0 {
					t.Logf("  %s: %v", cat, models)
				}
			}

			// Count unique models selected overall
			overallModels := make(map[string]int)
			for _, models := range categoryModelCounts {
				for model, count := range models {
					overallModels[model] += count
				}
			}
			t.Logf("  Overall model distribution: %v", overallModels)

			if len(overallModels) >= 2 {
				t.Logf("✓ %s selected %d different models across categories", alg, len(overallModels))
			}
		})
	}
}

// ============================================================================
// Algorithm comparison test
// ============================================================================

// TestAlgorithmComparison compares all algorithms on the same NEW queries
func TestAlgorithmComparison(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	t.Log("=== Algorithm Comparison: Same NEW queries across all algorithms ===")

	// Test cases covering key categories with NEW queries
	testCases := []TestCase{
		// Computer Science (coding) queries
		{"Implement a trie data structure for efficient string prefix matching", "computer science"},
		{"Write a function to find the shortest path in a weighted graph", "computer science"},
		{"Implement a LRU cache with O(1) get and put operations", "computer science"},

		// Math queries
		{"Solve the differential equation dy/dx = xy with y(0) = 1", "math"},
		{"Prove that the sum of first n odd numbers equals n squared", "math"},
		{"Calculate the volume of a sphere using integration", "math"},

		// Physics queries
		{"Derive the Lorentz transformation equations", "physics"},
		{"Explain the photoelectric effect and its significance", "physics"},

		// Other categories
		{"What are the main causes of climate change?", "biology"},
		{"Explain the concept of marginal utility in economics", "economics"},
		{"Describe the key principles of civil engineering", "engineering"},
	}

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	// Store results for comparison
	algorithmResults := make(map[string]map[string]int) // alg -> model -> count

	for _, alg := range algorithms {
		algorithmResults[alg] = make(map[string]int)

		selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
		if err != nil {
			t.Logf("%s: skipped (model not available)", alg)
			continue
		}

		for _, tc := range testCases {
			embedding, err := trainer.GetEmbedding(tc.Query)
			if err != nil {
				continue
			}

			featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
			ctx := &SelectionContext{
				QueryEmbedding: featureVec,
			}

			selected, err := selector.Select(ctx, modelRefs)
			if err == nil && selected != nil {
				algorithmResults[alg][selected.Model]++
			}
		}

		t.Logf("%s: %v", alg, algorithmResults[alg])
	}

	// Summary comparison
	t.Log("\n=== Algorithm Comparison Summary ===")
	for alg, models := range algorithmResults {
		uniqueModels := len(models)
		total := sumModelCounts(models)
		t.Logf("%s: %d queries, %d unique models selected", alg, total, uniqueModels)
	}
}

// ============================================================================
// Distribution tests
// ============================================================================

// TestModelDistributionPerCategory tests model selection distribution for each category
func TestModelDistributionPerCategory(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	t.Log("=== Model Distribution Per Category ===")

	// Multiple queries per category for distribution analysis
	categoryQueries := map[string][]string{
		"computer science": {
			"Implement quicksort algorithm",
			"Write a recursive fibonacci function",
			"Create a binary tree traversal function",
			"Design a hash table implementation",
			"Write a graph coloring algorithm",
		},
		"math": {
			"Solve x^2 - 4x + 3 = 0",
			"Find the limit of sin(x)/x as x approaches 0",
			"Calculate the area under y = x^2 from 0 to 2",
			"Prove the Pythagorean theorem",
			"Find the GCD of 48 and 18",
		},
		"physics": {
			"Explain conservation of momentum",
			"Derive the kinetic energy formula",
			"Explain Ohm's law",
			"Describe simple harmonic motion",
			"Explain the Doppler effect",
		},
		"biology": {
			"Explain DNA replication",
			"Describe protein synthesis",
			"Explain cellular respiration",
			"Describe the structure of a neuron",
			"Explain how blood clotting works",
		},
	}

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			for category, queries := range categoryQueries {
				modelCounts := make(map[string]int)

				for _, query := range queries {
					embedding, err := trainer.GetEmbedding(query)
					if err != nil {
						continue
					}

					featureVec := CombineEmbeddingWithCategory(embedding, category)
					ctx := &SelectionContext{
						QueryEmbedding: featureVec,
					}

					selected, err := selector.Select(ctx, modelRefs)
					if err == nil && selected != nil {
						modelCounts[selected.Model]++
					}
				}

				total := sumModelCounts(modelCounts)
				t.Logf("%s [%s]: %v (total: %d)", alg, category, modelCounts, total)
			}
		})
	}
}

// ============================================================================
// Complex query tests - Advanced/Expert-level queries
// ============================================================================

// TestComplexComputerScienceQueries tests with advanced CS/coding queries
func TestComplexComputerScienceQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Advanced algorithms
		{"Implement a self-balancing AVL tree with rotation operations in Python", "computer science"},
		{"Write a concurrent hash map implementation with fine-grained locking", "computer science"},
		{"Implement Dijkstra's algorithm with a Fibonacci heap for optimal performance", "computer science"},
		{"Create a lock-free queue using compare-and-swap atomic operations", "computer science"},
		{"Implement the Aho-Corasick string matching algorithm for multiple pattern search", "computer science"},

		// Systems programming
		{"Design a garbage collector using the mark-and-sweep algorithm", "computer science"},
		{"Implement a memory allocator similar to malloc with free list management", "computer science"},
		{"Write a simple virtual machine bytecode interpreter", "computer science"},
		{"Create a thread pool executor with work stealing for load balancing", "computer science"},
		{"Implement a B+ tree index structure for database storage", "computer science"},

		// Distributed systems
		{"Design a distributed consensus algorithm based on Raft", "computer science"},
		{"Implement a consistent hashing ring for distributed cache", "computer science"},
		{"Write a vector clock implementation for distributed ordering", "computer science"},
		{"Create a MapReduce framework for parallel data processing", "computer science"},
		{"Implement a two-phase commit protocol for distributed transactions", "computer science"},

		// Compilers and languages
		{"Write a recursive descent parser for arithmetic expressions", "computer science"},
		{"Implement a lexical analyzer for a simple programming language", "computer science"},
		{"Create an abstract syntax tree builder for a calculator language", "computer science"},
		{"Design a type inference algorithm for a functional language", "computer science"},
		{"Implement tail call optimization in an interpreter", "computer science"},

		// Machine learning implementations
		{"Implement gradient descent optimization from scratch in NumPy", "computer science"},
		{"Write a k-means clustering algorithm with random initialization", "computer science"},
		{"Create a neural network backpropagation implementation", "computer science"},
		{"Implement a decision tree classifier with information gain splitting", "computer science"},
		{"Write a support vector machine using sequential minimal optimization", "computer science"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "computer science (complex)")
}

// TestComplexMathQueries tests with advanced mathematics queries
func TestComplexMathQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Advanced calculus
		{"Evaluate the triple integral of xyz over the unit sphere", "math"},
		{"Prove the divergence theorem using Stokes' theorem", "math"},
		{"Find the Taylor series expansion of ln(1+x) and determine convergence", "math"},
		{"Solve the partial differential equation: ∂²u/∂x² + ∂²u/∂y² = 0", "math"},
		{"Calculate the Jacobian determinant for polar to Cartesian transformation", "math"},

		// Linear algebra
		{"Prove that eigenvalues of a symmetric matrix are real", "math"},
		{"Find the singular value decomposition of a 3x4 matrix", "math"},
		{"Prove the Cayley-Hamilton theorem for 2x2 matrices", "math"},
		{"Compute the Jordan normal form of a defective matrix", "math"},
		{"Prove that orthogonal matrices preserve inner products", "math"},

		// Abstract algebra
		{"Prove that every finite group of prime order is cyclic", "math"},
		{"Show that the kernel of a group homomorphism is a normal subgroup", "math"},
		{"Prove Lagrange's theorem about subgroup orders", "math"},
		{"Demonstrate that polynomial rings over fields are principal ideal domains", "math"},
		{"Prove the first isomorphism theorem for groups", "math"},

		// Number theory
		{"Prove Fermat's little theorem using group theory", "math"},
		{"Find all solutions to x² ≡ 1 (mod p) for prime p", "math"},
		{"Prove the law of quadratic reciprocity", "math"},
		{"Calculate the Euler phi function for 360", "math"},
		{"Prove that there are infinitely many primes of form 4k+3", "math"},

		// Complex analysis
		{"Evaluate the contour integral of 1/(z²+1) around |z|=2", "math"},
		{"Prove the Cauchy integral formula", "math"},
		{"Find the Laurent series of e^(1/z) around z=0", "math"},
		{"Calculate residues of 1/sin(z) at its poles", "math"},
		{"Prove Liouville's theorem about bounded entire functions", "math"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "math (complex)")
}

// TestComplexPhysicsQueries tests with advanced physics queries
func TestComplexPhysicsQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Quantum mechanics
		{"Derive the time-independent Schrödinger equation for hydrogen atom", "physics"},
		{"Calculate the expectation value of momentum for a particle in a box", "physics"},
		{"Explain the Pauli exclusion principle and its role in atomic structure", "physics"},
		{"Derive the selection rules for electric dipole transitions", "physics"},
		{"Solve the harmonic oscillator using ladder operators", "physics"},

		// Electromagnetism
		{"Derive Maxwell's equations from the action principle", "physics"},
		{"Calculate the magnetic field of a solenoid using Ampere's law", "physics"},
		{"Derive the wave equation for electromagnetic waves in vacuum", "physics"},
		{"Explain the Poynting vector and electromagnetic energy flow", "physics"},
		{"Calculate the radiation resistance of a half-wave dipole antenna", "physics"},

		// Relativity
		{"Derive time dilation from the Lorentz transformation", "physics"},
		{"Calculate the Schwarzschild radius for a black hole", "physics"},
		{"Explain gravitational lensing using general relativity", "physics"},
		{"Derive the relativistic energy-momentum relation", "physics"},
		{"Calculate proper time for a particle in circular motion", "physics"},

		// Statistical mechanics
		{"Derive the Fermi-Dirac distribution from first principles", "physics"},
		{"Calculate the partition function for a two-level system", "physics"},
		{"Explain the Boltzmann entropy formula and its implications", "physics"},
		{"Derive the equipartition theorem and its limitations", "physics"},
		{"Calculate the specific heat of a solid using the Debye model", "physics"},

		// Particle physics
		{"Explain the Higgs mechanism for mass generation", "physics"},
		{"Describe quark confinement in quantum chromodynamics", "physics"},
		{"Calculate the decay width of the Z boson", "physics"},
		{"Explain CP violation in the weak interaction", "physics"},
		{"Describe the running of coupling constants in QED", "physics"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "physics (complex)")
}

// TestComplexEngineeringQueries tests with advanced engineering queries
func TestComplexEngineeringQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Control systems
		{"Design a PID controller for a second-order underdamped system", "engineering"},
		{"Analyze stability using the Nyquist criterion", "engineering"},
		{"Design a state observer for a linear time-invariant system", "engineering"},
		{"Implement model predictive control for a MIMO system", "engineering"},
		{"Design a Kalman filter for sensor fusion applications", "engineering"},

		// Signal processing
		{"Derive the discrete Fourier transform from first principles", "engineering"},
		{"Design an FIR low-pass filter using the window method", "engineering"},
		{"Implement a fast Fourier transform algorithm", "engineering"},
		{"Design an IIR Butterworth filter with specified cutoff frequency", "engineering"},
		{"Analyze aliasing effects in digital-to-analog conversion", "engineering"},

		// Power systems
		{"Perform load flow analysis using the Newton-Raphson method", "engineering"},
		{"Design a power factor correction circuit", "engineering"},
		{"Analyze transient stability of a synchronous generator", "engineering"},
		{"Design a three-phase inverter for grid-tie applications", "engineering"},
		{"Calculate short circuit currents in a power distribution network", "engineering"},

		// Structural engineering
		{"Analyze stress distribution in a cantilever beam under load", "engineering"},
		{"Design a reinforced concrete column for axial and moment loads", "engineering"},
		{"Perform finite element analysis for a truss structure", "engineering"},
		{"Calculate natural frequencies of a multi-story building", "engineering"},
		{"Design a foundation for a tall building considering soil interaction", "engineering"},

		// Aerospace engineering
		{"Calculate lift and drag coefficients for an airfoil", "engineering"},
		{"Design a rocket nozzle for maximum thrust efficiency", "engineering"},
		{"Analyze orbital mechanics for satellite trajectory", "engineering"},
		{"Calculate heat shield requirements for atmospheric reentry", "engineering"},
		{"Design a control system for quadcopter stabilization", "engineering"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "engineering (complex)")
}

// TestComplexBiologyQueries tests with advanced biology queries
func TestComplexBiologyQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Molecular biology
		{"Explain the mechanism of CRISPR-Cas9 gene editing with PAM recognition", "biology"},
		{"Describe the process of mRNA splicing including snRNP complexes", "biology"},
		{"Explain epigenetic modifications and their role in gene expression", "biology"},
		{"Describe the lac operon regulation mechanism in E. coli", "biology"},
		{"Explain the polymerase chain reaction with detailed enzyme kinetics", "biology"},

		// Cell biology
		{"Describe the electron transport chain in mitochondria", "biology"},
		{"Explain the mechanism of vesicular transport and SNARE proteins", "biology"},
		{"Describe cell cycle checkpoints and their molecular regulators", "biology"},
		{"Explain the signal transduction pathway of receptor tyrosine kinases", "biology"},
		{"Describe the mechanism of apoptosis including caspase cascades", "biology"},

		// Genetics
		{"Explain Hardy-Weinberg equilibrium and its assumptions", "biology"},
		{"Describe quantitative trait loci mapping methodology", "biology"},
		{"Explain genomic imprinting and its molecular basis", "biology"},
		{"Describe the molecular basis of X-chromosome inactivation", "biology"},
		{"Explain mitochondrial inheritance patterns and heteroplasmy", "biology"},

		// Evolutionary biology
		{"Explain the molecular clock hypothesis and its calibration", "biology"},
		{"Describe phylogenetic tree construction using maximum likelihood", "biology"},
		{"Explain adaptive radiation with evolutionary mechanisms", "biology"},
		{"Describe the neutral theory of molecular evolution", "biology"},
		{"Explain horizontal gene transfer and its evolutionary significance", "biology"},

		// Neuroscience
		{"Describe long-term potentiation at the molecular level", "biology"},
		{"Explain the blood-brain barrier structure and function", "biology"},
		{"Describe the mechanism of action potential generation", "biology"},
		{"Explain neurotransmitter release and synaptic vesicle cycling", "biology"},
		{"Describe the molecular basis of memory formation", "biology"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "biology (complex)")
}

// TestComplexChemistryQueries tests with advanced chemistry queries
func TestComplexChemistryQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Organic chemistry
		{"Explain the mechanism of Diels-Alder cycloaddition reactions", "chemistry"},
		{"Describe stereoselectivity in aldol reactions", "chemistry"},
		{"Explain the mechanism of palladium-catalyzed cross-coupling", "chemistry"},
		{"Describe retrosynthetic analysis for complex molecule synthesis", "chemistry"},
		{"Explain the mechanism of olefin metathesis reactions", "chemistry"},

		// Inorganic chemistry
		{"Explain crystal field theory and d-orbital splitting", "chemistry"},
		{"Describe the bonding in metal carbonyl complexes", "chemistry"},
		{"Explain the 18-electron rule in organometallic chemistry", "chemistry"},
		{"Describe the synthesis and properties of coordination polymers", "chemistry"},
		{"Explain the mechanism of homogeneous catalysis by transition metals", "chemistry"},

		// Physical chemistry
		{"Derive the Arrhenius equation from transition state theory", "chemistry"},
		{"Explain the Langmuir adsorption isotherm derivation", "chemistry"},
		{"Describe phase diagrams and the Gibbs phase rule", "chemistry"},
		{"Explain the quantum mechanical treatment of molecular orbitals", "chemistry"},
		{"Describe electrochemical impedance spectroscopy principles", "chemistry"},

		// Analytical chemistry
		{"Explain the principles of mass spectrometry fragmentation", "chemistry"},
		{"Describe NMR spectroscopy chemical shift interpretation", "chemistry"},
		{"Explain chromatographic resolution and van Deemter equation", "chemistry"},
		{"Describe X-ray diffraction for crystal structure determination", "chemistry"},
		{"Explain electrochemical detection methods in chromatography", "chemistry"},

		// Biochemistry
		{"Describe the mechanism of ATP synthase rotary catalysis", "chemistry"},
		{"Explain enzyme kinetics with the Michaelis-Menten model", "chemistry"},
		{"Describe the allosteric regulation of hemoglobin", "chemistry"},
		{"Explain the chemistry of oxidative phosphorylation", "chemistry"},
		{"Describe protein folding thermodynamics and kinetics", "chemistry"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "chemistry (complex)")
}

// TestComplexEconomicsQueries tests with advanced economics queries
func TestComplexEconomicsQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Microeconomics
		{"Derive the Slutsky equation for substitution and income effects", "economics"},
		{"Analyze Nash equilibrium in a Cournot oligopoly model", "economics"},
		{"Explain the theory of second best in welfare economics", "economics"},
		{"Analyze adverse selection using the Akerlof lemons model", "economics"},
		{"Derive the conditions for Pareto optimal allocation", "economics"},

		// Macroeconomics
		{"Explain the IS-LM model with policy implications", "economics"},
		{"Analyze the Solow growth model with technological progress", "economics"},
		{"Describe the New Keynesian Phillips curve derivation", "economics"},
		{"Explain the Mundell-Fleming model for open economies", "economics"},
		{"Analyze the real business cycle theory framework", "economics"},

		// Econometrics
		{"Explain the assumptions of ordinary least squares regression", "economics"},
		{"Describe instrumental variables estimation for endogeneity", "economics"},
		{"Explain difference-in-differences methodology for causal inference", "economics"},
		{"Describe panel data analysis with fixed and random effects", "economics"},
		{"Explain the generalized method of moments estimation", "economics"},

		// Financial economics
		{"Derive the Black-Scholes option pricing formula", "economics"},
		{"Explain the Capital Asset Pricing Model derivation", "economics"},
		{"Describe portfolio optimization using mean-variance analysis", "economics"},
		{"Analyze market microstructure and price discovery", "economics"},
		{"Explain the term structure of interest rates models", "economics"},

		// Game theory
		{"Analyze the prisoner's dilemma with repeated game dynamics", "economics"},
		{"Describe mechanism design for auction theory", "economics"},
		{"Explain signaling games and separating equilibria", "economics"},
		{"Analyze bargaining theory with the Nash bargaining solution", "economics"},
		{"Describe evolutionary game theory and replicator dynamics", "economics"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "economics (complex)")
}

// TestComplexLawQueries tests with advanced law queries
func TestComplexLawQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Constitutional law
		{"Analyze the strict scrutiny standard for fundamental rights", "law"},
		{"Explain the commerce clause jurisprudence evolution", "law"},
		{"Describe the incorporation doctrine and selective incorporation", "law"},
		{"Analyze the political question doctrine and judicial review", "law"},
		{"Explain the nondelegation doctrine and administrative agencies", "law"},

		// Contract law
		{"Analyze the parol evidence rule and its exceptions", "law"},
		{"Explain promissory estoppel as a substitute for consideration", "law"},
		{"Describe the doctrine of unconscionability in contracts", "law"},
		{"Analyze anticipatory repudiation and adequate assurance", "law"},
		{"Explain the perfect tender rule in sales of goods", "law"},

		// Tort law
		{"Analyze proximate cause and the foreseeability test", "law"},
		{"Explain the learned intermediary doctrine in product liability", "law"},
		{"Describe the economic loss rule and its exceptions", "law"},
		{"Analyze joint and several liability in mass torts", "law"},
		{"Explain the rescue doctrine and duty to rescue", "law"},

		// Criminal law
		{"Analyze the felony murder rule and its limitations", "law"},
		{"Explain the insanity defense standards across jurisdictions", "law"},
		{"Describe the entrapment defense elements and standards", "law"},
		{"Analyze conspiracy law and the Pinkerton doctrine", "law"},
		{"Explain the exclusionary rule and its exceptions", "law"},

		// International law
		{"Analyze state sovereignty and humanitarian intervention", "law"},
		{"Explain the doctrine of state responsibility in international law", "law"},
		{"Describe treaty interpretation under the Vienna Convention", "law"},
		{"Analyze customary international law formation", "law"},
		{"Explain the principle of universal jurisdiction", "law"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "law (complex)")
}

// TestComplexPhilosophyQueries tests with advanced philosophy queries
func TestComplexPhilosophyQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Epistemology
		{"Analyze Gettier cases and the justified true belief theory", "philosophy"},
		{"Explain foundationalism versus coherentism in epistemology", "philosophy"},
		{"Describe the problem of induction and Hume's skeptical solution", "philosophy"},
		{"Analyze reliabilism as a theory of epistemic justification", "philosophy"},
		{"Explain the Münchhausen trilemma and its implications", "philosophy"},

		// Metaphysics
		{"Analyze the problem of personal identity over time", "philosophy"},
		{"Explain modal realism and possible worlds semantics", "philosophy"},
		{"Describe the problem of universals: realism vs nominalism", "philosophy"},
		{"Analyze causation theories: regularity, counterfactual, and powers", "philosophy"},
		{"Explain the debate between endurantism and perdurantism", "philosophy"},

		// Ethics
		{"Analyze the is-ought problem and naturalistic fallacy", "philosophy"},
		{"Explain moral realism versus anti-realism debates", "philosophy"},
		{"Describe virtue ethics and the doctrine of the mean", "philosophy"},
		{"Analyze contractualism in moral and political philosophy", "philosophy"},
		{"Explain the demandingness objection to consequentialism", "philosophy"},

		// Philosophy of mind
		{"Analyze the hard problem of consciousness", "philosophy"},
		{"Explain functionalism and multiple realizability arguments", "philosophy"},
		{"Describe the Chinese room argument against strong AI", "philosophy"},
		{"Analyze the knowledge argument and Mary's room", "philosophy"},
		{"Explain eliminative materialism and folk psychology", "philosophy"},

		// Philosophy of science
		{"Analyze the underdetermination thesis in philosophy of science", "philosophy"},
		{"Explain scientific realism versus instrumentalism", "philosophy"},
		{"Describe Kuhn's paradigm shifts and scientific revolutions", "philosophy"},
		{"Analyze the demarcation problem: science versus pseudoscience", "philosophy"},
		{"Explain the theory-ladenness of observation", "philosophy"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "philosophy (complex)")
}

// TestComplexPsychologyQueries tests with advanced psychology queries
func TestComplexPsychologyQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Cognitive psychology
		{"Explain the levels of processing theory of memory encoding", "psychology"},
		{"Describe the dual-process theory of reasoning (System 1 vs System 2)", "psychology"},
		{"Analyze the working memory model and executive functions", "psychology"},
		{"Explain cognitive load theory and instructional design", "psychology"},
		{"Describe the feature integration theory of attention", "psychology"},

		// Developmental psychology
		{"Analyze Vygotsky's zone of proximal development theory", "psychology"},
		{"Explain the theory of mind development in children", "psychology"},
		{"Describe Erikson's psychosocial stages of development", "psychology"},
		{"Analyze the strange situation and attachment styles", "psychology"},
		{"Explain moral development stages according to Kohlberg", "psychology"},

		// Social psychology
		{"Analyze the bystander effect and diffusion of responsibility", "psychology"},
		{"Explain cognitive dissonance theory and attitude change", "psychology"},
		{"Describe the elaboration likelihood model of persuasion", "psychology"},
		{"Analyze social identity theory and intergroup relations", "psychology"},
		{"Explain the Stanford prison experiment and situational factors", "psychology"},

		// Clinical psychology
		{"Describe the cognitive model of depression (Beck's theory)", "psychology"},
		{"Analyze the diathesis-stress model of psychopathology", "psychology"},
		{"Explain dialectical behavior therapy for borderline personality", "psychology"},
		{"Describe exposure therapy mechanisms for anxiety disorders", "psychology"},
		{"Analyze the biopsychosocial model of mental health", "psychology"},

		// Neuroscience/biopsychology
		{"Explain neuroplasticity and Hebbian learning principles", "psychology"},
		{"Describe the dopamine reward system and addiction", "psychology"},
		{"Analyze the HPA axis and stress response mechanisms", "psychology"},
		{"Explain split-brain research and hemispheric specialization", "psychology"},
		{"Describe the neuroscience of emotion regulation", "psychology"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "psychology (complex)")
}

// TestComplexHealthQueries tests with advanced health queries
func TestComplexHealthQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Pathophysiology
		{"Explain the pathophysiology of type 2 diabetes mellitus", "health"},
		{"Describe the inflammatory cascade in sepsis", "health"},
		{"Analyze the pathogenesis of atherosclerotic plaque formation", "health"},
		{"Explain the mechanisms of autoimmune diseases", "health"},
		{"Describe the pathophysiology of heart failure", "health"},

		// Pharmacology
		{"Explain pharmacokinetics: absorption, distribution, metabolism, excretion", "health"},
		{"Describe the mechanism of action of ACE inhibitors", "health"},
		{"Analyze drug-drug interactions and cytochrome P450", "health"},
		{"Explain the pharmacology of opioid receptors and pain management", "health"},
		{"Describe antibiotic resistance mechanisms", "health"},

		// Epidemiology
		{"Explain cohort versus case-control study designs", "health"},
		{"Describe confounding variables and methods to control them", "health"},
		{"Analyze sensitivity and specificity of diagnostic tests", "health"},
		{"Explain the concept of herd immunity thresholds", "health"},
		{"Describe the Bradford Hill criteria for causation", "health"},

		// Clinical medicine
		{"Explain the differential diagnosis approach for chest pain", "health"},
		{"Describe the management of acute myocardial infarction", "health"},
		{"Analyze the staging and treatment of cancer", "health"},
		{"Explain evidence-based medicine and systematic reviews", "health"},
		{"Describe clinical decision-making under uncertainty", "health"},

		// Public health
		{"Analyze social determinants of health disparities", "health"},
		{"Explain the health belief model for behavior change", "health"},
		{"Describe pandemic preparedness and response strategies", "health"},
		{"Analyze cost-effectiveness in healthcare resource allocation", "health"},
		{"Explain the epidemiological transition in global health", "health"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "health (complex)")
}

// TestComplexHistoryQueries tests with advanced history queries
func TestComplexHistoryQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Historiography
		{"Analyze the Annales school approach to historical research", "history"},
		{"Explain postcolonial historiography and subaltern studies", "history"},
		{"Describe the linguistic turn in historical methodology", "history"},
		{"Analyze Marxist interpretations of historical development", "history"},
		{"Explain microhistory and its methodological contributions", "history"},

		// Ancient history
		{"Analyze the decline and fall of the Western Roman Empire", "history"},
		{"Explain the political structure of Athenian democracy", "history"},
		{"Describe the Silk Road and its cultural exchanges", "history"},
		{"Analyze the unification of China under Qin Shi Huang", "history"},
		{"Explain the rise and spread of early Christianity", "history"},

		// Medieval history
		{"Analyze the feudal system and manorial economy", "history"},
		{"Explain the Investiture Controversy and church-state relations", "history"},
		{"Describe the Crusades and their long-term consequences", "history"},
		{"Analyze the Black Death's demographic and social impacts", "history"},
		{"Explain the development of medieval universities", "history"},

		// Modern history
		{"Analyze the causes of World War I using historiographical debate", "history"},
		{"Explain the ideological foundations of fascism and Nazism", "history"},
		{"Describe the decolonization process in Africa and Asia", "history"},
		{"Analyze the origins and dynamics of the Cold War", "history"},
		{"Explain the factors leading to the collapse of the Soviet Union", "history"},

		// Economic history
		{"Analyze the Industrial Revolution's global economic impact", "history"},
		{"Explain the Great Depression's causes and policy responses", "history"},
		{"Describe the Bretton Woods system and its collapse", "history"},
		{"Analyze the economic history of colonialism and imperialism", "history"},
		{"Explain the transition from feudalism to capitalism", "history"},
	}

	runGeneralizationTest(t, trainer, modelsPath, testCases, "history (complex)")
}

// ============================================================================
// Edge case and stress tests
// ============================================================================

// TestEdgeCaseQueries tests model selection with unusual query patterns
func TestEdgeCaseQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Very short queries
		{"Sort array", "computer science"},
		{"Solve x=5", "math"},
		{"DNA structure", "biology"},
		{"Ohm's law", "physics"},
		{"Supply demand", "economics"},

		// Technical jargon-heavy queries
		{"Implement CRDT for eventual consistency in distributed databases", "computer science"},
		{"Calculate the Christoffel symbols for Schwarzschild metric", "physics"},
		{"Analyze the renormalization group flow in QFT", "physics"},
		{"Explain the Hückel approximation in molecular orbital theory", "chemistry"},
		{"Describe the Wardrop equilibrium in traffic network analysis", "economics"},

		// Multi-step problem queries
		{"First find the roots of x^3 - 6x^2 + 11x - 6 = 0, then use them to factor the polynomial completely and verify your answer", "math"},
		{"Implement a graph algorithm that first performs topological sort, then uses the result to find the longest path in a DAG", "computer science"},
		{"Calculate the electric field of a uniformly charged sphere at all points, then integrate to find the potential energy", "physics"},

		// Queries with code snippets
		{"Debug this code: for i in range(10) print(i) - what's wrong and how to fix it?", "computer science"},
		{"Optimize this algorithm: def fib(n): return fib(n-1)+fib(n-2) if n>1 else n", "computer science"},
		{"Explain the time complexity of: while n > 1: if n % 2 == 0: n = n // 2 else: n = 3 * n + 1", "computer science"},

		// Cross-disciplinary queries
		{"Explain how information theory concepts apply to neural coding in the brain", "biology"},
		{"Describe the mathematical foundations of quantum computing algorithms", "computer science"},
		{"Analyze the game theory behind international climate negotiations", "economics"},
		{"Explain how entropy concepts in physics relate to thermodynamics in chemistry", "chemistry"},
		{"Describe the statistical mechanics of protein folding", "biology"},

		// Real-world application queries
		{"How would you design a recommendation system for a streaming service?", "computer science"},
		{"Calculate the trajectory of a Mars-bound spacecraft accounting for gravitational assists", "physics"},
		{"Design an experiment to test a new cancer drug treatment efficacy", "health"},
		{"Develop an economic model for pricing carbon emissions", "economics"},
		{"Create a machine learning model to predict protein structures", "computer science"},
	}

	t.Log("=== Edge Case Queries Test ===")
	t.Log("Testing unusual query patterns and edge cases")

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			modelCounts := make(map[string]int)
			successCount := 0

			for _, tc := range testCases {
				embedding, err := trainer.GetEmbedding(tc.Query)
				if err != nil {
					t.Logf("  Embedding error for '%s': %v", truncateQuery(tc.Query, 30), err)
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Logf("  Selection error for '%s': %v", truncateQuery(tc.Query, 30), err)
					continue
				}

				if selected != nil {
					successCount++
					modelCounts[selected.Model]++
					t.Logf("  [%s] '%s' -> %s", tc.Category, truncateQuery(tc.Query, 40), selected.Model)
				}
			}

			t.Logf("%s: %d/%d successful, distribution: %v", alg, successCount, len(testCases), modelCounts)
		})
	}
}

// TestLongFormQueries tests model selection with detailed, long queries
func TestLongFormQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		{
			"I need to implement a complete web application backend that includes user authentication with JWT tokens, a RESTful API with proper error handling, database integration with connection pooling, rate limiting, and comprehensive logging. Can you provide a detailed implementation guide with code examples in Python using FastAPI?",
			"computer science",
		},
		{
			"Please explain the complete process of how the human body responds to a viral infection, starting from when the virus first enters the body, through the innate immune response, the adaptive immune response including both B-cell and T-cell activation, antibody production, and finally how immunological memory is formed for future protection against the same pathogen.",
			"biology",
		},
		{
			"I'm trying to understand the mathematical foundations of neural networks. Can you derive the backpropagation algorithm from first principles, starting with the chain rule of calculus, showing how gradients are computed for each layer, and explaining why certain activation functions are preferred over others in terms of their gradients and the vanishing gradient problem?",
			"math",
		},
		{
			"Describe the complete lifecycle of a star from its formation in a molecular cloud through nuclear fusion stages, including the proton-proton chain and CNO cycle, stellar evolution through the main sequence, red giant phase, and final fate depending on mass, whether it becomes a white dwarf, neutron star, or black hole.",
			"physics",
		},
		{
			"Analyze the economic impacts of globalization on developing countries, considering both positive effects such as increased foreign direct investment, technology transfer, and job creation, as well as negative effects including environmental degradation, exploitation of labor, and widening income inequality. Include specific examples from at least three different regions.",
			"economics",
		},
	}

	t.Log("=== Long Form Queries Test ===")
	t.Log("Testing detailed, comprehensive queries")

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			modelCounts := make(map[string]int)

			for _, tc := range testCases {
				embedding, err := trainer.GetEmbedding(tc.Query)
				if err != nil {
					t.Errorf("Embedding error: %v", err)
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("Selection error: %v", err)
					continue
				}

				if selected != nil {
					modelCounts[selected.Model]++
					t.Logf("  [%s] '%s...' -> %s", tc.Category, truncateQuery(tc.Query, 60), selected.Model)
				}
			}

			t.Logf("%s distribution: %v", alg, modelCounts)
		})
	}
}

// TestMixedDifficultyQueries tests model selection across difficulty levels
func TestMixedDifficultyQueries(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	testCases := []TestCase{
		// Beginner level
		{"What is a variable in programming?", "computer science"},
		{"What is addition?", "math"},
		{"What are cells?", "biology"},

		// Intermediate level
		{"Explain object-oriented programming concepts", "computer science"},
		{"Solve a system of linear equations", "math"},
		{"Describe the cell cycle phases", "biology"},

		// Advanced level
		{"Implement a red-black tree with all rotation operations", "computer science"},
		{"Prove the Cauchy-Schwarz inequality", "math"},
		{"Explain CRISPR-Cas9 off-target effects and mitigation strategies", "biology"},

		// Expert level
		{"Design a distributed consensus algorithm tolerant to Byzantine failures", "computer science"},
		{"Prove the Riemann hypothesis implications for prime distribution", "math"},
		{"Describe the molecular mechanisms of long non-coding RNA in epigenetic regulation", "biology"},
	}

	t.Log("=== Mixed Difficulty Queries Test ===")
	t.Log("Testing beginner to expert level queries")

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			modelCounts := make(map[string]int)

			for _, tc := range testCases {
				embedding, err := trainer.GetEmbedding(tc.Query)
				if err != nil {
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err == nil && selected != nil {
					modelCounts[selected.Model]++
					t.Logf("  [%s] '%s' -> %s", tc.Category, truncateQuery(tc.Query, 50), selected.Model)
				}
			}

			t.Logf("%s distribution: %v", alg, modelCounts)
		})
	}
}

// TestComprehensiveStressTest runs a large-scale stress test with many queries
func TestComprehensiveStressTest(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	// Generate a comprehensive set of test queries across all categories
	allQueries := []TestCase{
		// Biology (10)
		{"Explain protein folding mechanisms", "biology"},
		{"Describe the Krebs cycle in detail", "biology"},
		{"What is homeostasis and how is it maintained?", "biology"},
		{"Explain horizontal gene transfer in bacteria", "biology"},
		{"Describe the mechanism of nerve impulse transmission", "biology"},
		{"What are stem cells and their therapeutic potential?", "biology"},
		{"Explain the role of telomeres in aging", "biology"},
		{"Describe the process of embryonic development", "biology"},
		{"What is the microbiome and its role in health?", "biology"},
		{"Explain the genetics of inherited diseases", "biology"},

		// Business (10)
		{"Explain lean manufacturing principles", "business"},
		{"Describe agile project management methodology", "business"},
		{"What is business process reengineering?", "business"},
		{"Explain the concept of disruptive innovation", "business"},
		{"Describe corporate governance best practices", "business"},
		{"What is the balanced scorecard framework?", "business"},
		{"Explain merger and acquisition strategies", "business"},
		{"Describe supply chain risk management", "business"},
		{"What is stakeholder management?", "business"},
		{"Explain business model canvas components", "business"},

		// Chemistry (10)
		{"Explain catalysis mechanisms in organic reactions", "chemistry"},
		{"Describe spectroscopic methods for structure determination", "chemistry"},
		{"What is stereochemistry and chirality?", "chemistry"},
		{"Explain thermodynamic versus kinetic control", "chemistry"},
		{"Describe polymer chemistry and polymerization", "chemistry"},
		{"What are supramolecular interactions?", "chemistry"},
		{"Explain green chemistry principles", "chemistry"},
		{"Describe electrochemistry fundamentals", "chemistry"},
		{"What is coordination chemistry?", "chemistry"},
		{"Explain solid-state chemistry concepts", "chemistry"},

		// Computer Science (10)
		{"Implement a bloom filter data structure", "computer science"},
		{"Write a skip list implementation", "computer science"},
		{"Design a load balancer algorithm", "computer science"},
		{"Implement a rate limiter using token bucket", "computer science"},
		{"Create a simple database query optimizer", "computer science"},
		{"Write a JSON parser from scratch", "computer science"},
		{"Implement a simple HTTP server", "computer science"},
		{"Design a URL shortener system", "computer science"},
		{"Write a concurrent web crawler", "computer science"},
		{"Implement a text search engine with TF-IDF", "computer science"},

		// Economics (10)
		{"Explain behavioral economics findings", "economics"},
		{"Describe international trade theory", "economics"},
		{"What is the quantity theory of money?", "economics"},
		{"Explain labor market economics", "economics"},
		{"Describe public choice theory", "economics"},
		{"What is the Keynesian multiplier effect?", "economics"},
		{"Explain development economics concepts", "economics"},
		{"Describe environmental economics principles", "economics"},
		{"What is mechanism design theory?", "economics"},
		{"Explain auction theory fundamentals", "economics"},

		// Engineering (10)
		{"Design a heat exchanger for industrial application", "engineering"},
		{"Explain embedded systems programming", "engineering"},
		{"Describe renewable energy systems design", "engineering"},
		{"What is computer-aided design methodology?", "engineering"},
		{"Explain robotics control systems", "engineering"},
		{"Describe telecommunications network design", "engineering"},
		{"What is biomedical engineering?", "engineering"},
		{"Explain environmental engineering principles", "engineering"},
		{"Describe manufacturing process optimization", "engineering"},
		{"What is systems engineering methodology?", "engineering"},

		// Health (10)
		{"Explain pharmacogenomics and personalized medicine", "health"},
		{"Describe the immune checkpoint therapy for cancer", "health"},
		{"What is regenerative medicine?", "health"},
		{"Explain telemedicine implementation challenges", "health"},
		{"Describe mental health treatment modalities", "health"},
		{"What is precision nutrition?", "health"},
		{"Explain global health challenges and solutions", "health"},
		{"Describe healthcare quality improvement methods", "health"},
		{"What is health economics and outcomes research?", "health"},
		{"Explain infectious disease epidemiology", "health"},

		// History (10)
		{"Analyze the causes of the Protestant Reformation", "history"},
		{"Describe the Age of Exploration and its impacts", "history"},
		{"What was the Enlightenment and its legacy?", "history"},
		{"Explain the history of the Ottoman Empire", "history"},
		{"Describe the Civil Rights Movement in America", "history"},
		{"What was the Renaissance and its achievements?", "history"},
		{"Explain the history of ancient Mesopotamia", "history"},
		{"Describe the Meiji Restoration in Japan", "history"},
		{"What was the Byzantine Empire's significance?", "history"},
		{"Explain the history of the British Empire", "history"},

		// Law (10)
		{"Explain antitrust law and competition policy", "law"},
		{"Describe environmental law frameworks", "law"},
		{"What is administrative law?", "law"},
		{"Explain employment discrimination law", "law"},
		{"Describe bankruptcy law procedures", "law"},
		{"What is securities regulation?", "law"},
		{"Explain family law principles", "law"},
		{"Describe data privacy regulations", "law"},
		{"What is immigration law?", "law"},
		{"Explain maritime law fundamentals", "law"},

		// Math (10)
		{"Explain topology fundamentals", "math"},
		{"Describe algebraic geometry basics", "math"},
		{"What is combinatorics and graph theory?", "math"},
		{"Explain differential geometry concepts", "math"},
		{"Describe numerical analysis methods", "math"},
		{"What is mathematical logic?", "math"},
		{"Explain category theory basics", "math"},
		{"Describe optimization theory", "math"},
		{"What is representation theory?", "math"},
		{"Explain stochastic processes", "math"},

		// Other (10)
		{"What are effective study techniques?", "other"},
		{"Explain creative writing principles", "other"},
		{"Describe leadership development strategies", "other"},
		{"What is critical thinking methodology?", "other"},
		{"Explain team building best practices", "other"},
		{"Describe personal finance management", "other"},
		{"What is emotional intelligence?", "other"},
		{"Explain negotiation strategies", "other"},
		{"Describe presentation skills development", "other"},
		{"What is design thinking methodology?", "other"},

		// Philosophy (10)
		{"Explain the philosophy of language", "philosophy"},
		{"Describe aesthetics and philosophy of art", "philosophy"},
		{"What is political philosophy?", "philosophy"},
		{"Explain the philosophy of religion", "philosophy"},
		{"Describe environmental ethics", "philosophy"},
		{"What is the philosophy of mathematics?", "philosophy"},
		{"Explain bioethics principles", "philosophy"},
		{"Describe feminist philosophy", "philosophy"},
		{"What is phenomenology?", "philosophy"},
		{"Explain pragmatism in philosophy", "philosophy"},

		// Physics (10)
		{"Explain condensed matter physics", "physics"},
		{"Describe nuclear physics fundamentals", "physics"},
		{"What is plasma physics?", "physics"},
		{"Explain astrophysics concepts", "physics"},
		{"Describe optical physics principles", "physics"},
		{"What is fluid dynamics?", "physics"},
		{"Explain acoustics and wave physics", "physics"},
		{"Describe semiconductor physics", "physics"},
		{"What is biophysics?", "physics"},
		{"Explain geophysics fundamentals", "physics"},

		// Psychology (10)
		{"Explain neuropsychology fundamentals", "psychology"},
		{"Describe positive psychology research", "psychology"},
		{"What is forensic psychology?", "psychology"},
		{"Explain industrial-organizational psychology", "psychology"},
		{"Describe health psychology interventions", "psychology"},
		{"What is cross-cultural psychology?", "psychology"},
		{"Explain sports psychology principles", "psychology"},
		{"Describe educational psychology applications", "psychology"},
		{"What is environmental psychology?", "psychology"},
		{"Explain evolutionary psychology theory", "psychology"},
	}

	t.Log("=== Comprehensive Stress Test ===")
	t.Logf("Testing %d queries across all 14 categories", len(allQueries))

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			categoryModelCounts := make(map[string]map[string]int)
			for _, cat := range VSRCategories {
				categoryModelCounts[cat] = make(map[string]int)
			}

			successCount := 0

			for _, tc := range allQueries {
				embedding, err := trainer.GetEmbedding(tc.Query)
				if err != nil {
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err == nil && selected != nil {
					successCount++
					categoryModelCounts[tc.Category][selected.Model]++
				}
			}

			// Summary statistics
			t.Logf("%s: %d/%d successful selections", alg, successCount, len(allQueries))

			// Overall model distribution
			overallCounts := make(map[string]int)
			for _, models := range categoryModelCounts {
				for model, count := range models {
					overallCounts[model] += count
				}
			}
			t.Logf("  Overall distribution: %v", overallCounts)

			// Check diversity
			if len(overallCounts) >= 2 {
				t.Logf("✓ %s used %d different models", alg, len(overallCounts))
			}
		})
	}
}

// ============================================================================
// Helper functions
// ============================================================================

// truncateQuery truncates a query string to the specified length
func truncateQuery(query string, maxLen int) string {
	if len(query) <= maxLen {
		return query
	}
	return query[:maxLen] + "..."
}

// runGeneralizationTest runs a generalization test for a specific category
func runGeneralizationTest(t *testing.T, trainer *Trainer, modelsPath string, testCases []TestCase, categoryName string) {
	t.Logf("=== Generalization Test: %s ===", categoryName)
	t.Logf("Testing %d NEW queries (not in training data)", len(testCases))

	modelRefs := getDefaultModelRefs()
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			modelCounts := make(map[string]int)

			for _, tc := range testCases {
				embedding, err := trainer.GetEmbedding(tc.Query)
				if err != nil {
					t.Errorf("Failed to get embedding: %v", err)
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tc.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("Selection error: %v", err)
					continue
				}

				if selected != nil {
					modelCounts[selected.Model]++

					queryPreview := tc.Query
					if len(queryPreview) > 50 {
						queryPreview = queryPreview[:50] + "..."
					}
					t.Logf("  '%s' -> %s", queryPreview, selected.Model)
				}
			}

			t.Logf("%s: %v", alg, modelCounts)
			uniqueModels := len(modelCounts)
			if uniqueModels >= 2 {
				t.Logf("✓ %s selected %d different models", alg, uniqueModels)
			}
		})
	}
}

// loadRealQwen3Embeddings loads training records with real Qwen3 embeddings from KNN model
func loadRealQwen3Embeddings(modelsPath string) ([]TrainingRecord, error) {
	knnPath := modelsPath + "/knn_model.json"
	data, err := os.ReadFile(knnPath)
	if err != nil {
		return nil, err
	}

	var model struct {
		Training []TrainingRecord `json:"training"`
	}
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, err
	}

	return model.Training, nil
}

// loadOptimalModelRecords loads only the BEST model record for each unique query
// This is used for accuracy testing - we evaluate if the algorithm selects the optimal model
func loadOptimalModelRecords(modelsPath string) ([]TrainingRecord, error) {
	allRecords, err := loadRealQwen3Embeddings(modelsPath)
	if err != nil {
		return nil, err
	}

	// Group by embedding (first 10 values as key) and keep highest quality
	type embKey [10]float64
	bestRecords := make(map[embKey]TrainingRecord)

	for _, rec := range allRecords {
		var key embKey
		for i := 0; i < 10 && i < len(rec.QueryEmbedding); i++ {
			key[i] = rec.QueryEmbedding[i]
		}

		if existing, ok := bestRecords[key]; ok {
			if rec.ResponseQuality > existing.ResponseQuality {
				bestRecords[key] = rec
			}
		} else {
			bestRecords[key] = rec
		}
	}

	// Convert to slice
	result := make([]TrainingRecord, 0, len(bestRecords))
	for _, rec := range bestRecords {
		result = append(result, rec)
	}

	return result, nil
}

// sumModelCounts returns the total count across all models
func sumModelCounts(m map[string]int) int {
	total := 0
	for _, v := range m {
		total += v
	}
	return total
}

// TestKNNVotingAnalysis analyzes why KNN votes the way it does
func TestKNNVotingAnalysis(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	t.Log("=== KNN Voting Analysis ===")

	// Load pre-trained KNN model
	selector, err := loadPretrainedSelectorFromPath("knn", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available: %v", err)
	}

	knn := selector.(*KNNSelector)
	training := knn.getTrainingData()
	t.Logf("Training data: %d records", len(training))

	// Analyze the training data
	t.Log("\n=== Training Data Analysis ===")
	modelStats := make(map[string]struct {
		count        int
		avgQuality   float64
		avgLatencyMs float64
	})

	for _, rec := range training {
		stats := modelStats[rec.SelectedModel]
		stats.count++
		stats.avgQuality += rec.ResponseQuality
		stats.avgLatencyMs += float64(rec.ResponseLatency().Milliseconds())
		modelStats[rec.SelectedModel] = stats
	}

	for model, stats := range modelStats {
		avgQ := stats.avgQuality / float64(stats.count)
		avgL := stats.avgLatencyMs / float64(stats.count)
		t.Logf("  %s: Count=%d, AvgQuality=%.3f, AvgLatencyMs=%.0f", model, stats.count, avgQ, avgL)
	}

	// Test with one query and trace the voting
	testQuery := "What is 2+2?"
	t.Logf("\n=== Test Query: '%s' ===", testQuery)

	embedding, err := trainer.GetEmbedding(testQuery)
	if err != nil {
		t.Fatalf("Failed to get embedding: %v", err)
	}
	t.Logf("Raw embedding length: %d", len(embedding))

	featureVec := CombineEmbeddingWithCategory(embedding, "math")
	t.Logf("Feature vector length (with category): %d", len(featureVec))
	t.Logf("Training embedding length: %d", len(training[0].QueryEmbedding))
	t.Logf("Test first 5 values: %.4f, %.4f, %.4f, %.4f, %.4f",
		featureVec[0], featureVec[1], featureVec[2], featureVec[3], featureVec[4])
	t.Logf("Train first 5 values: %.4f, %.4f, %.4f, %.4f, %.4f",
		training[0].QueryEmbedding[0], training[0].QueryEmbedding[1],
		training[0].QueryEmbedding[2], training[0].QueryEmbedding[3], training[0].QueryEmbedding[4])

	// Find 5 nearest neighbors manually
	type neighbor struct {
		model      string
		similarity float64
		quality    float64
		latencyMs  float64
	}
	var neighbors []neighbor

	for _, rec := range training {
		sim := CosineSimilarity(featureVec, rec.QueryEmbedding)
		neighbors = append(neighbors, neighbor{
			model:      rec.SelectedModel,
			similarity: sim,
			quality:    rec.ResponseQuality,
			latencyMs:  float64(rec.ResponseLatency().Milliseconds()),
		})
	}

	// Sort by similarity
	for i := 0; i < len(neighbors); i++ {
		for j := i + 1; j < len(neighbors); j++ {
			if neighbors[j].similarity > neighbors[i].similarity {
				neighbors[i], neighbors[j] = neighbors[j], neighbors[i]
			}
		}
	}

	// Take top 5
	t.Log("\n=== Top 5 Nearest Neighbors ===")
	maxLatency := 0.0
	for i := 0; i < 5 && i < len(neighbors); i++ {
		if neighbors[i].latencyMs > maxLatency {
			maxLatency = neighbors[i].latencyMs
		}
	}

	votes := make(map[string]float64)
	for i := 0; i < 5 && i < len(neighbors); i++ {
		n := neighbors[i]
		// Same formula as KNN selector
		weight := n.similarity
		weight *= (1.0 + n.quality)
		normalizedLatency := n.latencyMs / maxLatency
		efficiencyBonus := 1.0 / (1.0 + normalizedLatency)
		weight *= efficiencyBonus

		votes[n.model] += weight
		t.Logf("  [%d] Model=%s, Sim=%.3f, Quality=%.2f, LatencyMs=%.0f, Efficiency=%.3f, Vote=%.4f",
			i+1, n.model, n.similarity, n.quality, n.latencyMs, efficiencyBonus, weight)
	}

	t.Log("\n=== Final Votes ===")
	for model, vote := range votes {
		t.Logf("  %s: %.4f", model, vote)
	}

	// Find winner
	winner := ""
	maxVote := 0.0
	for model, vote := range votes {
		if vote > maxVote {
			maxVote = vote
			winner = model
		}
	}
	t.Logf("\n=== Winner: %s (vote=%.4f) ===", winner, maxVote)
}

// ============================================================================
// Test with BEST model per query training (correct approach)
// ============================================================================

// TestTrainWithBestModelPerQuery trains algorithms with ONLY the best model per query
// This is the correct training approach - each query maps to ONE best model
func TestTrainWithBestModelPerQuery(t *testing.T) {
	trainer := initLoadTestTrainer(t)

	t.Log("=== Training with BEST Model Per Query ===")
	t.Log("Loading benchmark data and selecting BEST model for each query")

	// Load all benchmark training data
	dataPath := "data/benchmark_training_data.jsonl"
	file, err := os.Open(dataPath)
	if err != nil {
		t.Skipf("Skipping: could not load benchmark data: %v", err)
	}
	defer file.Close()

	// Parse all records
	type BenchmarkRecord struct {
		EmbeddingID  int     `json:"embedding_id"`
		Query        string  `json:"query"`
		Category     string  `json:"category"`
		ModelName    string  `json:"model_name"`
		Performance  float64 `json:"performance"`
		ResponseTime float64 `json:"response_time"`
	}

	var allRecords []BenchmarkRecord
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		var rec BenchmarkRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue
		}
		allRecords = append(allRecords, rec)
	}
	t.Logf("Loaded %d total records", len(allRecords))

	// Group by embedding_id and find BEST model for each query
	type QueryGroup struct {
		Query    string
		Category string
		Records  []BenchmarkRecord
	}
	queryGroups := make(map[int]*QueryGroup)

	for _, rec := range allRecords {
		if queryGroups[rec.EmbeddingID] == nil {
			queryGroups[rec.EmbeddingID] = &QueryGroup{
				Query:    rec.Query,
				Category: rec.Category,
			}
		}
		queryGroups[rec.EmbeddingID].Records = append(queryGroups[rec.EmbeddingID].Records, rec)
	}
	t.Logf("Found %d unique queries", len(queryGroups))

	// Create training data with ONLY the best model per query
	var trainingRecords []TrainingRecord
	bestModelCounts := make(map[string]int)

	for _, qg := range queryGroups {
		// Find best performing model for this query
		var bestRec BenchmarkRecord
		bestPerf := -1.0
		for _, rec := range qg.Records {
			if rec.Performance > bestPerf {
				bestPerf = rec.Performance
				bestRec = rec
			}
		}

		if bestRec.ModelName == "" {
			continue
		}

		bestModelCounts[bestRec.ModelName]++

		// Generate embedding for this query
		embedding, err := trainer.GetEmbedding(qg.Query)
		if err != nil {
			continue
		}

		featureVec := CombineEmbeddingWithCategory(embedding, qg.Category)
		trainingRecords = append(trainingRecords, TrainingRecord{
			QueryEmbedding:  featureVec,
			SelectedModel:   bestRec.ModelName,
			ResponseQuality: bestRec.Performance,
			Success:         true,
		})
	}

	t.Log("\n=== Best Model Distribution (Ground Truth) ===")
	for model, count := range bestModelCounts {
		pct := float64(count) / float64(len(queryGroups)) * 100
		t.Logf("  %s: %d queries (%.1f%%)", model, count, pct)
	}

	// Split into train (80%) and test (20%)
	splitIdx := len(trainingRecords) * 80 / 100
	trainData := trainingRecords[:splitIdx]
	testData := trainingRecords[splitIdx:]

	t.Logf("\nTrain: %d, Test: %d", len(trainData), len(testData))

	// Model refs for the 4 trained LLMs
	modelRefs := getDefaultModelRefs()

	// Train and test each algorithm
	algorithms := []struct {
		name     string
		selector Selector
	}{
		{"knn", NewKNNSelector(5)},
		{"kmeans", NewKMeansSelector(8)},
		{"mlp", NewMLPSelector([]int{64, 32})},
		{"svm", NewSVMSelector("rbf")},
		{"mf", NewMatrixFactorizationSelector(16)},
	}

	t.Log("\n=== Training and Evaluation Results ===")

	for _, alg := range algorithms {
		t.Run(alg.name, func(t *testing.T) {
			// Train on BEST model per query data
			err := alg.selector.Train(trainData)
			if err != nil {
				t.Fatalf("Training failed: %v", err)
			}

			// Test
			correctCount := 0
			modelCounts := make(map[string]int)

			for _, testRec := range testData {
				ctx := &SelectionContext{
					QueryEmbedding: testRec.QueryEmbedding,
				}

				selected, err := alg.selector.Select(ctx, modelRefs)
				if err != nil {
					continue
				}

				if selected != nil {
					modelCounts[selected.Model]++
					if selected.Model == testRec.SelectedModel {
						correctCount++
					}
				}
			}

			accuracy := float64(correctCount) / float64(len(testData)) * 100.0
			t.Logf("%s: Accuracy=%.1f%% (%d/%d correct)", alg.name, accuracy, correctCount, len(testData))
			t.Logf("  Model distribution: %v", modelCounts)

			// Check if algorithm uses multiple models
			if len(modelCounts) >= 2 {
				t.Logf("✓ %s selected %d different models", alg.name, len(modelCounts))
			}
		})
	}
}

// ============================================================================
// Benchmark Tests with Real Data (llmrouter_training_data_with_category.jsonl)
// Uses 4 public LLMs: mistral-7b, gemma-2-9b, llama-3.1-8b, qwen2.5-7b
// ============================================================================

// RealBenchmarkRecord represents a record from llmrouter_training_data_with_category.jsonl
type RealBenchmarkRecord struct {
	TaskName     string  `json:"task_name"`
	Query        string  `json:"query"`
	GroundTruth  string  `json:"ground_truth"`
	ModelName    string  `json:"model_name"`
	Response     string  `json:"response"`
	Performance  float64 `json:"performance"`
	ResponseTime float64 `json:"response_time"`
	Category     string  `json:"category"`
	EmbeddingID  int     `json:"embedding_id"`
}

// The 4 public LLMs we use for benchmark testing (no API key required)
var benchmarkModels = []string{
	"mistral-7b-instruct-v0.3",
	"gemma-2-9b-it",
	"llama-3.1-8b-instruct",
	"qwen2.5-7b-instruct",
}

// loadRealBenchmarkData loads records from llmrouter_training_data_with_category.jsonl
// Filters to only include the 4 benchmark models
func loadRealBenchmarkData(maxRecords int) ([]RealBenchmarkRecord, error) {
	dataPath := "data/llmrouter_training_data_with_category.jsonl"
	file, err := os.Open(dataPath)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	modelSet := make(map[string]bool)
	for _, m := range benchmarkModels {
		modelSet[m] = true
	}

	var records []RealBenchmarkRecord
	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024) // 1MB buffer
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() && len(records) < maxRecords {
		var rec RealBenchmarkRecord
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue
		}
		// Only include our 4 benchmark models
		if modelSet[rec.ModelName] {
			records = append(records, rec)
		}
	}

	return records, scanner.Err()
}

// groupRecordsByQuery groups benchmark records by embedding_id (same query)
// Returns map of embedding_id -> records for all models
func groupRecordsByQuery(records []RealBenchmarkRecord) map[int][]RealBenchmarkRecord {
	grouped := make(map[int][]RealBenchmarkRecord)
	for _, rec := range records {
		grouped[rec.EmbeddingID] = append(grouped[rec.EmbeddingID], rec)
	}
	return grouped
}

// findBestModel finds the best performing model for a query
func findBestModel(records []RealBenchmarkRecord) (string, float64) {
	bestModel := ""
	bestScore := -1.0
	for _, rec := range records {
		if rec.Performance > bestScore {
			bestScore = rec.Performance
			bestModel = rec.ModelName
		}
	}
	return bestModel, bestScore
}

// TestBenchmarkWithRealData tests model selection using real benchmark data
// Loads pre-trained models from disk
// Generates NEW embeddings for test queries using Qwen3
// Tests on queries NOT in training data
func TestBenchmarkWithRealData(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	t.Log("=== Benchmark Test with Real Data ===")
	t.Log("Loading pre-trained models from disk")
	t.Log("Generating NEW embeddings for test queries using Qwen3")
	t.Log("Testing on queries NOT in training data")

	// Load real benchmark data for test queries
	records, err := loadRealBenchmarkData(30000)
	if err != nil {
		t.Skipf("Skipping: could not load benchmark data: %v", err)
	}
	t.Logf("Loaded %d benchmark records", len(records))

	// Group by query (embedding_id)
	grouped := groupRecordsByQuery(records)
	t.Logf("Found %d unique queries", len(grouped))

	// Load training data to get context
	trainingRecords, err := loadRealQwen3Embeddings(modelsPath)
	if err != nil {
		t.Skipf("Skipping: could not load training data: %v", err)
	}
	t.Logf("Training data has %d records", len(trainingRecords))

	// Create test data from NEW queries using real benchmark data
	type TestQueryData struct {
		Query    string
		Category string
	}
	var testQueries []TestQueryData

	for _, recs := range grouped {
		if len(recs) < 4 {
			continue
		}

		testQueries = append(testQueries, TestQueryData{
			Query:    recs[0].Query,
			Category: recs[0].Category,
		})

		// Limit test queries for reasonable test time
		if len(testQueries) >= 200 {
			break
		}
	}

	t.Logf("Test queries: %d (NEW queries NOT in original training)", len(testQueries))

	if len(testQueries) < 50 {
		t.Skipf("Not enough test queries (%d)", len(testQueries))
	}

	// Model refs matching what the pre-trained models were trained on
	modelRefs := getDefaultModelRefs()
	t.Log("Pre-trained models were trained on: llama-3.2-1b, llama-3.2-3b, codellama-7b, mistral-7b")

	// Test each algorithm with pre-trained models
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	t.Log("\n=== Evaluation Results (Pre-trained Models) ===")

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			// Load pre-trained model from disk
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}

			// Test with NEW embeddings generated by Qwen3
			modelCounts := make(map[string]int)
			categoryModelCounts := make(map[string]map[string]int)

			for _, tq := range testQueries {
				// Generate NEW embedding using Qwen3
				embedding, err := trainer.GetEmbedding(tq.Query)
				if err != nil {
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, tq.Category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					continue
				}

				if selected != nil {
					modelCounts[selected.Model]++
					if categoryModelCounts[tq.Category] == nil {
						categoryModelCounts[tq.Category] = make(map[string]int)
					}
					categoryModelCounts[tq.Category][selected.Model]++
				}
			}

			t.Logf("%s: Total model distribution: %v", alg, modelCounts)

			// Show per-category breakdown
			for cat, counts := range categoryModelCounts {
				t.Logf("  %s: %v", cat, counts)
			}

			// Check if algorithm uses multiple models
			if len(modelCounts) >= 2 {
				t.Logf("✓ %s selected %d different models", alg, len(modelCounts))
			} else if len(modelCounts) == 1 {
				t.Logf("⚠ %s selected only 1 model", alg)
			}
		})
	}
}

// TestBenchmarkPerCategory tests model selection accuracy per category
// Loads pre-trained models from disk
// Generates NEW embeddings for test queries using Qwen3
// Tests on NEW queries NOT in training data
func TestBenchmarkPerCategory(t *testing.T) {
	modelsPath := "data/trained_models"
	trainer := initLoadTestTrainer(t)

	t.Log("=== Benchmark Per Category ===")
	t.Log("Loading pre-trained models from disk")
	t.Log("Testing with NEW queries NOT in training data")

	// Load real benchmark data for ground truth analysis
	records, err := loadRealBenchmarkData(30000)
	if err != nil {
		t.Skipf("Skipping: could not load benchmark data: %v", err)
	}

	// Group by category, then by query
	categoryRecords := make(map[string]map[int][]RealBenchmarkRecord)
	for _, rec := range records {
		if categoryRecords[rec.Category] == nil {
			categoryRecords[rec.Category] = make(map[int][]RealBenchmarkRecord)
		}
		categoryRecords[rec.Category][rec.EmbeddingID] = append(categoryRecords[rec.Category][rec.EmbeddingID], rec)
	}

	// Analyze best model per category (ground truth)
	t.Log("\n=== Ground Truth: Best Model per Category ===")
	categoryBestModels := make(map[string]map[string]int)

	for category, queries := range categoryRecords {
		categoryBestModels[category] = make(map[string]int)
		for _, recs := range queries {
			if len(recs) >= 4 {
				bestModel, _ := findBestModel(recs)
				if bestModel != "" {
					categoryBestModels[category][bestModel]++
				}
			}
		}
		if len(categoryBestModels[category]) > 0 {
			t.Logf("  %s: %v", category, categoryBestModels[category])
		}
	}

	// Test categories with enough data
	testCategories := []string{"math", "other", "biology", "physics"}

	// Model refs matching what the pre-trained models were trained on
	modelRefs := getDefaultModelRefs()

	for _, category := range testCategories {
		t.Run(category, func(t *testing.T) {
			// Load pre-trained KNN model from disk
			selector, err := loadPretrainedSelectorFromPath("knn", modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available: %v", err)
			}
			t.Logf("Loaded pre-trained KNN model for category: %s", category)

			// Test with NEW queries NOT in training data
			modelCounts := make(map[string]int)
			testQueries := []string{
				"Solve the equation x^2 + 3x + 2 = 0",
				"What is the derivative of sin(x)?",
				"Calculate 15% of 240",
				"Find the area of a circle with radius 5",
				"What is the probability of rolling a 6?",
			}

			switch category {
			case "other":
				testQueries = []string{
					"What is the capital of France?",
					"Who wrote Romeo and Juliet?",
					"What year did World War 2 end?",
					"How many continents are there?",
					"What is the largest ocean?",
				}
			case "biology":
				testQueries = []string{
					"What is photosynthesis?",
					"How do cells divide?",
					"What is DNA?",
					"Explain the circulatory system",
					"What are enzymes?",
				}
			case "physics":
				testQueries = []string{
					"What is Newton's first law?",
					"Explain the theory of relativity",
					"What is kinetic energy?",
					"How does gravity work?",
					"What is the speed of light?",
				}
			}

			for _, query := range testQueries {
				// Generate NEW embedding using Qwen3
				embedding, err := trainer.GetEmbedding(query)
				if err != nil {
					continue
				}

				featureVec := CombineEmbeddingWithCategory(embedding, category)
				ctx := &SelectionContext{
					QueryEmbedding: featureVec,
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err == nil && selected != nil {
					modelCounts[selected.Model]++
					t.Logf("  '%s' -> %s", truncateQuery(query, 40), selected.Model)
				}
			}

			t.Logf("KNN selections for %s: %v", category, modelCounts)

			// Compare with ground truth
			if gtBest := categoryBestModels[category]; len(gtBest) > 0 {
				// Find most common ground truth model
				bestGT := ""
				bestCount := 0
				for m, c := range gtBest {
					if c > bestCount {
						bestCount = c
						bestGT = m
					}
				}
				t.Logf("Ground truth best model for %s: %s", category, bestGT)
			}
		})
	}
}

// TestBenchmarkModelAccuracyAnalysis analyzes real model performance from benchmark data
func TestBenchmarkModelAccuracyAnalysis(t *testing.T) {
	t.Log("=== Real Benchmark Data Analysis ===")
	t.Log("Analyzing actual model performance from llmrouter_training_data_with_category.jsonl")

	records, err := loadRealBenchmarkData(50000)
	if err != nil {
		t.Skipf("Skipping: could not load benchmark data: %v", err)
	}

	t.Logf("Total records loaded: %d", len(records))

	// Analyze by model
	modelStats := make(map[string]struct {
		total     int
		correct   int
		totalTime float64
	})

	for _, rec := range records {
		stats := modelStats[rec.ModelName]
		stats.total++
		if rec.Performance > 0 {
			stats.correct++
		}
		stats.totalTime += rec.ResponseTime
		modelStats[rec.ModelName] = stats
	}

	t.Log("\n=== Model Performance Summary ===")
	for model, stats := range modelStats {
		accuracy := float64(stats.correct) / float64(stats.total) * 100.0
		avgTime := stats.totalTime / float64(stats.total)
		t.Logf("%s: Accuracy=%.1f%%, AvgTime=%.2fs, N=%d",
			model, accuracy, avgTime, stats.total)
	}

	// Analyze by category
	categoryStats := make(map[string]map[string]struct {
		total   int
		correct int
	})

	for _, rec := range records {
		if categoryStats[rec.Category] == nil {
			categoryStats[rec.Category] = make(map[string]struct {
				total   int
				correct int
			})
		}
		stats := categoryStats[rec.Category][rec.ModelName]
		stats.total++
		if rec.Performance > 0 {
			stats.correct++
		}
		categoryStats[rec.Category][rec.ModelName] = stats
	}

	t.Log("\n=== Best Model per Category ===")
	for category, models := range categoryStats {
		if len(models) < 2 {
			continue
		}

		bestModel := ""
		bestAccuracy := 0.0
		for model, stats := range models {
			if stats.total >= 5 {
				accuracy := float64(stats.correct) / float64(stats.total)
				if accuracy > bestAccuracy {
					bestAccuracy = accuracy
					bestModel = model
				}
			}
		}
		if bestModel != "" {
			t.Logf("  %s: Best=%s (%.1f%% accuracy)", category, bestModel, bestAccuracy*100)
		}
	}
}
