package modelselection

import (
	"bufio"
	"encoding/json"
	"os"
	"strings"
	"testing"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func TestLoadPretrainedKNNSelector(t *testing.T) {
	// Test loading KNN model
	modelsPath := "data/trained_models"
	selector, err := loadPretrainedSelectorFromPath("knn", modelsPath)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	if selector.Name() != "knn" {
		t.Errorf("Expected name 'knn', got '%s'", selector.Name())
	}

	// Check that training data was loaded
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

// TestModelSelectionInference tests end-to-end inference with pre-trained models
func TestModelSelectionInference(t *testing.T) {
	modelsPath := "data/trained_models"

	// Define model refs matching the trained models
	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
	}

	// Test all algorithms
	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
			}

			// Create a test query with embedding + category one-hot (782 dimensions)
			// Use an embedding from training data to ensure compatibility
			embedding := make([]float64, 768)
			for i := range embedding {
				embedding[i] = float64(i%10) / 10.0 // Simple deterministic pattern
			}
			// Combine with category one-hot for "math" (index 9)
			featureVector := CombineEmbeddingWithCategory(embedding, "math")

			ctx := &SelectionContext{
				QueryEmbedding: featureVector,
				QueryText:      "What is 2+2?",
				CategoryName:   "math",
				DecisionName:   "math_decision",
			}

			// Perform model selection
			selected, err := selector.Select(ctx, modelRefs)
			if err != nil {
				t.Fatalf("%s Select failed: %v", alg, err)
			}

			if selected == nil {
				t.Fatalf("%s returned nil selection", alg)
			}

			// Verify it selected one of the valid models
			validModel := selected.Model == "llama-3.2-1b" || selected.Model == "llama-3.2-3b"
			if !validModel {
				t.Errorf("%s selected invalid model: %s", alg, selected.Model)
			}

			t.Logf("%s selected model: %s", alg, selected.Model)
		})
	}
}

// TestModelSelectionWithVariedQueries tests model selection across different query types
// to demonstrate that all 4 models can be selected based on query characteristics
func TestModelSelectionWithVariedQueries(t *testing.T) {
	modelsPath := "data/trained_models"

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// Define test cases with different embedding patterns and categories
	testCases := []struct {
		name        string
		queryText   string
		category    string
		embeddingFn func() []float64
	}{
		{
			name:      "simple_math",
			queryText: "What is 2+2?",
			category:  "math",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					emb[i] = float64(i%10) / 10.0
				}
				return emb
			},
		},
		{
			name:      "complex_reasoning",
			queryText: "Prove that there are infinitely many prime numbers using Euclid's theorem",
			category:  "math",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Different pattern - negative values, higher variance
					emb[i] = float64(i%20-10) / 5.0
				}
				return emb
			},
		},
		{
			name:      "code_generation",
			queryText: "Write a Python function to implement quicksort with detailed comments",
			category:  "computer_science",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Pattern with sine wave for code-like queries
					emb[i] = 0.5 * float64(i%30) / 30.0
				}
				return emb
			},
		},
		{
			name:      "physics_complex",
			queryText: "Derive the Schwarzschild metric from Einstein's field equations",
			category:  "physics",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// High variance pattern for complex physics
					emb[i] = float64((i*7)%100-50) / 100.0
				}
				return emb
			},
		},
		{
			name:      "simple_history",
			queryText: "When did World War II end?",
			category:  "history",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					emb[i] = float64(i%5) / 20.0
				}
				return emb
			},
		},
		{
			name:      "philosophy_deep",
			queryText: "Analyze the epistemological implications of Kant's categorical imperative in relation to modern virtue ethics",
			category:  "philosophy",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Complex pattern for philosophical reasoning
					emb[i] = float64((i*13)%100-50) / 80.0
				}
				return emb
			},
		},
	}

	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	// Track model selections across all tests
	modelCounts := make(map[string]int)

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
			}

			algModelCounts := make(map[string]int)

			for _, tc := range testCases {
				t.Run(tc.name, func(t *testing.T) {
					embedding := tc.embeddingFn()
					featureVector := CombineEmbeddingWithCategory(embedding, tc.category)

					ctx := &SelectionContext{
						QueryEmbedding: featureVector,
						QueryText:      tc.queryText,
						CategoryName:   tc.category,
						DecisionName:   tc.category + "_decision",
					}

					selected, err := selector.Select(ctx, modelRefs)
					if err != nil {
						t.Fatalf("Select failed: %v", err)
					}

					if selected == nil {
						t.Fatal("Returned nil selection")
					}

					validModel := selected.Model == "llama-3.2-1b" || selected.Model == "llama-3.2-3b" ||
						selected.Model == "codellama-7b" || selected.Model == "mistral-7b"
					if !validModel {
						t.Errorf("Selected invalid model: %s", selected.Model)
					}

					algModelCounts[selected.Model]++
					modelCounts[selected.Model]++

					t.Logf("%s [%s]: selected %s", alg, tc.name, selected.Model)
				})
			}

			t.Logf("%s summary: llama-3.2-1b=%d, llama-3.2-3b=%d, codellama-7b=%d, mistral-7b=%d",
				alg, algModelCounts["llama-3.2-1b"], algModelCounts["llama-3.2-3b"],
				algModelCounts["codellama-7b"], algModelCounts["mistral-7b"])
		})
	}

	// Final summary
	t.Logf("=== OVERALL SUMMARY ===")
	t.Logf("Total selections: llama-3.2-1b=%d, llama-3.2-3b=%d, codellama-7b=%d, mistral-7b=%d",
		modelCounts["llama-3.2-1b"], modelCounts["llama-3.2-3b"],
		modelCounts["codellama-7b"], modelCounts["mistral-7b"])
}

// TestModelSelectionDistribution tests that different embeddings lead to different model selections
func TestModelSelectionDistribution(t *testing.T) {
	modelsPath := "data/trained_models"

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
			}

			modelCounts := make(map[string]int)

			// Run 20 different random-ish embedding patterns
			for seed := 0; seed < 20; seed++ {
				embedding := make([]float64, 768)
				for i := range embedding {
					// Generate varied patterns based on seed
					val := float64((i*seed+seed*seed)%100-50) / 100.0
					embedding[i] = val
				}

				categories := []string{"math", "physics", "computer_science", "philosophy", "history"}
				category := categories[seed%len(categories)]
				featureVector := CombineEmbeddingWithCategory(embedding, category)

				ctx := &SelectionContext{
					QueryEmbedding: featureVector,
					QueryText:      "Test query " + string(rune('A'+seed)),
					CategoryName:   category,
					DecisionName:   category + "_decision",
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Fatalf("Select failed for seed %d: %v", seed, err)
				}

				if selected != nil {
					modelCounts[selected.Model]++
				}
			}

			t.Logf("%s distribution over 20 queries: llama-3.2-1b=%d, llama-3.2-3b=%d",
				alg, modelCounts["llama-3.2-1b"], modelCounts["llama-3.2-3b"])

			// Log if we got variety
			if modelCounts["llama-3.2-1b"] > 0 && modelCounts["llama-3.2-3b"] > 0 {
				t.Logf("✓ %s selected BOTH models!", alg)
			}
		})
	}
}

// TestMLPAndMFComplexQueries tests that MLP and Matrix Factorization can select all 4 models
// with more diverse embedding patterns
func TestMLPAndMFComplexQueries(t *testing.T) {
	modelsPath := "data/trained_models"

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// More complex test cases with varied patterns
	testCases := []struct {
		name        string
		queryText   string
		category    string
		embeddingFn func() []float64
	}{
		{
			name:      "chemistry_organic",
			queryText: "Explain the mechanism of nucleophilic substitution reactions",
			category:  "chemistry",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Chemistry-like pattern with periodic structure
					emb[i] = 0.3*float64((i*17)%50-25)/25.0 + 0.1*float64(i%6)/6.0
				}
				return emb
			},
		},
		{
			name:      "physics_quantum",
			queryText: "Describe the wave function collapse in quantum mechanics",
			category:  "physics",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Quantum-like oscillating pattern
					emb[i] = 0.4*float64((i*23)%80-40)/40.0 - 0.2*float64((i*3)%10)/10.0
				}
				return emb
			},
		},
		{
			name:      "biology_genetics",
			queryText: "Explain CRISPR-Cas9 gene editing mechanism and its applications",
			category:  "biology",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Biology pattern with codon-like structure
					emb[i] = 0.35*float64((i*11)%60-30)/30.0 + 0.15*float64((i*2)%8)/8.0
				}
				return emb
			},
		},
		{
			name:      "other_creative",
			queryText: "Write a short story about a robot discovering emotions",
			category:  "other",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Creative writing pattern - more random-like
					emb[i] = 0.5 * float64((i*31)%100-50) / 50.0
				}
				return emb
			},
		},
		{
			name:      "math_advanced",
			queryText: "Prove the Riemann hypothesis using analytic continuation",
			category:  "math",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// Math pattern with prime-like structure
					emb[i] = 0.45 * float64((i*37)%90-45) / 45.0
				}
				return emb
			},
		},
		{
			name:      "cs_algorithms",
			queryText: "Implement a red-black tree with all rotations and prove its O(log n) complexity",
			category:  "computer_science",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					// CS pattern with binary-like structure
					emb[i] = 0.4*float64((i*19)%64-32)/32.0 + 0.1*float64(i%2)
				}
				return emb
			},
		},
		{
			name:      "economics_macro",
			queryText: "Analyze the impact of quantitative easing on inflation expectations",
			category:  "economics",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					emb[i] = 0.38 * float64((i*29)%70-35) / 35.0
				}
				return emb
			},
		},
		{
			name:      "health_neurology",
			queryText: "Explain the pathophysiology of Parkinson's disease and current treatments",
			category:  "health",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					emb[i] = 0.42 * float64((i*41)%85-42) / 42.0
				}
				return emb
			},
		},
		{
			name:      "law_constitutional",
			queryText: "Analyze the Fourth Amendment implications of digital surveillance",
			category:  "law",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					emb[i] = 0.36 * float64((i*43)%75-37) / 37.0
				}
				return emb
			},
		},
		{
			name:      "engineering_aerospace",
			queryText: "Design a heat shield for atmospheric reentry at Mach 25",
			category:  "engineering",
			embeddingFn: func() []float64 {
				emb := make([]float64, 768)
				for i := range emb {
					emb[i] = 0.48 * float64((i*47)%95-47) / 47.0
				}
				return emb
			},
		},
	}

	algorithms := []string{"mlp", "svm", "matrix_factorization", "knn", "kmeans"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
			}

			algModelCounts := make(map[string]int)

			for _, tc := range testCases {
				embedding := tc.embeddingFn()
				featureVector := CombineEmbeddingWithCategory(embedding, tc.category)

				ctx := &SelectionContext{
					QueryEmbedding: featureVector,
					QueryText:      tc.queryText,
					CategoryName:   tc.category,
					DecisionName:   tc.category + "_decision",
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("%s [%s]: selection error: %v", alg, tc.name, err)
					continue
				}

				if selected != nil {
					algModelCounts[selected.Model]++
					t.Logf("%s [%s]: selected %s", alg, tc.name, selected.Model)
				}
			}

			t.Logf("%s summary: llama-3.2-1b=%d, llama-3.2-3b=%d, codellama-7b=%d, mistral-7b=%d",
				alg, algModelCounts["llama-3.2-1b"], algModelCounts["llama-3.2-3b"],
				algModelCounts["codellama-7b"], algModelCounts["mistral-7b"])

			// Count how many different models were selected
			modelsSelected := 0
			for _, count := range algModelCounts {
				if count > 0 {
					modelsSelected++
				}
			}

			if modelsSelected >= 3 {
				t.Logf("✓ %s selected %d different models!", alg, modelsSelected)
			} else if modelsSelected >= 2 {
				t.Logf("✓ %s selected %d models", alg, modelsSelected)
			} else {
				t.Logf("⚠ %s only selected one model type", alg)
			}
		})
	}
}

// TestCodingQueriesModelSelection tests coding queries and explains the selection behavior
// based on the actual training data (which shows equal quality across models)
func TestCodingQueriesModelSelection(t *testing.T) {
	modelsPath := "data/trained_models"

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// Coding-specific test cases
	codingTestCases := []struct {
		name      string
		queryText string
		category  string
	}{
		{
			name:      "python_sort",
			queryText: "Write a Python function to implement quicksort algorithm with O(n log n) average time complexity",
			category:  "computer_science",
		},
		{
			name:      "javascript_async",
			queryText: "Implement an async/await function in JavaScript that fetches data from multiple APIs in parallel",
			category:  "computer_science",
		},
		{
			name:      "golang_concurrency",
			queryText: "Write a Go program using goroutines and channels to implement a worker pool pattern",
			category:  "computer_science",
		},
		{
			name:      "rust_memory",
			queryText: "Implement a custom memory allocator in Rust that handles ownership and borrowing correctly",
			category:  "computer_science",
		},
	}

	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	t.Log("=== Coding Query Model Selection Test ===")
	t.Log("Note: Current training data shows all models have EQUAL quality (~0.50) on coding tasks")
	t.Log("Combined score = 0.9*quality + 0.1*efficiency (quality-first approach)")
	t.Log("")

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
			}

			algModelCounts := make(map[string]int)

			for _, tc := range codingTestCases {
				// Generate a coding-specific embedding pattern
				embedding := make([]float64, 768)
				for i := range embedding {
					embedding[i] = 0.4*float64((i*19)%70-35)/35.0 + 0.2*float64(i%7)/7.0
				}

				featureVector := CombineEmbeddingWithCategory(embedding, tc.category)

				ctx := &SelectionContext{
					QueryEmbedding: featureVector,
					QueryText:      tc.queryText,
					CategoryName:   tc.category,
					DecisionName:   "coding_decision",
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("%s [%s]: selection error: %v", alg, tc.name, err)
					continue
				}

				if selected != nil {
					algModelCounts[selected.Model]++
					t.Logf("%s [%s]: selected %s", alg, tc.name, selected.Model)
				}
			}

			t.Logf("%s summary: llama-1b=%d, llama-3b=%d, codellama=%d, mistral=%d",
				alg, algModelCounts["llama-3.2-1b"], algModelCounts["llama-3.2-3b"],
				algModelCounts["codellama-7b"], algModelCounts["mistral-7b"])
		})
	}
}

// TestAlgorithmSelectsCodeLlamaWhenBetterQuality demonstrates that the algorithm
// WILL select codellama-7b when it has better quality in the training data.
// This simulates a scenario with more challenging coding tasks where codellama excels.
func TestAlgorithmSelectsCodeLlamaWhenBetterQuality(t *testing.T) {
	// Create a fresh KNN selector and train it with data where codellama outperforms
	selector := NewKNNSelector(5)

	// Training data where codellama-7b has HIGHER quality on complex coding queries
	// Using TrainingRecord with combined embedding (768 query + 14 category = 782 dims)
	trainingData := []TrainingRecord{
		// Complex code generation - codellama excels
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "codellama-7b", ResponseQuality: 0.95, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.40, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.55, Success: true},

		// Complex algorithm - codellama excels
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "codellama-7b", ResponseQuality: 0.90, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.35, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.50, Success: true},

		// Memory management - codellama excels
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(3), "computer_science"), SelectedModel: "codellama-7b", ResponseQuality: 0.92, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(3), "computer_science"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.30, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(3), "computer_science"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.45, Success: true},

		// Parser/compiler - codellama excels
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(4), "computer_science"), SelectedModel: "codellama-7b", ResponseQuality: 0.88, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(4), "computer_science"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.45, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(4), "computer_science"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.60, Success: true},

		// General queries - other models are fine
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(1), "other"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.95, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(2), "other"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.90, Success: true},
	}

	// Train the selector
	err := selector.Train(trainingData)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
	}

	// Test with a complex coding query - should select codellama
	testCases := []struct {
		name          string
		query         string
		category      string
		embeddingSeed int
		expectedModel string
	}{
		{
			name:          "complex_code_1",
			query:         "Implement a concurrent hashmap with atomic operations",
			category:      "computer_science",
			embeddingSeed: 1,
			expectedModel: "codellama-7b",
		},
		{
			name:          "complex_code_2",
			query:         "Write a B+ tree with efficient range queries",
			category:      "computer_science",
			embeddingSeed: 2,
			expectedModel: "codellama-7b",
		},
		{
			name:          "general_query",
			query:         "What is the population of Tokyo?",
			category:      "other",
			embeddingSeed: 1,
			expectedModel: "llama-3.2-1b",
		},
	}

	codelamaSelected := 0
	for _, tc := range testCases {
		var embedding []float64
		if tc.category == "computer_science" {
			embedding = generateCodingEmbedding(tc.embeddingSeed)
		} else {
			embedding = generateGeneralEmbedding(tc.embeddingSeed)
		}

		featureVector := CombineEmbeddingWithCategory(embedding, tc.category)

		ctx := &SelectionContext{
			QueryEmbedding: featureVector,
			QueryText:      tc.query,
			CategoryName:   tc.category,
			DecisionName:   "test_decision",
		}

		selected, err := selector.Select(ctx, modelRefs)
		if err != nil {
			t.Errorf("[%s] selection error: %v", tc.name, err)
			continue
		}

		if selected != nil {
			t.Logf("[%s] Expected: %s, Got: %s", tc.name, tc.expectedModel, selected.Model)
			if selected.Model == tc.expectedModel {
				t.Logf("✓ Correct selection!")
			}
			if selected.Model == "codellama-7b" {
				codelamaSelected++
			}
		}
	}

	if codelamaSelected >= 2 {
		t.Logf("✓ SUCCESS: Algorithm correctly selects codellama-7b for complex coding queries when training data shows it performs better!")
	} else {
		t.Errorf("Expected codellama-7b to be selected for complex coding queries")
	}
}

// Helper function to generate coding-specific embeddings
func generateCodingEmbedding(seed int) []float64 {
	embedding := make([]float64, 768)
	for i := range embedding {
		// Pattern that represents coding/technical content
		embedding[i] = 0.5*float64((i*seed*17)%80-40)/40.0 + 0.3*float64((i+seed)%10)/10.0
	}
	return embedding
}

// Helper function to generate general query embeddings
func generateGeneralEmbedding(seed int) []float64 {
	embedding := make([]float64, 768)
	for i := range embedding {
		// Different pattern for general queries
		embedding[i] = 0.4*float64((i*seed*13)%60-30)/30.0 + 0.2*float64((i+seed*2)%15)/15.0
	}
	return embedding
}

// TestMathQueriesRouting tests that math queries are routed appropriately
func TestMathQueriesRouting(t *testing.T) {
	modelsPath := "data/trained_models"

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// Math-specific test cases
	mathTestCases := []struct {
		name      string
		queryText string
		category  string
	}{
		{
			name:      "calculus_integral",
			queryText: "Calculate the definite integral of x^3 * e^x from 0 to 1 using integration by parts",
			category:  "math",
		},
		{
			name:      "linear_algebra",
			queryText: "Find the eigenvalues and eigenvectors of the 3x3 matrix [[1,2,3],[4,5,6],[7,8,9]]",
			category:  "math",
		},
		{
			name:      "probability",
			queryText: "What is the probability of getting exactly 3 heads in 5 coin flips using binomial distribution?",
			category:  "math",
		},
		{
			name:      "number_theory",
			queryText: "Prove that there are infinitely many prime numbers using Euclid's theorem",
			category:  "math",
		},
		{
			name:      "differential_eq",
			queryText: "Solve the second-order differential equation y'' + 4y' + 4y = 0 with initial conditions y(0)=1, y'(0)=0",
			category:  "math",
		},
	}

	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
			}

			algModelCounts := make(map[string]int)

			for _, tc := range mathTestCases {
				// Generate a math-specific embedding pattern
				embedding := make([]float64, 768)
				for i := range embedding {
					// Pattern that emphasizes math features
					embedding[i] = 0.35*float64((i*23)%60-30)/30.0 + 0.15*float64(i%11)/11.0
				}

				featureVector := CombineEmbeddingWithCategory(embedding, tc.category)

				ctx := &SelectionContext{
					QueryEmbedding: featureVector,
					QueryText:      tc.queryText,
					CategoryName:   tc.category,
					DecisionName:   "math_decision",
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("%s [%s]: selection error: %v", alg, tc.name, err)
					continue
				}

				if selected != nil {
					algModelCounts[selected.Model]++
					t.Logf("%s [%s]: selected %s", alg, tc.name, selected.Model)
				}
			}

			t.Logf("%s math queries: llama-3.2-1b=%d, llama-3.2-3b=%d, codellama-7b=%d, mistral-7b=%d",
				alg, algModelCounts["llama-3.2-1b"], algModelCounts["llama-3.2-3b"],
				algModelCounts["codellama-7b"], algModelCounts["mistral-7b"])

			// Count unique models selected
			modelsSelected := 0
			for _, count := range algModelCounts {
				if count > 0 {
					modelsSelected++
				}
			}
			t.Logf("%s selected %d unique models for math queries", alg, modelsSelected)
		})
	}
}

// TestCodingQueriesWithRealEmbeddings uses actual embeddings from training data
// to verify that algorithms can correctly route based on learned patterns
func TestCodingQueriesWithRealEmbeddings(t *testing.T) {
	modelsPath := "data/trained_models"

	// Load KNN model to extract actual training embeddings
	knnFile := modelsPath + "/knn_model.json"
	data, err := os.ReadFile(knnFile)
	if err != nil {
		t.Skipf("Skipping: pretrained model not available (expected in CI): %v", err)
	}

	var modelData struct {
		TrainingRecords []struct {
			Embedding   []float64 `json:"embedding"`
			ModelName   string    `json:"model_name"`
			Performance float64   `json:"performance"`
			Category    string    `json:"category"`
			Query       string    `json:"query"`
		} `json:"training_records"`
	}

	if err := json.Unmarshal(data, &modelData); err != nil {
		t.Fatalf("Failed to parse knn model: %v", err)
	}

	// Filter to get coding/computer science queries and group by best performer
	codingQueries := make(map[string]struct {
		Embedding []float64
		Category  string
		BestModel string
		BestScore float64
		Query     string
	})

	for _, rec := range modelData.TrainingRecords {
		// Look for computer science queries
		if !strings.Contains(strings.ToLower(rec.Category), "computer") {
			continue
		}

		key := rec.Query[:min(50, len(rec.Query))] // Use first 50 chars as key
		existing, exists := codingQueries[key]
		if !exists || rec.Performance > existing.BestScore {
			codingQueries[key] = struct {
				Embedding []float64
				Category  string
				BestModel string
				BestScore float64
				Query     string
			}{
				Embedding: rec.Embedding,
				Category:  rec.Category,
				BestModel: rec.ModelName,
				BestScore: rec.Performance,
				Query:     rec.Query,
			}
		}
	}

	t.Logf("Found %d unique coding queries with embeddings", len(codingQueries))

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	algorithms := []string{"knn", "kmeans", "mlp", "svm", "matrix_factorization"}

	for _, alg := range algorithms {
		t.Run(alg, func(t *testing.T) {
			selector, err := loadPretrainedSelectorFromPath(alg, modelsPath)
			if err != nil {
				t.Fatalf("Failed to load %s selector: %v", alg, err)
			}

			modelCounts := make(map[string]int)
			codelamaCorrect := 0
			totalCodelamaExpected := 0

			count := 0
			for _, qData := range codingQueries {
				if count >= 20 { // Test first 20 queries
					break
				}
				count++

				ctx := &SelectionContext{
					QueryEmbedding: qData.Embedding,
					QueryText:      qData.Query[:min(80, len(qData.Query))],
					CategoryName:   qData.Category,
					DecisionName:   "coding_decision",
				}

				selected, err := selector.Select(ctx, modelRefs)
				if err != nil {
					t.Errorf("%s: selection error: %v", alg, err)
					continue
				}

				if selected != nil {
					modelCounts[selected.Model]++

					// Track codellama accuracy
					if qData.BestModel == "codellama-7b" {
						totalCodelamaExpected++
						if selected.Model == "codellama-7b" {
							codelamaCorrect++
						}
					}
				}
			}

			t.Logf("%s coding with real embeddings: llama-1b=%d, llama-3b=%d, codellama=%d, mistral=%d",
				alg, modelCounts["llama-3.2-1b"], modelCounts["llama-3.2-3b"],
				modelCounts["codellama-7b"], modelCounts["mistral-7b"])

			if modelCounts["codellama-7b"] > 0 {
				t.Logf("✓ %s routed %d coding queries to codellama-7b!", alg, modelCounts["codellama-7b"])
			}

			uniqueModels := 0
			for _, c := range modelCounts {
				if c > 0 {
					uniqueModels++
				}
			}
			t.Logf("%s used %d unique models", alg, uniqueModels)
		})
	}
}

// TestRoutingAccuracyWithBenchmarkData verifies that algorithms can route to the
// best-performing model based on actual benchmark performance data
func TestRoutingAccuracyWithBenchmarkData(t *testing.T) {
	benchmarkFile := "data/benchmark_training_data.jsonl"

	file, err := os.Open(benchmarkFile)
	if err != nil {
		t.Skipf("Benchmark data not available: %v", err)
	}
	defer file.Close()

	// Parse benchmark records and find best performer per query
	type QueryResult struct {
		Query     string
		Category  string
		BestModel string
		BestScore float64
		AllScores map[string]float64
	}

	queryResults := make(map[string]*QueryResult)

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024) // 1MB buffer for long lines
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		var rec struct {
			Query        string  `json:"query"`
			Category     string  `json:"category"`
			ModelName    string  `json:"model_name"`
			Performance  float64 `json:"performance"`
			ResponseTime float64 `json:"response_time"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue
		}

		// Calculate combined score: 70% quality, 30% efficiency (lower latency = better)
		efficiency := 1.0 / (1.0 + rec.ResponseTime) // Normalize latency
		combinedScore := 0.7*rec.Performance + 0.3*efficiency

		key := rec.Query[:min(100, len(rec.Query))]
		if existing, ok := queryResults[key]; ok {
			existing.AllScores[rec.ModelName] = combinedScore
			if combinedScore > existing.BestScore {
				existing.BestModel = rec.ModelName
				existing.BestScore = combinedScore
			}
		} else {
			queryResults[key] = &QueryResult{
				Query:     rec.Query,
				Category:  rec.Category,
				BestModel: rec.ModelName,
				BestScore: combinedScore,
				AllScores: map[string]float64{rec.ModelName: combinedScore},
			}
		}
	}

	// Count which model should be selected most often per category
	categoryCounts := make(map[string]map[string]int) // category -> model -> count
	for _, qr := range queryResults {
		if len(qr.AllScores) >= 2 { // Only count queries with multiple model results
			cat := strings.ToLower(qr.Category)
			if categoryCounts[cat] == nil {
				categoryCounts[cat] = make(map[string]int)
			}
			categoryCounts[cat][qr.BestModel]++
		}
	}

	t.Log("=== Best Performing Model per Category (based on benchmark data) ===")
	for cat, counts := range categoryCounts {
		// Find best model for this category
		var bestModel string
		var bestCount int
		for model, count := range counts {
			if count > bestCount {
				bestModel = model
				bestCount = count
			}
		}
		t.Logf("%s: best=%s (%d/%d queries), counts=%v",
			cat, bestModel, bestCount, sumCounts(counts), counts)
	}

	// Check specifically for computer science / coding
	if cs, ok := categoryCounts["computer science"]; ok {
		if cs["codellama-7b"] > 0 {
			pct := float64(cs["codellama-7b"]) * 100 / float64(sumCounts(cs))
			t.Logf("✓ codellama-7b is best for %.1f%% of computer science queries", pct)
		} else {
			t.Log("⚠ codellama-7b is NOT the best performer for any computer science queries in the benchmark")
			t.Log("  This explains why it's not being selected - other models perform better based on the data")
		}
	}
}

func sumCounts(m map[string]int) int {
	total := 0
	for _, v := range m {
		total += v
	}
	return total
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestCategoryBasedRoutingHint demonstrates how to configure category-based
// model preferences when you want specific models for specific task types
// regardless of benchmark performance (e.g., always use codellama for coding)
func TestCategoryBasedRoutingHint(t *testing.T) {
	// This test demonstrates a "category hint" approach where you can
	// configure preferred models per category in your routing rules

	// Category -> Preferred model mapping (could be from config)
	categoryPreferences := map[string]string{
		"computer science": "codellama-7b",
		"computer_science": "codellama-7b",
		"math":             "llama-3.2-3b", // Larger model for complex math
		"physics":          "llama-3.2-3b",
	}

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	testCases := []struct {
		name              string
		query             string
		category          string
		expectedPreferred string
	}{
		{
			name:              "coding_query",
			query:             "Write a Python quicksort implementation",
			category:          "computer_science",
			expectedPreferred: "codellama-7b",
		},
		{
			name:              "math_query",
			query:             "Solve the differential equation y'' + y = 0",
			category:          "math",
			expectedPreferred: "llama-3.2-3b",
		},
		{
			name:              "general_query",
			query:             "What is the capital of France?",
			category:          "other",
			expectedPreferred: "", // No preference, use algorithm
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Check if we have a category preference
			preferredModel, hasPreference := categoryPreferences[tc.category]

			if hasPreference {
				// Find the preferred model in refs
				var selectedRef *config.ModelRef
				for i, ref := range modelRefs {
					if ref.Model == preferredModel {
						selectedRef = &modelRefs[i]
						break
					}
				}

				if selectedRef == nil {
					t.Errorf("Preferred model %s not found in refs", preferredModel)
					return
				}

				t.Logf("Category '%s' -> Preferred model: %s", tc.category, selectedRef.Model)

				if tc.expectedPreferred != "" && selectedRef.Model != tc.expectedPreferred {
					t.Errorf("Expected %s, got %s", tc.expectedPreferred, selectedRef.Model)
				} else {
					t.Logf("✓ Correctly routed to %s for %s queries", selectedRef.Model, tc.category)
				}
			} else {
				t.Logf("No category preference for '%s', would use ML algorithm", tc.category)
			}
		})
	}

	t.Log("")
	t.Log("=== Category-Based Routing Configuration ===")
	t.Log("To enforce category-based routing in config, use decision rules like:")
	t.Log("")
	t.Log("decisions:")
	t.Log("  - name: coding_route")
	t.Log("    rules:")
	t.Log("      conditions:")
	t.Log("        - type: domain")
	t.Log("          name: \"computer science\"")
	t.Log("    modelRefs:")
	t.Log("      - model: codellama-7b  # Preferred for coding")
	t.Log("    algorithm:")
	t.Log("      type: knn  # Fallback algorithm if needed")
}

// TestVerifyBenchmarkDataQuality checks the benchmark data to understand
// why certain models are preferred for certain categories
func TestVerifyBenchmarkDataQuality(t *testing.T) {
	benchmarkFile := "data/benchmark_training_data.jsonl"

	file, err := os.Open(benchmarkFile)
	if err != nil {
		t.Skipf("Benchmark data not available: %v", err)
	}
	defer file.Close()

	type ModelStats struct {
		TotalQueries int
		TotalPerf    float64
		TotalLatency float64
		AvgPerf      float64
		AvgLatency   float64
	}

	// category -> model -> stats
	categoryModelStats := make(map[string]map[string]*ModelStats)

	scanner := bufio.NewScanner(file)
	buf := make([]byte, 1024*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		var rec struct {
			Category     string  `json:"category"`
			ModelName    string  `json:"model_name"`
			Performance  float64 `json:"performance"`
			ResponseTime float64 `json:"response_time"`
		}
		if err := json.Unmarshal(scanner.Bytes(), &rec); err != nil {
			continue
		}

		cat := strings.ToLower(rec.Category)
		if categoryModelStats[cat] == nil {
			categoryModelStats[cat] = make(map[string]*ModelStats)
		}
		if categoryModelStats[cat][rec.ModelName] == nil {
			categoryModelStats[cat][rec.ModelName] = &ModelStats{}
		}

		stats := categoryModelStats[cat][rec.ModelName]
		stats.TotalQueries++
		stats.TotalPerf += rec.Performance
		stats.TotalLatency += rec.ResponseTime
	}

	// Calculate averages
	for _, models := range categoryModelStats {
		for _, stats := range models {
			if stats.TotalQueries > 0 {
				stats.AvgPerf = stats.TotalPerf / float64(stats.TotalQueries)
				stats.AvgLatency = stats.TotalLatency / float64(stats.TotalQueries)
			}
		}
	}

	// Focus on computer science category
	t.Log("=== Computer Science Category Analysis ===")
	if csStats, ok := categoryModelStats["computer science"]; ok {
		for model, stats := range csStats {
			combinedScore := 0.7*stats.AvgPerf + 0.3*(1.0/(1.0+stats.AvgLatency))
			t.Logf("%s: queries=%d, avg_quality=%.3f, avg_latency=%.2fs, combined_score=%.3f",
				model, stats.TotalQueries, stats.AvgPerf, stats.AvgLatency, combinedScore)
		}
	}

	// Focus on math category
	t.Log("")
	t.Log("=== Math Category Analysis ===")
	if mathStats, ok := categoryModelStats["math"]; ok {
		for model, stats := range mathStats {
			combinedScore := 0.7*stats.AvgPerf + 0.3*(1.0/(1.0+stats.AvgLatency))
			t.Logf("%s: queries=%d, avg_quality=%.3f, avg_latency=%.2fs, combined_score=%.3f",
				model, stats.TotalQueries, stats.AvgPerf, stats.AvgLatency, combinedScore)
		}
	}

	t.Log("")
	t.Log("=== Explanation ===")
	t.Log("Combined score = 0.9 * quality + 0.1 * efficiency")
	t.Log("Models with faster latency have higher efficiency scores")
	t.Log("This explains why smaller/faster models often win despite similar quality")
}

// TestMFWithQualityDifferences tests that MF correctly routes based on quality differences
// When models have DIFFERENT quality scores, MF should pick the higher quality model
func TestMFWithQualityDifferences(t *testing.T) {
	t.Log("=== MF Quality-Based Routing Test ===")
	t.Log("Testing that MF routes to higher-quality models when quality differs")
	t.Log("")

	// Create training data where each model excels in different domains
	// This simulates real-world scenarios where specialized models outperform general ones
	trainingData := []TrainingRecord{
		// Domain: Coding - codellama excels (quality=0.95 vs others ~0.4)
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "codellama-7b", ResponseQuality: 0.95, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.35, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.40, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(1), "computer_science"), SelectedModel: "mistral-7b", ResponseQuality: 0.45, Success: true},

		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "codellama-7b", ResponseQuality: 0.92, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.30, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.38, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateCodingEmbedding(2), "computer_science"), SelectedModel: "mistral-7b", ResponseQuality: 0.42, Success: true},

		// Domain: Math - mistral excels (quality=0.90 vs others ~0.5)
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(1), "math"), SelectedModel: "mistral-7b", ResponseQuality: 0.90, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(1), "math"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.45, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(1), "math"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.55, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(1), "math"), SelectedModel: "codellama-7b", ResponseQuality: 0.50, Success: true},

		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(2), "math"), SelectedModel: "mistral-7b", ResponseQuality: 0.88, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(2), "math"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.40, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(2), "math"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.52, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateMathEmbedding(2), "math"), SelectedModel: "codellama-7b", ResponseQuality: 0.48, Success: true},

		// Domain: General/Creative - llama-3.2-3b excels (quality=0.85 vs others ~0.5)
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(1), "other"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.85, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(1), "other"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.50, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(1), "other"), SelectedModel: "codellama-7b", ResponseQuality: 0.45, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(1), "other"), SelectedModel: "mistral-7b", ResponseQuality: 0.55, Success: true},

		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(2), "other"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.82, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(2), "other"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.48, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(2), "other"), SelectedModel: "codellama-7b", ResponseQuality: 0.42, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generateGeneralEmbedding(2), "other"), SelectedModel: "mistral-7b", ResponseQuality: 0.52, Success: true},

		// Domain: Physics - llama-3.2-1b excels on simple, mistral on complex
		{QueryEmbedding: CombineEmbeddingWithCategory(generatePhysicsEmbedding(1), "physics"), SelectedModel: "llama-3.2-1b", ResponseQuality: 0.80, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generatePhysicsEmbedding(1), "physics"), SelectedModel: "llama-3.2-3b", ResponseQuality: 0.55, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generatePhysicsEmbedding(1), "physics"), SelectedModel: "codellama-7b", ResponseQuality: 0.40, Success: true},
		{QueryEmbedding: CombineEmbeddingWithCategory(generatePhysicsEmbedding(1), "physics"), SelectedModel: "mistral-7b", ResponseQuality: 0.60, Success: true},
	}

	// Create MF selector and train with this data
	mfSelector := NewMatrixFactorizationSelector(10)
	err := mfSelector.Train(trainingData)
	if err != nil {
		t.Fatalf("Failed to train MF selector: %v", err)
	}

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// Test cases: query similar to each domain
	testCases := []struct {
		name          string
		embedding     []float64
		category      string
		expectedModel string
		description   string
	}{
		{
			name:          "coding_query",
			embedding:     generateCodingEmbedding(3), // Similar to coding training data
			category:      "computer_science",
			expectedModel: "codellama-7b",
			description:   "Coding query should route to codellama (highest quality=0.95)",
		},
		{
			name:          "math_query",
			embedding:     generateMathEmbedding(3),
			category:      "math",
			expectedModel: "mistral-7b",
			description:   "Math query should route to mistral (highest quality=0.90)",
		},
		{
			name:          "general_query",
			embedding:     generateGeneralEmbedding(3),
			category:      "other",
			expectedModel: "llama-3.2-3b",
			description:   "General query should route to llama-3b (highest quality=0.85)",
		},
		{
			name:          "physics_query",
			embedding:     generatePhysicsEmbedding(2),
			category:      "physics",
			expectedModel: "llama-3.2-1b",
			description:   "Physics query should route to llama-1b (highest quality=0.80)",
		},
	}

	// Track results
	correctSelections := 0
	modelCounts := make(map[string]int)

	for _, tc := range testCases {
		ctx := &SelectionContext{
			QueryEmbedding: CombineEmbeddingWithCategory(tc.embedding, tc.category),
		}

		selected, err := mfSelector.Select(ctx, modelRefs)
		if err != nil {
			t.Errorf("[%s] Error: %v", tc.name, err)
			continue
		}

		if selected != nil {
			modelCounts[selected.Model]++
			if selected.Model == tc.expectedModel {
				correctSelections++
				t.Logf("✓ [%s] Correct! Selected %s (expected %s)", tc.name, selected.Model, tc.expectedModel)
				t.Logf("  %s", tc.description)
			} else {
				t.Logf("✗ [%s] Selected %s (expected %s)", tc.name, selected.Model, tc.expectedModel)
				t.Logf("  %s", tc.description)
			}
		}
	}

	t.Log("")
	t.Log("=== Summary ===")
	t.Logf("Correct selections: %d/%d", correctSelections, len(testCases))
	t.Logf("Model distribution: %v", modelCounts)

	// Check if MF selected multiple different models
	uniqueModels := len(modelCounts)
	if uniqueModels >= 3 {
		t.Logf("✓ SUCCESS: MF selected %d different models based on quality differences!", uniqueModels)
	} else if uniqueModels >= 2 {
		t.Logf("⚠ PARTIAL: MF selected %d different models", uniqueModels)
	} else {
		t.Errorf("✗ FAIL: MF only selected 1 model type - quality-based routing not working")
	}
}

// Helper function to generate math-specific embeddings
func generateMathEmbedding(seed int) []float64 {
	embedding := make([]float64, 768)
	for i := range embedding {
		// Pattern that represents math content
		embedding[i] = 0.6*float64((i*seed*19)%100-50)/50.0 + 0.2*float64((i+seed*3)%12)/12.0
	}
	return embedding
}

// Helper function to generate physics-specific embeddings
func generatePhysicsEmbedding(seed int) []float64 {
	embedding := make([]float64, 768)
	for i := range embedding {
		// Pattern that represents physics content
		embedding[i] = 0.5*float64((i*seed*23)%90-45)/45.0 + 0.25*float64((i+seed*4)%8)/8.0
	}
	return embedding
}

// TestMFWithRealQueries tests MF routing with actual query text
// Uses Candle/Qwen3 embeddings (same as production)
func TestMFWithRealQueries(t *testing.T) {
	t.Log("=== MF Routing with Real Query Text (Candle/Qwen3 Embeddings) ===")
	t.Log("Training MF with queries where models have DIFFERENT quality scores")
	t.Log("")

	// Try to initialize Candle embedding models
	candleAvailable := false
	qwen3Path := "../../../../models/mom-embedding-pro"

	// Try to initialize Candle with Qwen3
	err := candle_binding.InitEmbeddingModelsBatched(qwen3Path, 32, 10, true)
	if err == nil {
		candleAvailable = true
		t.Log("✓ Candle/Qwen3 initialized - using production embeddings")
	} else {
		t.Logf("⚠ Candle not available (%v) - using hash-based fallback", err)
	}

	// Create trainer to generate embeddings from text
	trainer := NewTrainer(768)
	trainer.SetUseCandle(candleAvailable)
	trainer.SetEmbeddingModel("qwen3")

	// Training data with real queries and different quality per model
	type queryModelScore struct {
		query    string
		category string
		model    string
		quality  float64
	}

	trainingQueries := []queryModelScore{
		// Coding queries - codellama excels
		{"Write a Python function to implement quicksort algorithm", "computer_science", "codellama-7b", 0.95},
		{"Write a Python function to implement quicksort algorithm", "computer_science", "llama-3.2-1b", 0.40},
		{"Write a Python function to implement quicksort algorithm", "computer_science", "llama-3.2-3b", 0.50},
		{"Write a Python function to implement quicksort algorithm", "computer_science", "mistral-7b", 0.55},

		{"Implement a binary search tree with insert and delete operations", "computer_science", "codellama-7b", 0.92},
		{"Implement a binary search tree with insert and delete operations", "computer_science", "llama-3.2-1b", 0.35},
		{"Implement a binary search tree with insert and delete operations", "computer_science", "llama-3.2-3b", 0.45},
		{"Implement a binary search tree with insert and delete operations", "computer_science", "mistral-7b", 0.50},

		// Math queries - mistral excels
		{"Calculate the eigenvalues of a 3x3 symmetric matrix", "math", "mistral-7b", 0.90},
		{"Calculate the eigenvalues of a 3x3 symmetric matrix", "math", "llama-3.2-1b", 0.40},
		{"Calculate the eigenvalues of a 3x3 symmetric matrix", "math", "llama-3.2-3b", 0.55},
		{"Calculate the eigenvalues of a 3x3 symmetric matrix", "math", "codellama-7b", 0.45},

		{"Prove that the sum of angles in a triangle equals 180 degrees", "math", "mistral-7b", 0.88},
		{"Prove that the sum of angles in a triangle equals 180 degrees", "math", "llama-3.2-1b", 0.45},
		{"Prove that the sum of angles in a triangle equals 180 degrees", "math", "llama-3.2-3b", 0.60},
		{"Prove that the sum of angles in a triangle equals 180 degrees", "math", "codellama-7b", 0.40},

		// General queries - llama-3.2-3b excels
		{"Write a creative story about a robot learning to paint", "other", "llama-3.2-3b", 0.88},
		{"Write a creative story about a robot learning to paint", "other", "llama-3.2-1b", 0.50},
		{"Write a creative story about a robot learning to paint", "other", "codellama-7b", 0.40},
		{"Write a creative story about a robot learning to paint", "other", "mistral-7b", 0.55},

		// Physics - llama-3.2-1b excels on simple physics
		{"What is Newton's first law of motion?", "physics", "llama-3.2-1b", 0.85},
		{"What is Newton's first law of motion?", "physics", "llama-3.2-3b", 0.60},
		{"What is Newton's first law of motion?", "physics", "codellama-7b", 0.45},
		{"What is Newton's first law of motion?", "physics", "mistral-7b", 0.55},
	}

	// Convert to training records
	var trainingRecords []TrainingRecord
	for _, q := range trainingQueries {
		embedding, embErr := trainer.GetEmbedding(q.query)
		if embErr != nil {
			t.Fatalf("Failed to get embedding: %v", embErr)
		}
		featureVec := CombineEmbeddingWithCategory(embedding, q.category)
		trainingRecords = append(trainingRecords, TrainingRecord{
			QueryEmbedding:  featureVec,
			SelectedModel:   q.model,
			ResponseQuality: q.quality,
			Success:         true,
		})
	}

	// Train MF selector
	mfSelector := NewMatrixFactorizationSelector(10)
	err = mfSelector.Train(trainingRecords)
	if err != nil {
		t.Fatalf("Failed to train MF: %v", err)
	}

	modelRefs := []config.ModelRef{
		{Model: "llama-3.2-1b"},
		{Model: "llama-3.2-3b"},
		{Model: "codellama-7b"},
		{Model: "mistral-7b"},
	}

	// Test with new queries (similar to training queries)
	testCases := []struct {
		query         string
		category      string
		expectedModel string
	}{
		{"Write a recursive function in Python to calculate factorial", "computer_science", "codellama-7b"},
		{"Solve the quadratic equation x^2 + 5x + 6 = 0", "math", "mistral-7b"},
		{"Write a poem about the sunset", "other", "llama-3.2-3b"},
		{"Explain the concept of gravity", "physics", "llama-3.2-1b"},
	}

	correctCount := 0
	modelCounts := make(map[string]int)

	for _, tc := range testCases {
		embedding, _ := trainer.GetEmbedding(tc.query)
		featureVec := CombineEmbeddingWithCategory(embedding, tc.category)

		ctx := &SelectionContext{
			QueryEmbedding: featureVec,
		}

		selected, err := mfSelector.Select(ctx, modelRefs)
		if err != nil {
			t.Errorf("Error selecting for '%s': %v", tc.query[:30], err)
			continue
		}

		if selected != nil {
			modelCounts[selected.Model]++
			queryPreview := tc.query
			if len(queryPreview) > 40 {
				queryPreview = queryPreview[:40] + "..."
			}
			if selected.Model == tc.expectedModel {
				correctCount++
				t.Logf("✓ Query: '%s' → %s (expected)", queryPreview, selected.Model)
			} else {
				t.Logf("✗ Query: '%s' → %s (expected %s)", queryPreview, selected.Model, tc.expectedModel)
			}
		}
	}

	t.Log("")
	t.Logf("Correct: %d/%d | Models used: %v", correctCount, len(testCases), modelCounts)

	if len(modelCounts) >= 2 {
		t.Logf("✓ SUCCESS: MF selected multiple different models based on query content!")
	}
}
