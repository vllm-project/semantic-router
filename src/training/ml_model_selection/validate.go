package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"time"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
)

const (
	// HuggingFace repositories
	defaultModelsRepo = "abdallah1008/semantic-router-ml-models"
	defaultDataRepo   = "abdallah1008/ml-selection-benchmark-data"
)

// BenchmarkRecord represents a record from the benchmark data JSONL
type BenchmarkRecord struct {
	Query        string    `json:"query"`
	Category     string    `json:"category"`
	ModelName    string    `json:"model_name"`
	Performance  float64   `json:"performance"`
	ResponseTime float64   `json:"response_time"`
	Embedding    []float64 `json:"embedding,omitempty"`
}

// QueryResult groups all model results for a single query
type QueryResult struct {
	Query      string
	Category   string
	Embedding  []float64
	ModelPerfs map[string]float64 // model_name -> performance
	ModelLats  map[string]float64 // model_name -> latency
	BestModel  string
	BestPerf   float64
}

// StrategyResult holds evaluation results for a strategy
type StrategyResult struct {
	Name         string
	AvgQuality   float64
	AvgLatency   float64
	BestModelPct float64
}

func main() {
	// Command line flags
	dataFile := flag.String("data-file", "",
		"Path to benchmark data JSONL file (optional - downloads from HuggingFace if not provided)")
	modelsDir := flag.String("models-dir", ".cache/ml-models",
		"Directory for downloaded/trained models")
	algorithm := flag.String("algorithm", "all",
		"Algorithm to validate: knn, kmeans, svm, all")
	testSplit := flag.Float64("test-split", 1.0,
		"Fraction of data to use for testing (default: 1.0 = all data)")
	seed := flag.Int64("seed", 42,
		"Random seed for reproducibility")
	modelsRepo := flag.String("models-repo", defaultModelsRepo,
		"HuggingFace repository for pretrained models")
	dataRepo := flag.String("data-repo", defaultDataRepo,
		"HuggingFace dataset repository for benchmark data")
	noDownload := flag.Bool("no-download", false,
		"Skip HuggingFace download, use local files only")
	qwen3ModelPath := flag.String("qwen3-model", "",
		"Path to Qwen3-Embedding-0.6B model (downloads from HuggingFace if not provided)")
	noEmbeddings := flag.Bool("no-embeddings", false,
		"Skip embedding generation (use random vectors for testing)")

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `ML Model Selection Validation Tool

Validate that ML-based routing provides benefit over baselines.

This tool uses the ACTUAL production Go/Rust selectors to validate
model selection performance, proving the end-to-end system works.

It automatically downloads pretrained models and benchmark data from HuggingFace:
  - Models: %s
  - Data: %s

It compares:
  - Oracle (best possible): Always picks the actual best model
  - ML Selection (KNN/KMeans/SVM): Uses trained ML model via Rust FFI
  - Random Selection: Randomly picks a model
  - Single Model: Always picks one specific model

Usage:
  go run validate.go [flags]

Examples:
  go run validate.go                                    # Downloads everything from HuggingFace
  go run validate.go --algorithm knn                    # Validate only KNN
  go run validate.go --data-file local.jsonl            # Use local data file
  go run validate.go --no-download                      # Skip downloads, use local files only

Flags:
`, defaultModelsRepo, defaultDataRepo)
		flag.PrintDefaults()
	}

	flag.Parse()

	rand.Seed(*seed)

	fmt.Println("=" + strings.Repeat("=", 69))
	fmt.Println("  ML Model Selection Validation (Production Go/Rust Code)")
	fmt.Println("=" + strings.Repeat("=", 69))

	// Download from HuggingFace if needed
	if !*noDownload {
		if err := downloadFromHuggingFace(*modelsDir, *modelsRepo, *dataRepo, *dataFile); err != nil {
			fmt.Fprintf(os.Stderr, "Error: failed to download from HuggingFace: %v\n", err)
			os.Exit(1)
		}
	}

	// Set data file path if not provided (use validation data, not training data)
	actualDataFile := *dataFile
	if actualDataFile == "" {
		// Look for validation data in the same directory as models
		actualDataFile = filepath.Join(*modelsDir, "validation_benchmark_with_gt.jsonl")
	}

	fmt.Println()
	fmt.Printf("Data file:   %s\n", actualDataFile)
	fmt.Printf("Models dir:  %s\n", *modelsDir)
	fmt.Printf("Algorithm:   %s\n", *algorithm)
	fmt.Printf("Test split:  %.0f%%\n", *testSplit*100)
	fmt.Println()

	// Load benchmark data
	fmt.Println("Loading benchmark data...")
	queryResults, modelNames, err := loadBenchmarkData(actualDataFile, *testSplit)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: failed to load benchmark data: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Loaded %d test queries with %d models\n", len(queryResults), len(modelNames))
	fmt.Printf("Models: %s\n\n", strings.Join(modelNames, ", "))

	// Initialize Qwen3 embedding model (unless --no-embeddings)
	if !*noEmbeddings {
		fmt.Println("Initializing Qwen3 embedding model...")
		qwen3Path := *qwen3ModelPath
		if qwen3Path == "" {
			// Default to HuggingFace model ID (will be downloaded automatically)
			qwen3Path = "Qwen/Qwen3-Embedding-0.6B"
		}

		// Initialize Qwen3 for embeddings (1024-dim, matches training)
		// Using batched initialization with batch size 32 and 100ms timeout
		if err := candle_binding.InitEmbeddingModelsBatched(qwen3Path, 32, 100, false); err != nil {
			fmt.Fprintf(os.Stderr, "Error: failed to initialize Qwen3 embeddings: %v\nPlease download the model or use --no-embeddings for testing\n", err)
			os.Exit(1)
		}
		fmt.Printf("Loaded Qwen3 embedding model: %s\n", qwen3Path)
	}

	// Generate embeddings for queries that don't have them
	if !*noEmbeddings {
		fmt.Println("\nGenerating embeddings for test queries...")
		for i := range queryResults {
			if len(queryResults[i].Embedding) == 0 {
				// Generate embedding using Qwen3 batched API (1024-dim)
				output, err := candle_binding.GetEmbeddingBatched(queryResults[i].Query, "qwen3", 1024)
				if err != nil {
					fmt.Fprintf(os.Stderr, "Error: failed to generate embedding for query %d: %v\n", i, err)
					os.Exit(1)
				}
				// Convert float32 to float64
				queryResults[i].Embedding = make([]float64, len(output.Embedding))
				for j, v := range output.Embedding {
					queryResults[i].Embedding[j] = float64(v)
				}
			}
			// Progress indicator
			if (i+1)%20 == 0 || i+1 == len(queryResults) {
				fmt.Printf("\r  Generated embeddings: %d/%d", i+1, len(queryResults))
			}
		}
		fmt.Println()
	} else {
		// Use random embeddings for testing
		fmt.Println("Using random embeddings (--no-embeddings mode)...")
		for i := range queryResults {
			if len(queryResults[i].Embedding) == 0 {
				queryResults[i].Embedding = make([]float64, 1024)
				for j := range queryResults[i].Embedding {
					queryResults[i].Embedding[j] = rand.Float64()*2 - 1
				}
			}
		}
	}

	// Load selectors
	selectors := make(map[string]modelselection.Selector)
	algorithms := []string{"knn", "kmeans", "svm"}
	if *algorithm != "all" {
		algorithms = []string{*algorithm}
	}

	for _, alg := range algorithms {
		cfg := &config.MLModelSelectionConfig{
			Type:       alg,
			ModelsPath: *modelsDir,
		}
		selector, err := modelselection.NewSelector(cfg)
		if err != nil {
			fmt.Printf("Warning: Could not load %s selector: %v\n", alg, err)
			continue
		}
		selectors[alg] = selector
		fmt.Printf("Loaded %s selector from %s\n", strings.ToUpper(alg), *modelsDir)
	}

	if len(selectors) == 0 {
		fmt.Println("\nWarning: No ML selectors loaded. Only baseline strategies will be evaluated.")
	}

	// Evaluate strategies
	fmt.Println("\nEvaluating strategies...")
	results := evaluateStrategies(queryResults, modelNames, selectors)

	// Print results
	printResults(results, len(queryResults), len(modelNames))
}

// downloadFromHuggingFace downloads pretrained models and benchmark data from HuggingFace.
// Uses huggingface-cli for robust downloads (handles auth, LFS, caching, parquet, etc.)
// Reference: https://github.com/huggingface/candle/blob/main/candle-datasets/src/hub.rs
func downloadFromHuggingFace(modelsDir, modelsRepo, dataRepo, dataFile string) error {
	// Create models directory if it doesn't exist
	if err := os.MkdirAll(modelsDir, 0o755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	// Check if huggingface-cli is available
	if !hfCliAvailable() {
		return fmt.Errorf("huggingface-cli not found. Install with: pip install huggingface-hub")
	}

	fmt.Println("Downloading from HuggingFace...")

	// Download models using huggingface-cli
	modelFiles := []string{"knn_model.json", "kmeans_model.json", "svm_model.json"}
	for _, file := range modelFiles {
		destPath := filepath.Join(modelsDir, file)
		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("  %s already exists, skipping\n", file)
			continue
		}

		fmt.Printf("  Downloading %s...\n", file)
		if err := downloadWithHfCli(modelsRepo, file, destPath, false); err != nil {
			fmt.Printf("  Warning: Could not download %s: %v\n", file, err)
			continue
		}
		fmt.Printf("  Downloaded %s\n", file)
	}

	// Download validation benchmark data
	benchmarkFile := "validation_benchmark_with_gt.jsonl"
	destPath := filepath.Join(modelsDir, benchmarkFile)
	if dataFile == "" {
		if _, err := os.Stat(destPath); err == nil {
			fmt.Printf("  %s already exists, skipping\n", benchmarkFile)
		} else {
			fmt.Printf("  Downloading %s from dataset...\n", benchmarkFile)
			if err := downloadWithHfCli(dataRepo, benchmarkFile, destPath, true); err != nil {
				fmt.Printf("  Warning: Could not download benchmark data: %v\n", err)
				fmt.Println("  Please provide --data-file or upload validation_benchmark_with_gt.jsonl to the HF dataset")
			} else {
				fmt.Printf("  Downloaded %s\n", benchmarkFile)
			}
		}
	}

	return nil
}

// hfCliAvailable checks if huggingface-cli is available
func hfCliAvailable() bool {
	_, err := exec.LookPath("huggingface-cli")
	return err == nil
}

// downloadWithHfCli downloads a file using huggingface-cli
func downloadWithHfCli(repo, filename, destPath string, isDataset bool) error {
	// huggingface-cli download <repo> <filename> --local-dir <dir>
	args := []string{"download", repo, filename, "--local-dir", filepath.Dir(destPath)}
	if isDataset {
		args = append(args, "--repo-type", "dataset")
	}

	cmd := exec.Command("huggingface-cli", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("huggingface-cli failed: %w\nOutput: %s", err, string(output))
	}

	// huggingface-cli downloads to <local-dir>/<filename>, move to destPath if needed
	downloadedPath := filepath.Join(filepath.Dir(destPath), filename)
	if downloadedPath != destPath {
		return os.Rename(downloadedPath, destPath)
	}
	return nil
}

func loadBenchmarkData(dataFile string, testSplit float64) ([]QueryResult, []string, error) {
	file, err := os.Open(dataFile)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// Read all records and group by query
	queryMap := make(map[string]*QueryResult)
	modelSet := make(map[string]bool)

	scanner := bufio.NewScanner(file)
	// Increase buffer size for long lines
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if strings.TrimSpace(line) == "" {
			continue
		}

		var record BenchmarkRecord
		if err := json.Unmarshal([]byte(line), &record); err != nil {
			continue
		}

		if record.Query == "" || record.ModelName == "" {
			continue
		}

		modelSet[record.ModelName] = true

		qr, exists := queryMap[record.Query]
		if !exists {
			qr = &QueryResult{
				Query:      record.Query,
				Category:   record.Category,
				Embedding:  record.Embedding,
				ModelPerfs: make(map[string]float64),
				ModelLats:  make(map[string]float64),
			}
			queryMap[record.Query] = qr
		}

		qr.ModelPerfs[record.ModelName] = record.Performance
		qr.ModelLats[record.ModelName] = record.ResponseTime

		// Track best model
		if record.Performance > qr.BestPerf {
			qr.BestPerf = record.Performance
			qr.BestModel = record.ModelName
		}
	}

	if err := scanner.Err(); err != nil {
		return nil, nil, err
	}

	// Convert to slice and apply test split
	allQueries := make([]QueryResult, 0, len(queryMap))
	for _, qr := range queryMap {
		// Only include queries that have results for all models
		if len(qr.ModelPerfs) >= 2 {
			allQueries = append(allQueries, *qr)
		}
	}

	// Shuffle and take test split
	rand.Shuffle(len(allQueries), func(i, j int) {
		allQueries[i], allQueries[j] = allQueries[j], allQueries[i]
	})

	testSize := int(float64(len(allQueries)) * testSplit)
	if testSize < 1 {
		testSize = len(allQueries)
	}
	testQueries := allQueries[:testSize]

	// Get model names
	modelNames := make([]string, 0, len(modelSet))
	for model := range modelSet {
		modelNames = append(modelNames, model)
	}
	sort.Strings(modelNames)

	return testQueries, modelNames, nil
}

func evaluateStrategies(queries []QueryResult, modelNames []string, selectors map[string]modelselection.Selector) []StrategyResult {
	results := make([]StrategyResult, 0)

	// Oracle (best possible)
	oracleResult := evaluateOracle(queries)
	results = append(results, oracleResult)

	// ML Selectors
	for name, selector := range selectors {
		result := evaluateSelector(queries, modelNames, selector, name)
		results = append(results, result)
	}

	// Random selection
	randomResult := evaluateRandom(queries, modelNames)
	results = append(results, randomResult)

	// Single model baselines
	for _, model := range modelNames {
		result := evaluateSingleModel(queries, model)
		results = append(results, result)
	}

	return results
}

// normalizeModelName normalizes model names for comparison.
func normalizeModelName(name string) string {
	return strings.ToLower(strings.TrimSpace(name))
}

// findModelInMap looks up a model in a map, handling name format differences.
// Returns the actual key and value if found.
func findModelInMap(predicted string, modelMap map[string]float64) (string, float64, bool) {
	// Direct lookup first
	if val, ok := modelMap[predicted]; ok {
		return predicted, val, true
	}

	// Try normalized lookup
	predNorm := normalizeModelName(predicted)
	for key, val := range modelMap {
		if normalizeModelName(key) == predNorm {
			return key, val, true
		}
	}

	return "", 0, false
}

func evaluateOracle(queries []QueryResult) StrategyResult {
	var totalQuality, totalLatency float64
	bestCount := 0

	for _, qr := range queries {
		totalQuality += qr.BestPerf
		totalLatency += qr.ModelLats[qr.BestModel]
		bestCount++
	}

	n := float64(len(queries))
	return StrategyResult{
		Name:         "Oracle (best)",
		AvgQuality:   totalQuality / n,
		AvgLatency:   totalLatency / n,
		BestModelPct: 100.0,
	}
}

func evaluateSelector(queries []QueryResult, modelNames []string, selector modelselection.Selector, name string) StrategyResult {
	var totalQuality, totalLatency float64
	bestCount := 0

	// Create model refs for the selector
	refs := make([]config.ModelRef, len(modelNames))
	for i, modelName := range modelNames {
		refs[i] = config.ModelRef{Model: modelName}
	}

	for _, qr := range queries {
		// Create selection context
		ctx := &modelselection.SelectionContext{
			QueryEmbedding: qr.Embedding,
			QueryText:      qr.Query,
			CategoryName:   qr.Category,
		}

		// Get selection
		selected, err := selector.Select(ctx, refs)
		var selectedModel string
		if err != nil || selected == nil {
			// Fallback to random
			selectedModel = modelNames[rand.Intn(len(modelNames))]
		} else {
			selectedModel = selected.Model
		}

		// Get performance for selected model (with name normalization)
		actualKey, perf, found := findModelInMap(selectedModel, qr.ModelPerfs)
		if !found {
			// Model not found - use 0
			perf = 0
		}
		_, lat, _ := findModelInMap(selectedModel, qr.ModelLats)

		totalQuality += perf
		totalLatency += lat

		// Check if this is the best model (with normalization)
		if found && normalizeModelName(actualKey) == normalizeModelName(qr.BestModel) {
			bestCount++
		}
	}

	n := float64(len(queries))
	return StrategyResult{
		Name:         fmt.Sprintf("%s Selection", strings.ToUpper(name)),
		AvgQuality:   totalQuality / n,
		AvgLatency:   totalLatency / n,
		BestModelPct: float64(bestCount) / n * 100,
	}
}

func evaluateRandom(queries []QueryResult, modelNames []string) StrategyResult {
	var totalQuality, totalLatency float64
	bestCount := 0

	for _, qr := range queries {
		selected := modelNames[rand.Intn(len(modelNames))]
		totalQuality += qr.ModelPerfs[selected]
		totalLatency += qr.ModelLats[selected]
		if selected == qr.BestModel {
			bestCount++
		}
	}

	n := float64(len(queries))
	return StrategyResult{
		Name:         "Random Selection",
		AvgQuality:   totalQuality / n,
		AvgLatency:   totalLatency / n,
		BestModelPct: float64(bestCount) / n * 100,
	}
}

func evaluateSingleModel(queries []QueryResult, model string) StrategyResult {
	var totalQuality, totalLatency float64
	bestCount := 0

	for _, qr := range queries {
		perf, ok := qr.ModelPerfs[model]
		if !ok {
			continue
		}
		totalQuality += perf
		totalLatency += qr.ModelLats[model]
		if model == qr.BestModel {
			bestCount++
		}
	}

	n := float64(len(queries))
	return StrategyResult{
		Name:         fmt.Sprintf("Always %s", model),
		AvgQuality:   totalQuality / n,
		AvgLatency:   totalLatency / n,
		BestModelPct: float64(bestCount) / n * 100,
	}
}

func printResults(results []StrategyResult, numQueries, numModels int) {
	// Sort by avg quality descending
	sort.Slice(results, func(i, j int) bool {
		return results[i].AvgQuality > results[j].AvgQuality
	})

	fmt.Println()
	fmt.Printf("Validation Results (%d test queries, %d models)\n", numQueries, numModels)
	fmt.Println("=" + strings.Repeat("=", 69))
	fmt.Printf("%-26s %12s %12s %14s\n", "Strategy", "Avg Quality", "Avg Latency", "Best Model %")
	fmt.Println("-" + strings.Repeat("-", 69))

	for _, r := range results {
		fmt.Printf("%-26s %12.3f %11.2fs %13.1f%%\n",
			r.Name, r.AvgQuality, r.AvgLatency, r.BestModelPct)
	}
	fmt.Println("=" + strings.Repeat("=", 69))

	// Calculate ML benefit over random
	var randomResult *StrategyResult
	for i := range results {
		if results[i].Name == "Random Selection" {
			randomResult = &results[i]
			break
		}
	}

	if randomResult != nil {
		fmt.Println("\nML Routing Benefit:")
		for _, r := range results {
			if strings.Contains(r.Name, "Selection") && r.Name != "Random Selection" {
				qualityImprovement := 0.0
				if randomResult.AvgQuality > 0 {
					qualityImprovement = (r.AvgQuality - randomResult.AvgQuality) / randomResult.AvgQuality * 100
				}
				bestImprovement := 0.0
				if randomResult.BestModelPct > 0 {
					bestImprovement = r.BestModelPct / randomResult.BestModelPct
				}
				fmt.Printf("  - %s improves quality by %+.1f%% over random\n", r.Name, qualityImprovement)
				fmt.Printf("  - %s selects best model %.1fx more often than random\n", r.Name, bestImprovement)
			}
		}
	}

	fmt.Println()
	fmt.Println("Note: This validation uses the ACTUAL production Go/Rust selectors.")
	fmt.Printf("Timestamp: %s\n", time.Now().Format(time.RFC3339))
}
