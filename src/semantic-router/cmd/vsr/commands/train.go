/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package commands

import (
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/spf13/cobra"
	"gopkg.in/yaml.v3"

	candle_binding "github.com/vllm-project/semantic-router/candle-binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelselection"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

var trainCmd = &cobra.Command{
	Use:   "train",
	Short: "Train model selection algorithms (DEPRECATED - use Python training)",
	Long: `⚠️  DEPRECATED: This command is deprecated. Use Python training instead:

    cd src/training/ml_model_selection
    python train.py --data-file benchmark.jsonl --output-dir models/

Or download pretrained models from HuggingFace:

    python download_model.py --output-dir models/

See: src/training/ml_model_selection/README.md

---

[LEGACY] Train the 3 model selection algorithms (KNN, KMeans, SVM)
to select the best LLM for each query type.

TRAINING FLOW:
  1. Configure LLMs and API keys in your config file (model_config section)
  2. Run training with --benchmark to call your LLMs and collect performance data
  3. System generates embeddings + trains all 5 ML algorithms
  4. Use trained models for online inference

MODES:
  --benchmark     Run LIVE benchmarks against your LLM endpoints
                  (e.g., 10 models × 5000 queries = 50,000 API calls)

  (no flag)       Use existing pre-benchmarked training data
                  (no API calls, uses training_data_with_category.jsonl)

CONFIG REQUIREMENTS (for --benchmark mode):
  model_config:
    "your-model":
      preferred_endpoints: ["nvidia"]     # Which endpoint to use
      external_model_ids:
        nvidia: "provider/model-id"       # Model ID for that endpoint
      access_key: "optional-per-model-key"

  vllm_endpoints:
    - name: "nvidia"
      address: "integrate.api.nvidia.com"
      port: 443
      type: "nvidia"
      api_key: "nvapi-xxx"  # Or set NVIDIA_API_KEY env var

EXAMPLES:
  # Analyze config (no training)
  vsr train --config config/config.yaml --analyze-only

  # Train using existing benchmark data (no API calls)
  vsr train --config config/config.yaml

  # Run LIVE benchmarks and train (requires API access)
  vsr train --config config/config.yaml --benchmark

  # Limit queries for testing (100 queries × N models)
  vsr train --config config/config.yaml --benchmark --query-limit 100`,
	RunE: runTrain,
}

var (
	trainDataFile       string
	trainOutputDir      string
	trainEmbeddingModel string
	trainEmbeddingDim   int
	trainConfigFile     string
	trainAnalyzeOnly    bool
	trainQueryLimit     int
	trainRunBenchmark   bool
	trainConcurrency    int
	trainRateLimitDelay int
	trainUseFallback    bool
	// Algorithm hyperparameters
	trainKnnK              int
	trainKmeansNumClusters int
	trainQualityWeight     float64 // Global quality weight for all algorithms
)

func init() {
	trainCmd.Flags().StringVar(&trainDataFile, "data-file",
		"pkg/modelselection/data/training_data_with_category.jsonl",
		"Path to training data file (JSONL format)")
	trainCmd.Flags().StringVar(&trainOutputDir, "output-dir",
		"pkg/modelselection/data/trained_models",
		"Directory to save trained models")
	trainCmd.Flags().StringVar(&trainEmbeddingModel, "embedding-model",
		"qwen3",
		"Embedding model to use: qwen3, gemma, or bert")
	trainCmd.Flags().IntVar(&trainEmbeddingDim, "embedding-dim",
		768,
		"Embedding dimension (128, 256, 512, 768, or 1024)")
	trainCmd.Flags().BoolVar(&trainUseFallback, "use-fallback",
		false,
		"Use hash-based fallback embeddings (no Candle library required)")
	trainCmd.Flags().StringVar(&trainConfigFile, "config",
		"",
		"Path to VSR config file (analyzes which categories need model selection)")
	trainCmd.Flags().BoolVar(&trainAnalyzeOnly, "analyze-only",
		false,
		"Only analyze config for multi-model categories, don't train")
	trainCmd.Flags().IntVar(&trainQueryLimit, "query-limit",
		0,
		"Limit number of queries to benchmark (0 = no limit, use for testing)")
	trainCmd.Flags().BoolVar(&trainRunBenchmark, "benchmark",
		false,
		"Run live benchmark against LLM endpoints (requires running models)")
	trainCmd.Flags().IntVar(&trainConcurrency, "concurrency",
		1,
		"Number of concurrent API requests (1 for free tier, 4+ for paid APIs)")
	trainCmd.Flags().IntVar(&trainRateLimitDelay, "rate-limit-delay",
		500,
		"Delay between requests in milliseconds (500 for free tier, 0 for paid APIs)")

	// Algorithm hyperparameters
	trainCmd.Flags().IntVar(&trainKnnK, "knn-k",
		5,
		"KNN: Number of neighbors to consider (default: 5)")
	trainCmd.Flags().IntVar(&trainKmeansNumClusters, "kmeans-clusters",
		8,
		"KMeans: Number of clusters (default: 8, or number of models)")
	trainCmd.Flags().Float64Var(&trainQualityWeight, "quality-weight",
		0.9,
		"Global quality vs speed weight for all algorithms: 0-1 (0=pure speed, 1=pure quality, default: 0.9)")
}

// buildHyperparams builds AlgorithmHyperparams from command-line flags
func buildHyperparams() (modelselection.AlgorithmHyperparams, error) {
	// Validate quality weight is in range [0, 1]
	if trainQualityWeight < 0 || trainQualityWeight > 1 {
		return modelselection.AlgorithmHyperparams{}, fmt.Errorf("--quality-weight must be between 0 and 1, got %f", trainQualityWeight)
	}

	return modelselection.AlgorithmHyperparams{
		QualityWeight:     trainQualityWeight,
		KnnK:              trainKnnK,
		KmeansNumClusters: trainKmeansNumClusters,
	}, nil
}

func runTrain(cmd *cobra.Command, args []string) error {
	startTime := time.Now()

	// If config file is provided, analyze it first
	if trainConfigFile != "" {
		return runConfigDrivenTraining(cmd, args)
	}

	// Validate inputs
	if _, err := os.Stat(trainDataFile); os.IsNotExist(err) {
		return fmt.Errorf("training data file not found: %s", trainDataFile)
	}

	// Create output directory if needed
	if err := os.MkdirAll(trainOutputDir, 0o755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}

	// Build hyperparameters from flags
	hyperparams, err := buildHyperparams()
	if err != nil {
		return err
	}

	fmt.Println("==============================================")
	fmt.Println("  VSR Model Selection Training")
	fmt.Println("==============================================")
	fmt.Printf("Data file:        %s\n", trainDataFile)
	fmt.Printf("Output dir:       %s\n", trainOutputDir)
	fmt.Printf("Embedding model:  %s\n", trainEmbeddingModel)
	fmt.Printf("Embedding dim:    %d\n", trainEmbeddingDim)
	fmt.Printf("Feature dim:      %d (embedding + 14 category one-hot)\n", trainEmbeddingDim+14)
	fmt.Println("----------------------------------------------")
	fmt.Println("Algorithm Hyperparameters:")
	fmt.Printf("  Quality weight:     %.2f (%.0f%% quality, %.0f%% speed)\n",
		hyperparams.QualityWeight,
		hyperparams.QualityWeight*100,
		(1-hyperparams.QualityWeight)*100)
	fmt.Printf("  KNN k:              %d\n", hyperparams.KnnK)
	fmt.Printf("  KMeans clusters:    %d\n", hyperparams.KmeansNumClusters)
	fmt.Println("----------------------------------------------")

	// Create trainer with appropriate embedding mode
	var trainer *modelselection.Trainer
	if trainUseFallback {
		// Use hash-based fallback (no Candle required)
		trainer = modelselection.NewTrainerWithFallback(trainEmbeddingDim)
		trainer.SetHyperparams(hyperparams)
		fmt.Println("\n[1/4] Setting up embedding provider...")
		fmt.Printf("      Using hash-based fallback embeddings (dim=%d)\n", trainEmbeddingDim)
		fmt.Println("      For real embeddings, remove --use-fallback flag")
	} else {
		// Use Candle for real embeddings (Qwen3 by default)
		fmt.Println("\n[1/4] Setting up embedding provider...")
		fmt.Printf("      Initializing Candle embedding models...\n")

		// Initialize Candle embedding models before using them
		qwen3ModelPath := "models/mom-embedding-pro"
		if initErr := candle_binding.InitEmbeddingModelsBatched(qwen3ModelPath, 32, 100, true); initErr != nil {
			logging.Warnf("Failed to initialize batched embeddings: %v", initErr)
			// Try standard initialization as fallback
			gemmaModelPath := "models/embeddinggemma-300m"
			if initErr2 := candle_binding.InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, "", true); initErr2 != nil {
				logging.Warnf("Failed to initialize Candle embeddings: %v. Falling back to hash-based.", initErr2)
				trainer = modelselection.NewTrainerWithFallback(trainEmbeddingDim)
				trainer.SetHyperparams(hyperparams)
				fmt.Printf("      ⚠️  Candle init failed, using hash-based fallback (dim=%d)\n", trainEmbeddingDim)
				goto skipCandleInit
			}
		}
		fmt.Printf("      ✅ Candle embedding models initialized\n")

		trainer = modelselection.NewTrainer(trainEmbeddingDim)
		trainer.SetEmbeddingModel(trainEmbeddingModel)
		trainer.SetHyperparams(hyperparams)
		fmt.Printf("      Using Candle with %s model (dim=%d)\n", trainEmbeddingModel, trainEmbeddingDim)
	}

skipCandleInit:

	// Load training data
	fmt.Println("\n[2/4] Loading training data...")
	if loadErr := trainer.LoadBenchmarkData(trainDataFile); loadErr != nil {
		return fmt.Errorf("failed to load training data: %w", loadErr)
	}
	fmt.Printf("      Loaded %d unique queries\n", len(trainer.RoutingData))

	// Show category distribution
	categoryCounts := make(map[string]int)
	for _, rd := range trainer.RoutingData {
		categoryCounts[rd.QueryType]++
	}
	fmt.Println("      Category distribution:")
	for cat, count := range categoryCounts {
		fmt.Printf("        - %s: %d queries\n", cat, count)
	}

	// Show LLM models found
	if trainer.LLMCandidates != nil {
		fmt.Printf("\n      Found %d LLM models:\n", len(trainer.LLMCandidates.LLMCandidates))
		for modelName, candidate := range trainer.LLMCandidates.LLMCandidates {
			fmt.Printf("        - %s (avg quality: %.2f, avg latency: %.0fms)\n",
				modelName, candidate.QualityScore, candidate.AvgLatencyMs)
		}
	}

	// Convert to training records (generates embeddings + category one-hot)
	fmt.Println("\n[3/4] Generating feature vectors (embedding + category)...")
	trainingRecords := trainer.ConvertToTrainingRecords()
	fmt.Printf("      Generated %d training records\n", len(trainingRecords))
	if len(trainingRecords) > 0 {
		fmt.Printf("      Feature dimension: %d\n", len(trainingRecords[0].QueryEmbedding))
	}

	// Train all algorithms
	fmt.Println("\n[4/4] Training algorithms...")

	// Train each algorithm and report progress
	algorithms := []string{"KNN", "KMeans", "SVM"}
	for i, alg := range algorithms {
		fmt.Printf("      [%d/%d] Training %s...\n", i+1, len(algorithms), alg)
	}

	if trainErr := trainer.TrainAllAlgorithms(trainOutputDir); trainErr != nil {
		return fmt.Errorf("failed to train algorithms: %w", trainErr)
	}

	// Summary
	elapsed := time.Since(startTime)
	fmt.Println("\n==============================================")
	fmt.Println("  Training Complete!")
	fmt.Println("==============================================")
	fmt.Printf("Total time:       %s\n", elapsed.Round(time.Second))
	fmt.Printf("Models saved to:  %s\n", trainOutputDir)
	fmt.Printf("Feature dim:      %d (768 embedding + 14 category)\n", trainer.FeatureDim)
	fmt.Println("\nTrained models:")

	// List saved files
	files, err := filepath.Glob(filepath.Join(trainOutputDir, "*.json"))
	if err == nil {
		for _, f := range files {
			info, _ := os.Stat(f)
			if info != nil {
				fmt.Printf("  - %s (%d bytes)\n", filepath.Base(f), info.Size())
			}
		}
	}

	fmt.Println("\nNext steps:")
	fmt.Println("  1. Models are ready for inference")
	fmt.Println("  2. Configure VSR to use trained models")
	fmt.Println("  3. Test with: vsr test-prompt \"your query\"")

	logging.Infof("Training completed in %s", elapsed)
	return nil
}

// runConfigDrivenTraining handles training based on VSR config analysis
func runConfigDrivenTraining(cmd *cobra.Command, args []string) error {
	startTime := time.Now()

	// Build hyperparameters from flags
	hyperparams, err := buildHyperparams()
	if err != nil {
		return err
	}

	fmt.Println("==============================================")
	fmt.Println("  VSR Config-Driven Model Selection Training")
	fmt.Println("==============================================")
	fmt.Printf("Config file:     %s\n", trainConfigFile)
	fmt.Printf("Training data:   %s\n", trainDataFile)
	fmt.Printf("Benchmark mode:  %v\n", trainRunBenchmark)
	fmt.Println("----------------------------------------------")
	fmt.Println("Algorithm Hyperparameters:")
	fmt.Printf("  Quality weight:     %.2f (%.0f%% quality, %.0f%% speed)\n",
		hyperparams.QualityWeight,
		hyperparams.QualityWeight*100,
		(1-hyperparams.QualityWeight)*100)
	fmt.Printf("  KNN k:              %d\n", hyperparams.KnnK)
	fmt.Printf("  KMeans clusters:    %d\n", hyperparams.KmeansNumClusters)
	fmt.Println("----------------------------------------------")

	// Analyze config
	fmt.Println("\n[1/5] Analyzing config for multi-model categories...")
	analysis, err := modelselection.AnalyzeConfigForModelSelection(trainConfigFile)
	if err != nil {
		return fmt.Errorf("failed to analyze config: %w", err)
	}

	// Print analysis summary
	analysis.PrintAnalysisSummary()

	// If analyze-only, stop here
	if trainAnalyzeOnly {
		fmt.Println("\n✅ Analysis complete (--analyze-only mode)")
		return nil
	}

	// Check if training is needed
	if !analysis.NeedsModelSelectionTraining() {
		fmt.Println("\n⏭️  No multi-model categories found. Training not needed.")
		return nil
	}

	// Load config for benchmark runner
	cfg, err := loadRouterConfig(trainConfigFile)
	if err != nil {
		return fmt.Errorf("failed to load config: %w", err)
	}

	var trainingDataPath string

	// Decide between live benchmark or existing data
	if trainRunBenchmark {
		// Run live benchmarks against LLM endpoints
		trainingDataPath, err = runLiveBenchmark(cfg, analysis, trainDataFile)
		if err != nil {
			return fmt.Errorf("benchmark failed: %w", err)
		}
	} else {
		// Use existing training data (no API calls)
		trainingDataPath = trainDataFile
		fmt.Println("\n[2/5] Using existing training data (no live benchmark)...")
		fmt.Println("      To run live benchmarks, add --benchmark flag")
	}

	// Create trainer with appropriate embedding mode
	fmt.Println("\n[3/5] Setting up trainer...")
	var trainer *modelselection.Trainer
	if trainUseFallback {
		trainer = modelselection.NewTrainerWithFallback(trainEmbeddingDim)
		trainer.SetHyperparams(hyperparams)
		fmt.Printf("      Using hash-based fallback embeddings (dim=%d)\n", trainEmbeddingDim)
	} else {
		// Initialize Candle embedding models before using them
		fmt.Printf("      Initializing Candle embedding models...\n")
		qwen3ModelPath := "models/mom-embedding-pro" // Default Qwen3 embedding model path
		if err := candle_binding.InitEmbeddingModelsBatched(qwen3ModelPath, 32, 100, true); err != nil {
			logging.Warnf("Failed to initialize batched embeddings, trying standard init: %v", err)
			// Try standard initialization as fallback
			gemmaModelPath := "models/embeddinggemma-300m"
			if err := candle_binding.InitEmbeddingModels(qwen3ModelPath, gemmaModelPath, "", true); err != nil {
				logging.Warnf("Failed to initialize Candle embeddings: %v. Falling back to hash-based.", err)
				trainer = modelselection.NewTrainerWithFallback(trainEmbeddingDim)
				trainer.SetHyperparams(hyperparams)
				fmt.Printf("      ⚠️  Candle init failed, using hash-based fallback (dim=%d)\n", trainEmbeddingDim)
				goto trainerReady
			}
		}
		fmt.Printf("      ✅ Candle embedding models initialized\n")
		trainer = modelselection.NewTrainer(trainEmbeddingDim)
		trainer.SetEmbeddingModel(trainEmbeddingModel)
		trainer.SetHyperparams(hyperparams)
		fmt.Printf("      Using Candle with %s model (dim=%d)\n", trainEmbeddingModel, trainEmbeddingDim)
	}

trainerReady:
	// Load training data filtered by models from config
	fmt.Printf("\n[4/5] Loading training data for configured models...\n")
	fmt.Printf("      Models to train: %v\n", analysis.ModelsNeedingTraining)

	// Apply query limit if specified
	if trainQueryLimit > 0 {
		fmt.Printf("      Query limit: %d\n", trainQueryLimit)
	}

	// Load training data filtered by models in config
	if err := trainer.LoadBenchmarkDataFiltered(trainingDataPath, analysis.ModelsNeedingTraining); err != nil {
		return fmt.Errorf("failed to load training data: %w", err)
	}
	fmt.Printf("      Loaded %d unique queries with benchmark data\n", len(trainer.RoutingData))

	// Show LLM models and their stats from training data
	if trainer.LLMCandidates != nil {
		fmt.Printf("\n      Model performance from training data:\n")
		for modelName, candidate := range trainer.LLMCandidates.LLMCandidates {
			fmt.Printf("        • %s (avg quality: %.2f, avg latency: %.0fms)\n",
				modelName, candidate.QualityScore, candidate.AvgLatencyMs)
		}
	}

	// Show category distribution
	categoryCounts := make(map[string]int)
	for _, rd := range trainer.RoutingData {
		categoryCounts[rd.QueryType]++
	}
	fmt.Println("\n      Category distribution:")
	for cat, count := range categoryCounts {
		fmt.Printf("        - %s: %d queries\n", cat, count)
	}

	// Train all algorithms
	fmt.Println("\n[5/5] Training model selection algorithms...")
	algorithms := []string{"KNN", "KMeans", "SVM"}
	for i, alg := range algorithms {
		fmt.Printf("      [%d/%d] Training %s...\n", i+1, len(algorithms), alg)
	}

	if err := trainer.TrainAllAlgorithms(trainOutputDir); err != nil {
		return fmt.Errorf("failed to train algorithms: %w", err)
	}

	// Summary
	elapsed := time.Since(startTime)
	fmt.Println("\n==============================================")
	fmt.Println("  Config-Driven Training Complete!")
	fmt.Println("==============================================")
	fmt.Printf("Total time:      %s\n", elapsed.Round(time.Second))
	fmt.Printf("Models trained:  %d\n", len(analysis.ModelsNeedingTraining))
	fmt.Printf("Queries used:    %d\n", len(trainer.RoutingData))
	fmt.Printf("Output dir:      %s\n", trainOutputDir)

	// List saved files
	files, _ := filepath.Glob(filepath.Join(trainOutputDir, "*.json"))
	if len(files) > 0 {
		fmt.Println("\nTrained models:")
		for _, f := range files {
			info, _ := os.Stat(f)
			if info != nil {
				fmt.Printf("  • %s (%d bytes)\n", filepath.Base(f), info.Size())
			}
		}
	}

	fmt.Println("\n📋 Next steps:")
	fmt.Println("  1. Models are ready for inference")
	fmt.Println("  2. Test with: vsr test-prompt --algorithm knn \"your query\"")

	logging.Infof("Config-driven training completed in %s", elapsed)
	return nil
}

// runLiveBenchmark runs live benchmarks against LLM endpoints and returns the path to generated training data
func runLiveBenchmark(cfg *config.RouterConfig, analysis *modelselection.ConfigAnalysisResult, sourceDataPath string) (string, error) {
	fmt.Println("\n[2/5] Running live benchmark against LLM endpoints...")

	// Clear previous benchmark data to start fresh
	benchmarkOutputPath := filepath.Join(filepath.Dir(sourceDataPath), "benchmark_training_data.jsonl")
	if _, err := os.Stat(benchmarkOutputPath); err == nil {
		os.Remove(benchmarkOutputPath)
		fmt.Printf("      Cleared previous benchmark data: %s\n", benchmarkOutputPath)
	}

	// Create benchmark runner with models needing training
	runner := modelselection.NewBenchmarkRunner(cfg, analysis.ModelsNeedingTraining)

	// Set concurrency and rate limit delay from command line
	runner.Concurrency = trainConcurrency
	runner.RateLimitDelayMs = trainRateLimitDelay

	// Set query limit if specified
	if trainQueryLimit > 0 {
		runner.QueryLimit = trainQueryLimit
		fmt.Printf("      Query limit set to: %d\n", trainQueryLimit)
	}

	// Load queries from existing training data
	fmt.Printf("      Loading queries from: %s\n", sourceDataPath)
	if err := runner.LoadQueriesFromTrainingData(sourceDataPath); err != nil {
		return "", fmt.Errorf("failed to load queries: %w", err)
	}
	fmt.Printf("      Loaded %d unique queries with categories\n", len(runner.Queries))

	// Show model endpoints
	fmt.Println("\n      Model endpoints:")
	for model, endpoint := range runner.ModelEndpoints {
		fmt.Printf("        • %s → %s\n", model, endpoint)
	}

	// Run benchmarks
	fmt.Printf("\n      Running benchmarks: %d queries × %d models = %d API calls\n",
		len(runner.Queries), len(runner.Models), len(runner.Queries)*len(runner.Models))

	err := runner.RunBenchmarks(func(completed, total int) {
		if completed%100 == 0 || completed == total {
			fmt.Printf("      Progress: %d/%d (%.1f%%)\n", completed, total, float64(completed)/float64(total)*100)
		}
	})
	if err != nil {
		return "", fmt.Errorf("benchmark failed: %w", err)
	}

	// Print statistics
	stats := runner.GetStatistics()
	fmt.Println("\n      Benchmark statistics:")
	if modelStats, ok := stats["models"].(map[string]map[string]int); ok {
		for model, counts := range modelStats {
			fmt.Printf("        • %s: %d success, %d failed\n", model, counts["success"], counts["failure"])
		}
	}

	// Save benchmark results as training data
	if err := runner.SaveTrainingData(benchmarkOutputPath); err != nil {
		return "", fmt.Errorf("failed to save benchmark data: %w", err)
	}
	fmt.Printf("      Saved benchmark data to: %s\n", benchmarkOutputPath)

	return benchmarkOutputPath, nil
}

// loadRouterConfig loads the router config from a YAML file
func loadRouterConfig(configPath string) (*config.RouterConfig, error) {
	data, err := os.ReadFile(configPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read config: %w", err)
	}

	var cfg config.RouterConfig
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	return &cfg, nil
}

// GetTrainCmd returns the train command for registration
func GetTrainCmd() *cobra.Command {
	return trainCmd
}
