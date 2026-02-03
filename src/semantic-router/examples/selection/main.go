/*
Selection Demo - Demonstrates advanced model selection methods

This demo exercises the actual selection package code.
Run with: cd src/semantic-router && go run ./examples/selection/main.go

Logs are printed to show the decision-making process.
*/
package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/selection"
)

func init() {
	// Initialize logging to see selection decisions
	os.Setenv("LOG_LEVEL", "info")
	_, _ = logging.InitLoggerFromEnv()
}

func main() {
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("Demo: Advanced Model Selection Methods")
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
	fmt.Println()
	fmt.Println("Command: cd src/semantic-router && go run ./examples/selection/main.go")
	fmt.Println()
	fmt.Println("This demo calls the actual selection package code.")
	fmt.Println("Log lines starting with [EloSelector], [AutoMix], [RouterDC], [HybridSelector]")
	fmt.Println("show the real decision-making process.")
	fmt.Println()

	// Define test models (these are model IDs, not requiring actual models to be running)
	candidates := []config.ModelRef{
		{Model: "llama3.2:3b"},
		{Model: "phi4"},
		{Model: "gemma3:27b"},
	}

	fmt.Println("Candidate Models (for demonstration):")
	fmt.Println("  - llama3.2:3b (small, cheap)")
	fmt.Println("  - phi4 (medium)")
	fmt.Println("  - gemma3:27b (large, expensive)")
	fmt.Println()

	// Demo 1: Static Selection
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ DEMO 1: Static Selection (Baseline - BEFORE)                                  │")
	fmt.Println("│ Always picks first model or highest configured score                          │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────┘")
	staticSelector := selection.NewStaticSelector(&selection.StaticConfig{})
	demoSelector(staticSelector, candidates, "How do I fix a memory leak?")

	// Demo 2: Elo Selection
	fmt.Println()
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ DEMO 2: Elo Rating Selection                                                  │")
	fmt.Println("│ Models have ratings based on user preference feedback                         │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────┘")
	eloSelector := selection.NewEloSelector(&selection.EloConfig{
		InitialRating:    1500,
		KFactor:          32,
		CategoryWeighted: true,
	})

	// Simulate some feedback to adjust ratings
	fmt.Println("\nSimulating user feedback to adjust Elo ratings...")
	fmt.Println("  - gemma3:27b beats llama3.2:3b")
	fmt.Println("  - gemma3:27b beats phi4")
	fmt.Println("  - phi4 beats llama3.2:3b")
	ctx := context.Background()
	_ = eloSelector.UpdateFeedback(ctx, &selection.Feedback{
		WinnerModel:  "gemma3:27b",
		LoserModel:   "llama3.2:3b",
		DecisionName: "tech",
	})
	_ = eloSelector.UpdateFeedback(ctx, &selection.Feedback{
		WinnerModel:  "gemma3:27b",
		LoserModel:   "phi4",
		DecisionName: "tech",
	})
	_ = eloSelector.UpdateFeedback(ctx, &selection.Feedback{
		WinnerModel:  "phi4",
		LoserModel:   "llama3.2:3b",
		DecisionName: "tech",
	})

	// Show current ratings
	leaderboard := eloSelector.GetLeaderboard("tech")
	fmt.Println("\nCurrent Elo Ratings (after feedback):")
	for _, entry := range leaderboard {
		fmt.Printf("  %s: %.0f (W:%d L:%d)\n", entry.Model, entry.Rating, entry.Wins, entry.Losses)
	}

	demoSelector(eloSelector, candidates, "How do I fix a memory leak?")

	// Demo 3: AutoMix Selection - Cost vs Quality
	fmt.Println()
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ DEMO 3: AutoMix Selection - Cost-Quality Tradeoff                             │")
	fmt.Println("│ Shows how different cost_quality_tradeoff values affect selection             │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────┘")

	// Set model capabilities for AutoMix
	fmt.Println("\nModel capabilities (for cost-quality tradeoff):")
	fmt.Println("  llama3.2:3b: cost=$0.05/1M, quality=0.70 (small, cheap)")
	fmt.Println("  phi4:        cost=$0.15/1M, quality=0.85 (medium)")
	fmt.Println("  gemma3:27b:  cost=$0.50/1M, quality=0.95 (large, expensive)")

	// Helper to set up capabilities
	setupAutoMix := func(tradeoff float64) *selection.AutoMixSelector {
		am := selection.NewAutoMixSelector(&selection.AutoMixConfig{
			CostQualityTradeoff: tradeoff,
			CostAwareRouting:    true,
		})
		am.SetCapability("llama3.2:3b", &selection.ModelCapability{
			Model: "llama3.2:3b", ParamSize: 3.0, Cost: 0.05, AvgQuality: 0.70,
		})
		am.SetCapability("phi4", &selection.ModelCapability{
			Model: "phi4", ParamSize: 14.0, Cost: 0.15, AvgQuality: 0.85,
		})
		am.SetCapability("gemma3:27b", &selection.ModelCapability{
			Model: "gemma3:27b", ParamSize: 27.0, Cost: 0.50, AvgQuality: 0.95,
		})
		return am
	}

	// Low cost weight (prefer quality)
	fmt.Println("\n>>> Config: cost_quality_tradeoff = 0.2 (PREFER QUALITY)")
	autoMixQuality := setupAutoMix(0.2)
	demoSelector(autoMixQuality, candidates, "Explain quantum computing in detail")

	// High cost weight (prefer cost)
	fmt.Println("\n>>> Config: cost_quality_tradeoff = 0.8 (PREFER COST)")
	autoMixCost := setupAutoMix(0.8)
	demoSelector(autoMixCost, candidates, "What is 2+2?")

	// Demo 4: RouterDC Selection
	fmt.Println()
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ DEMO 4: RouterDC Selection - Query-to-Model Matching                          │")
	fmt.Println("│ Matches query embeddings to model capability embeddings                       │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────┘")
	routerDC := selection.NewRouterDCSelector(&selection.RouterDCConfig{
		Temperature:   0.07,
		MinSimilarity: 0.3,
	})

	// Set model embeddings (normally learned, here we configure them)
	fmt.Println("\nSetting model capability embeddings:")
	fmt.Println("  phi4         → optimized for [code, debugging, technical] queries")
	fmt.Println("  gemma3:27b   → optimized for [reasoning, analysis, complex] queries")
	fmt.Println("  llama3.2:3b  → general purpose (balanced)")
	routerDC.SetModelEmbedding("phi4", []float32{0.9, 0.85, 0.88, 0.5, 0.4})       // Code-focused
	routerDC.SetModelEmbedding("gemma3:27b", []float32{0.5, 0.4, 0.3, 0.95, 0.92}) // Reasoning-focused
	routerDC.SetModelEmbedding("llama3.2:3b", []float32{0.6, 0.6, 0.6, 0.6, 0.6})  // Balanced

	// Test with code query (embedding similar to code domain)
	fmt.Println("\n>>> Query: Code/Debugging (embedding: [0.85, 0.9, 0.88, 0.4, 0.3])")
	codeQuery := "Debug this Go function that has a nil pointer dereference"
	demoSelectorWithEmbedding(routerDC, candidates, codeQuery, []float32{0.85, 0.9, 0.88, 0.4, 0.3})

	// Test with reasoning query (embedding similar to reasoning domain)
	fmt.Println("\n>>> Query: Reasoning/Analysis (embedding: [0.3, 0.4, 0.2, 0.92, 0.88])")
	reasoningQuery := "Analyze the philosophical implications of AI consciousness"
	demoSelectorWithEmbedding(routerDC, candidates, reasoningQuery, []float32{0.3, 0.4, 0.2, 0.92, 0.88})

	fmt.Println()
	fmt.Println("⚠️  RouterDC LIMITATION NOTE:")
	fmt.Println("   Demo uses simple 5-dimension embeddings for illustration.")
	fmt.Println("   For production, model embeddings should be:")
	fmt.Println("   - Pre-computed from benchmark results / model capabilities")
	fmt.Println("   - Or learned via dual-contrastive training (see RouterDC paper)")
	fmt.Println("   The mechanism works - production needs real embeddings.")

	// Demo 5: Hybrid Selection
	fmt.Println()
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ DEMO 5: Hybrid Selection - Combines All Methods                               │")
	fmt.Println("│ Weights: elo=0.3, routerdc=0.3, automix=0.2, cost=0.2                         │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────┘")

	// Create hybrid with all component selectors
	hybridSelector := selection.NewHybridSelectorWithComponents(&selection.HybridConfig{
		EloWeight:      0.3,
		RouterDCWeight: 0.3,
		AutoMixWeight:  0.2,
		CostWeight:     0.2,
	}, eloSelector, routerDC, autoMixQuality)

	fmt.Println("\nCombining scores from:")
	fmt.Println("  - Elo ratings (gemma3:27b has highest)")
	fmt.Println("  - RouterDC similarity (depends on query)")
	fmt.Println("  - AutoMix cost-quality (balanced)")

	demoSelectorWithEmbedding(hybridSelector, candidates, "Write an efficient sorting algorithm", []float32{0.8, 0.85, 0.9, 0.5, 0.4})

	// Demo 6: ML-based Selectors (KNN, KMeans, SVM)
	fmt.Println()
	fmt.Println("┌────────────────────────────────────────────────────────────────────────────────┐")
	fmt.Println("│ DEMO 6: ML-based Selectors (KNN, KMeans, SVM)                                  │")
	fmt.Println("│ Machine learning algorithms for query-to-model routing                        │")
	fmt.Println("└────────────────────────────────────────────────────────────────────────────────┘")

	fmt.Println("\nML Selectors require training data (embeddings + quality scores).")
	fmt.Println("These are created via Factory.CreateAll() and use pretrained models.")
	fmt.Println()
	fmt.Println("Available ML methods (configured per-decision in YAML):")
	fmt.Println("  - knn:    Quality-weighted voting among K nearest neighbors")
	fmt.Println("  - kmeans: Cluster-based routing with quality+latency scores")
	fmt.Println("  - svm:    RBF kernel decision boundaries between model preferences")
	fmt.Println()
	fmt.Println("Example config:")
	fmt.Println("  decisions:")
	fmt.Println("    - name: \"math_decision\"")
	fmt.Println("      algorithm:")
	fmt.Println("        type: \"knn\"")
	fmt.Println("        knn:")
	fmt.Println("          k: 5")
	fmt.Println()

	// Create ML adapters (without pretrained models, they use fallback selection)
	knnAdapter, err := selection.CreateKNNSelector(selection.DefaultMLSelectorConfig(), nil)
	if err != nil {
		fmt.Printf("⚠️  Failed to create KNN adapter: %v\n", err)
	} else {
		fmt.Println(">>> KNN Selector (without training data - uses fallback)")
		demoSelector(knnAdapter, candidates, "What is machine learning?")
	}

	fmt.Println()
	fmt.Println("📝 NOTE: For full ML selector functionality, train models using Python:")
	fmt.Println("   cd src/training/ml_model_selection")
	fmt.Println("   python train.py --data-file benchmark.jsonl --output-dir models/")
	fmt.Println("   Or download from HuggingFace:")
	fmt.Println("   python download_model.py --output-dir models/")
	fmt.Println("   Then configure pretrained_path in your config YAML.")

	fmt.Println()
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
	fmt.Println("✅ DEMO COMPLETE - All selection methods demonstrated with REAL code execution")
	fmt.Println("════════════════════════════════════════════════════════════════════════════════")
}

func demoSelector(selector selection.Selector, candidates []config.ModelRef, query string) {
	ctx := context.Background()
	selCtx := &selection.SelectionContext{
		Query:           query,
		CandidateModels: candidates,
		DecisionName:    "tech",
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	printResult(query, result)
}

func demoSelectorWithEmbedding(selector selection.Selector, candidates []config.ModelRef, query string, embedding []float32) {
	ctx := context.Background()
	selCtx := &selection.SelectionContext{
		Query:           query,
		QueryEmbedding:  embedding,
		CandidateModels: candidates,
		DecisionName:    "tech",
	}

	result, err := selector.Select(ctx, selCtx)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		return
	}

	printResult(query, result)
}

func printResult(query string, result *selection.SelectionResult) {
	fmt.Println()
	fmt.Printf("Query: \"%s\"\n", truncate(query, 60))
	fmt.Println()
	fmt.Println("Selection Result:")
	fmt.Printf("  ✅ SELECTED MODEL: %s\n", result.SelectedModel)
	fmt.Printf("  📊 Score: %.4f\n", result.Score)
	fmt.Printf("  🎯 Confidence: %.4f\n", result.Confidence)
	fmt.Printf("  🔧 Method: %s\n", result.Method)
	fmt.Printf("  💭 Reasoning: %s\n", result.Reasoning)

	if len(result.AllScores) > 0 {
		fmt.Println()
		fmt.Println("  All Candidate Scores:")
		scoresJSON, _ := json.MarshalIndent(result.AllScores, "    ", "  ")
		fmt.Printf("    %s\n", scoresJSON)
	}
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen-3] + "..."
}
