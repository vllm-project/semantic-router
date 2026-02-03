package commands

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/spf13/cobra"
)

var (
	testSelectorAlgorithm string
	testSelectorCategory  string
	testSelectorModelsDir string
)

var testSelectorCmd = &cobra.Command{
	Use:   "test-selector [query]",
	Short: "Test trained model selection algorithms",
	Long: `Test the trained model selection algorithms by showing model info.

This command loads a trained model and displays its training statistics.

Examples:
  vsr test-selector --algorithm knn "What is 2+2?"
  vsr test-selector --algorithm kmeans --category math "Solve x^2 + 5x + 6 = 0"`,
	Args: cobra.ExactArgs(1),
	RunE: runTestSelector,
}

// NewTestSelectorCmd creates the test-selector command
func NewTestSelectorCmd() *cobra.Command {
	testSelectorCmd.Flags().StringVar(&testSelectorAlgorithm, "algorithm", "knn",
		"Algorithm to test: knn, kmeans, svm")
	testSelectorCmd.Flags().StringVar(&testSelectorCategory, "category", "other",
		"Category for the query (biology, math, physics, etc.)")
	testSelectorCmd.Flags().StringVar(&testSelectorModelsDir, "models-dir",
		"src/semantic-router/pkg/modelselection/data/trained_models",
		"Directory containing trained models")

	return testSelectorCmd
}

// ModelInfo contains basic model metadata
type ModelInfo struct {
	Version   string `json:"version"`
	Algorithm string `json:"algorithm"`
	Training  []struct {
		QueryEmbedding  []float64 `json:"query_embedding"`
		SelectedModel   string    `json:"selected_model"`
		ResponseQuality float64   `json:"response_quality"`
	} `json:"training"`
}

func runTestSelector(cmd *cobra.Command, args []string) error {
	query := args[0]

	fmt.Println("=" + strings.Repeat("=", 59))
	fmt.Println("  Model Selection Test")
	fmt.Println("=" + strings.Repeat("=", 59))
	fmt.Printf("Query:     %s\n", query)
	fmt.Printf("Category:  %s\n", testSelectorCategory)
	fmt.Printf("Algorithm: %s\n", testSelectorAlgorithm)
	fmt.Println()

	// Test all algorithms
	algorithms := []string{"knn", "kmeans", "svm"}

	fmt.Println("Trained Model Statistics:")
	fmt.Println("-" + strings.Repeat("-", 59))

	for _, alg := range algorithms {
		modelPath := fmt.Sprintf("%s/%s_model.json", testSelectorModelsDir, alg)

		// Read the model file
		data, err := os.ReadFile(modelPath)
		if err != nil {
			fmt.Printf("  %-8s: ❌ Not found\n", strings.ToUpper(alg))
			continue
		}

		var info ModelInfo
		if err := json.Unmarshal(data, &info); err != nil {
			fmt.Printf("  %-8s: ❌ Parse error\n", strings.ToUpper(alg))
			continue
		}

		// Count models in training data
		modelCounts := make(map[string]int)
		for _, t := range info.Training {
			if t.SelectedModel != "" {
				modelCounts[t.SelectedModel]++
			}
		}

		fmt.Printf("  %-8s: ✅ %d training records\n", strings.ToUpper(alg), len(info.Training))

		if len(modelCounts) > 0 {
			fmt.Printf("           Models: ")
			first := true
			for model, count := range modelCounts {
				if !first {
					fmt.Printf(", ")
				}
				fmt.Printf("%s(%d)", model, count)
				first = false
			}
			fmt.Println()
		}
	}

	fmt.Println()
	fmt.Println("-" + strings.Repeat("-", 59))
	fmt.Println("Note: Full inference testing requires VSR to be running.")
	fmt.Println("The trained models are automatically used during routing.")
	fmt.Println("-" + strings.Repeat("-", 59))

	return nil
}
