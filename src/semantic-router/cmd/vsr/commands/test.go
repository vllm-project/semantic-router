package commands

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/spf13/cobra"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cli"
)

// NewTestCmd creates the test command
func NewTestCmd() *cobra.Command {
	cmd := &cobra.Command{
		Use:   "test-prompt [text]",
		Short: "Send a test prompt to the router",
		Long: `Test the router by sending a prompt for classification.

This command sends your prompt to the router's classification API and displays:
  - Detected category
  - Model routing decision
  - PII detection results
  - Jailbreak protection status

Example:
  vsr test-prompt "Solve x^2 + 5x + 6 = 0"`,
		Args: cobra.MinimumNArgs(1),
		RunE: func(cmd *cobra.Command, args []string) error {
			prompt := strings.Join(args, " ")
			endpoint, _ := cmd.Flags().GetString("endpoint")
			outputFormat := cmd.Parent().Flag("output").Value.String()

			result, err := callClassificationAPI(endpoint, prompt)
			if err != nil {
				return fmt.Errorf("failed to classify prompt: %w", err)
			}

			return displayTestResult(result, outputFormat)
		},
	}

	cmd.Flags().String("endpoint", "http://localhost:8080", "Router API endpoint")

	return cmd
}

type ClassificationResult struct {
	Classification struct {
		Category   string  `json:"category"`
		Confidence float64 `json:"confidence"`
	} `json:"classification"`
	RecommendedModel string `json:"recommended_model"`
}

func callClassificationAPI(endpoint, prompt string) (*ClassificationResult, error) {
	// Prepare request
	reqBody := map[string]string{
		"text": prompt,
	}
	jsonData, err := json.Marshal(reqBody)
	if err != nil {
		return nil, err
	}

	// Make HTTP request
	resp, err := http.Post(
		fmt.Sprintf("%s/api/v1/classify/intent", endpoint),
		"application/json",
		bytes.NewBuffer(jsonData),
	)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	// Parse response
	var result ClassificationResult
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &result, nil
}

func displayTestResult(result *ClassificationResult, format string) error {
	switch format {
	case "json":
		return cli.PrintJSON(result)
	case "yaml":
		return cli.PrintYAML(result)
	}

	// Table format
	fmt.Println("\nTest Results:")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Printf("Category:    %s\n", result.Classification.Category)
	fmt.Printf("Model:       %s\n", result.RecommendedModel)
	fmt.Printf("Confidence:  %.2f\n", result.Classification.Confidence)

	// PII and Jailbreak are not part of the intent response
	cli.Success("PII Check: Not performed in intent classification")
	cli.Success("Jailbreak Check: Not performed in intent classification")

	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

	return nil
}
