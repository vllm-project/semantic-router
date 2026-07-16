package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"regexp"
	"strings"
	"time"

	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("complexity-model-routing", pkgtestcases.TestCase{
		Description: "Boot the router with a method: model complexity rule and verify the emitted rule:easy|medium|hard signal",
		Tags:        []string{"kubernetes", "routing", "complexity", "model"},
		Fn:          testComplexityModelRouting,
	})
}

// The complexity signal is emitted (comma-joined) on the x-vsr-matched-complexity
// response header as "<rule>:<difficulty>", where difficulty is one of the three
// trained classes. This is the behavioral surface a model-mode rule must produce.
const complexityMatchedHeader = "x-vsr-matched-complexity"

var complexityMatchPattern = regexp.MustCompile(`^[^:]+:(easy|medium|hard)$`)

// ComplexityModelRoutingCase is a single model-mode complexity test case.
type ComplexityModelRoutingCase struct {
	Name         string `json:"name"`
	Description  string `json:"description"`
	Query        string `json:"query"`
	ExpectedRule string `json:"expected_rule"`
}

func testComplexityModelRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing model-mode complexity signal emission")
	}

	localPort, stopPortForward, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stopPortForward()

	cases, err := loadComplexityModelRoutingCases("e2e/testcases/testdata/complexity_model_routing_cases.json")
	if err != nil {
		return fmt.Errorf("failed to load test cases: %w", err)
	}

	total := 0
	emitted := 0
	var failures []string

	for _, tc := range cases {
		total++
		response, err := sendLocalChatCompletion(ctx, localPort, "MoM", tc.Query, 60*time.Second)
		if err != nil {
			failures = append(failures, fmt.Sprintf("%s: request error: %v", tc.Name, err))
			continue
		}
		if response.StatusCode != http.StatusOK {
			logUnexpectedChatCompletionStatus(opts.Verbose, response, "test case: "+tc.Name, "Query: "+tc.Query)
			failures = append(failures, fmt.Sprintf("%s: %s", tc.Name, formatUnexpectedChatCompletionStatus(response)))
			continue
		}

		header := response.Headers.Get(complexityMatchedHeader)
		match := firstComplexityMatchForRule(header, tc.ExpectedRule)
		if match == "" {
			failures = append(failures, fmt.Sprintf(
				"%s: expected a %q:<easy|medium|hard> value on %s, got %q",
				tc.Name, tc.ExpectedRule, complexityMatchedHeader, header))
			continue
		}

		emitted++
		if opts.Verbose {
			fmt.Printf("[Test] %s -> %s=%q (matched %q)\n", tc.Name, complexityMatchedHeader, header, match)
		}
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"total_tests":     total,
			"signals_emitted": emitted,
			"failed_tests":    total - emitted,
		})
	}

	if len(failures) > 0 {
		fmt.Println("\n[Test] Complexity model-mode failures:")
		for _, f := range failures {
			fmt.Printf("  - %s\n", f)
		}
	}

	if emitted == 0 {
		return fmt.Errorf("complexity model-mode test failed: no valid rule:difficulty signal emitted (0/%d)", total)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Complexity model-mode test completed: %d/%d cases emitted a valid signal\n", emitted, total)
	}
	return nil
}

func loadComplexityModelRoutingCases(path string) ([]ComplexityModelRoutingCase, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read test cases file: %w", err)
	}
	var cases []ComplexityModelRoutingCase
	if err := json.Unmarshal(data, &cases); err != nil {
		return nil, fmt.Errorf("failed to parse test cases: %w", err)
	}
	return cases, nil
}

// firstComplexityMatchForRule returns the first comma-separated header entry that
// belongs to expectedRule and has a valid easy/medium/hard difficulty suffix, or
// "" if none is present.
func firstComplexityMatchForRule(header, expectedRule string) string {
	if header == "" {
		return ""
	}
	for _, raw := range strings.Split(header, ",") {
		entry := strings.TrimSpace(raw)
		if !complexityMatchPattern.MatchString(entry) {
			continue
		}
		if expectedRule == "" || strings.HasPrefix(entry, expectedRule+":") {
			return entry
		}
	}
	return ""
}
