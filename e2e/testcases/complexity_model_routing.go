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

	var failures []string
	emitted := 0
	for _, tc := range cases {
		match, failure := evaluateComplexityCase(ctx, localPort, tc, opts.Verbose)
		if failure != "" {
			failures = append(failures, failure)
			continue
		}
		emitted++
		if opts.Verbose {
			fmt.Printf("[Test] %s -> %s=%q\n", tc.Name, complexityMatchedHeader, match)
		}
	}

	reportComplexityResults(opts, len(cases), emitted, failures)

	if emitted == 0 {
		return fmt.Errorf("complexity model-mode test failed: no valid rule:difficulty signal emitted (0/%d)", len(cases))
	}
	return nil
}

// evaluateComplexityCase sends one case's query and returns the matched
// "<rule>:<difficulty>" value, or a non-empty failure description.
func evaluateComplexityCase(ctx context.Context, localPort string, tc ComplexityModelRoutingCase, verbose bool) (string, string) {
	response, err := sendLocalChatCompletion(ctx, localPort, "MoM", tc.Query, 60*time.Second)
	if err != nil {
		return "", fmt.Sprintf("%s: request error: %v", tc.Name, err)
	}
	if response.StatusCode != http.StatusOK {
		logUnexpectedChatCompletionStatus(verbose, response, "test case: "+tc.Name, "Query: "+tc.Query)
		return "", fmt.Sprintf("%s: %s", tc.Name, formatUnexpectedChatCompletionStatus(response))
	}
	header := response.Headers.Get(complexityMatchedHeader)
	match := firstComplexityMatchForRule(header, tc.ExpectedRule)
	if match == "" {
		return "", fmt.Sprintf("%s: expected a %q:<easy|medium|hard> value on %s, got %q",
			tc.Name, tc.ExpectedRule, complexityMatchedHeader, header)
	}
	return match, ""
}

// reportComplexityResults records run details and prints any per-case failures.
func reportComplexityResults(opts pkgtestcases.TestCaseOptions, total, emitted int, failures []string) {
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
	if opts.Verbose {
		fmt.Printf("[Test] Complexity model-mode completed: %d/%d cases emitted a valid signal\n", emitted, total)
	}
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
