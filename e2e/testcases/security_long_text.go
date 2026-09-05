package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("security-long-text", pkgtestcases.TestCase{
		Description: "Verify /api/v1/classify/security detects a jailbreak placed past the classifier's sequence limit",
		Tags:        []string{"kubernetes", "apiserver", "classification", "security", "jailbreak", "api"},
		Fn:          testSecurityLongText,
	})
}

// The jailbreak classifier scores at most its sequence limit of tokens, so one
// classification call only ever saw the start of a long prompt (#3204). This
// drives the endpoint with a jailbreak the profile already blocks on the
// routing path, placed after enough benign text to fall outside that window.
// The filler is scored on its own first: every chunk of it reads as benign to
// the shipped model, so a detection on the combined text can only come from
// the trailing prompt. No mocked backend reproduces this; the truncation lives
// in the tokenizer.
func testSecurityLongText(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	session, err := fixtures.OpenRouterAPISession(ctx, client, opts)
	if err != nil {
		return err
	}
	defer session.Close()

	httpClient := session.HTTPClient(60 * time.Second)
	url := session.URL("/api/v1/classify/security")

	prompt, err := firstBlockedJailbreakPrompt("e2e/testcases/testdata/jailbreak_detection_cases.json")
	if err != nil {
		return err
	}

	sentences := []string{
		"Sedimentary rock forms when layers of sand, mud and organic material settle and compact over long periods. ",
		"Limestone consists mostly of calcium carbonate from the shells of marine organisms. ",
		"Erosion by wind and water gradually reshapes cliffs, valleys and coastlines. ",
		"Fossils preserved in these layers record the sequence of life across geological time. ",
	}
	var builder strings.Builder
	for i := 0; i < 96; i++ {
		builder.WriteString(sentences[i%len(sentences)])
	}
	filler := builder.String()

	control, err := classifySecurity(ctx, httpClient, url, filler)
	if err != nil {
		return err
	}
	result, err := classifySecurity(ctx, httpClient, url, filler+prompt)
	if err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"filler_runes":       utf8.RuneCountInString(filler),
			"control_jailbreak":  control.IsJailbreak,
			"control_risk_score": control.RiskScore,
			"jailbreak":          result.IsJailbreak,
			"risk_score":         result.RiskScore,
		})
	}
	if opts.Verbose {
		fmt.Printf("[Test] filler_runes=%d control=(jailbreak=%v risk=%.3f) combined=(jailbreak=%v risk=%.3f)\n",
			utf8.RuneCountInString(filler), control.IsJailbreak, control.RiskScore, result.IsJailbreak, result.RiskScore)
	}

	if control.IsJailbreak {
		return fmt.Errorf(
			"the benign filler alone was flagged (risk_score=%.3f), so this case cannot tell a detection past the window from a false positive",
			control.RiskScore)
	}
	if !result.IsJailbreak {
		return fmt.Errorf(
			"expected the jailbreak placed after %d runes of benign text to be detected, got is_jailbreak=false risk_score=%.3f: the endpoint scored only the start of the text",
			utf8.RuneCountInString(filler), result.RiskScore)
	}
	return nil
}

// firstBlockedJailbreakPrompt returns the first fixture prompt the
// jailbreak-detection case expects the profile to block, so the long-text case
// reuses a prompt this deployment is already known to catch.
func firstBlockedJailbreakPrompt(path string) (string, error) {
	cases, err := loadJailbreakCases(path)
	if err != nil {
		return "", err
	}
	for _, testCase := range cases {
		if testCase.ExpectedBlocked {
			return testCase.Question, nil
		}
	}
	return "", fmt.Errorf("the jailbreak fixture %s has no expected_blocked case to place past the window", path)
}

type securityClassifyResponse struct {
	IsJailbreak bool    `json:"is_jailbreak"`
	RiskScore   float64 `json:"risk_score"`
	Confidence  float64 `json:"confidence"`
}

func classifySecurity(
	ctx context.Context,
	httpClient *http.Client,
	url string,
	text string,
) (securityClassifyResponse, error) {
	var result securityClassifyResponse
	body, err := json.Marshal(map[string]interface{}{"text": text})
	if err != nil {
		return result, fmt.Errorf("marshal /api/v1/classify/security payload: %w", err)
	}

	resp, err := postJSON(ctx, httpClient, http.MethodPost, url, body)
	if err != nil {
		return result, err
	}
	if resp.StatusCode != http.StatusOK {
		return result, fmt.Errorf("expected /api/v1/classify/security status 200, got %d: %s", resp.StatusCode, string(resp.Body))
	}
	if err := json.Unmarshal(resp.Body, &result); err != nil {
		return result, fmt.Errorf("decode /api/v1/classify/security response: %w", err)
	}
	return result, nil
}
