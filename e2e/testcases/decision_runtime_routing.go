package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"k8s.io/client-go/kubernetes"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
)

func init() {
	pkgtestcases.Register("decision-runtime-routing", pkgtestcases.TestCase{
		Description: "Verify decision signals and the decision selector follow the model runtime's own answers and fail open",
		Tags:        []string{"decision", "model-runtime", "routing", "selection"},
		Fn:          testDecisionRuntimeRouting,
	})
}

// decisionRuntimeQuestion mirrors the runtime request the Router builds from
// e2e/profiles/decision-runtime/values.yaml.
type decisionRuntimeQuestion struct {
	Type         string                  `json:"type"`
	Instructions string                  `json:"instructions"`
	Choices      []decisionRuntimeChoice `json:"choices,omitempty"`
}

type decisionRuntimeChoice struct {
	Key         string `json:"key"`
	Description string `json:"description,omitempty"`
}

type decisionRuntimeAnswer struct {
	Choice        string             `json:"choice"`
	Noul          *float64           `json:"noul"`
	Probabilities map[string]float64 `json:"probabilities"`
	Error         string             `json:"error"`
}

var decisionRuntimeSignalQuestions = map[string]decisionRuntimeQuestion{
	"answered": {Type: "noul", Instructions: "Is this request about software?"},
	"request_kind": {
		Type:         "choice",
		Instructions: "What kind of work does this request ask for?",
		Choices: []decisionRuntimeChoice{
			{Key: "code", Description: "Writing, reviewing or debugging code"},
			{Key: "math", Description: "Mathematics or quantitative reasoning"},
			{Key: "chat", Description: "Anything else"},
		},
	},
}

var decisionRuntimeSelectorQuestion = map[string]decisionRuntimeQuestion{
	"selector": {
		Type:         "choice",
		Instructions: "Which model should answer this request?",
		Choices: []decisionRuntimeChoice{
			{Key: "general-expert", Description: "General assistant for everyday requests"},
			{Key: "math-expert", Description: "Specialist for mathematics and quantitative reasoning"},
		},
	},
}

var decisionRuntimePrompts = []string{
	"Write a Go function that merges two sorted slices.",
	"What is the integral of x squared from zero to three?",
	"Suggest a name for a friendly golden retriever.",
}

// testDecisionRuntimeRouting compares the Router with the runtime itself. The
// fixture has random weights, so the expected answers are whatever the
// runtime returns for the identical request; the Router must route on exactly
// those answers. The offline deployment never answers, so its route must
// never match while every request still succeeds.
func testDecisionRuntimeRouting(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	localPort, stop, err := setupServiceConnection(ctx, client, opts)
	if err != nil {
		return err
	}
	defer stop()
	runtime, err := fixtures.OpenServiceEndpointSession(ctx, client, opts, "default", "decision-runtime", "8100")
	if err != nil {
		return err
	}
	defer runtime.Close()

	if err := waitForDecisionRuntimeRoute(ctx, localPort); err != nil {
		return err
	}
	details := map[string]interface{}{}
	for _, prompt := range decisionRuntimePrompts {
		observed, err := checkDecisionRuntimePrompt(ctx, localPort, runtime.BaseURL(), prompt)
		if err != nil {
			return err
		}
		details[prompt] = observed
		if opts.Verbose {
			fmt.Printf("[Test] %q -> %s\n", prompt, observed)
		}
	}
	if opts.SetDetails != nil {
		details["runtime"] = "tiny random-weight Decision 2.0 fixture on CPU, attached over HTTP"
		opts.SetDetails(details)
	}
	return nil
}

// waitForDecisionRuntimeRoute waits until the Router reports the attached
// runtime ready. Until then decision signals are unknown by design, so the
// request falls through to default-route.
func waitForDecisionRuntimeRoute(ctx context.Context, localPort string) error {
	deadline := time.Now().Add(2 * time.Minute)
	last := ""
	for time.Now().Before(deadline) {
		resp, err := sendLocalChatCompletion(ctx, localPort, "auto", decisionRuntimePrompts[0], 30*time.Second)
		if err == nil && resp.StatusCode == http.StatusOK {
			last = resp.Headers.Get("x-vsr-selected-decision")
			switch last {
			case "selector_route":
				return nil
			case "offline_route":
				return fmt.Errorf("offline_route matched: an unanswered decision signal must not match")
			}
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-time.After(2 * time.Second):
		}
	}
	return fmt.Errorf("the Router never routed on the runtime's answers; last decision %q", last)
}

func checkDecisionRuntimePrompt(ctx context.Context, localPort, runtimeURL, prompt string) (string, error) {
	resp, err := sendLocalChatCompletion(ctx, localPort, "auto", prompt, 30*time.Second)
	if err != nil {
		return "", fmt.Errorf("chat request %q: %w", prompt, err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("chat request %q: %s", prompt, formatUnexpectedChatCompletionStatus(resp))
	}
	decision := resp.Headers.Get("x-vsr-selected-decision")
	if decision != "selector_route" {
		return "", fmt.Errorf("%q selected decision %q, want selector_route", prompt, decision)
	}

	signals, err := askDecisionRuntime(ctx, runtimeURL, prompt, decisionRuntimeSignalQuestions)
	if err != nil {
		return "", err
	}
	kind := signals["request_kind"].Choice
	matched := strings.Split(resp.Headers.Get("x-vsr-matched-decision-model"), ",")
	for _, want := range []string{"answered", "request_kind:" + kind} {
		if !containsHeaderItem(matched, want) {
			return "", fmt.Errorf("%q: x-vsr-matched-decision-model %v lacks %q, the runtime's own answer", prompt, matched, want)
		}
	}
	if containsHeaderItem(matched, "offline") {
		return "", fmt.Errorf("%q: the offline deployment's signal matched", prompt)
	}

	selector, err := askDecisionRuntime(ctx, runtimeURL, prompt, decisionRuntimeSelectorQuestion)
	if err != nil {
		return "", err
	}
	want := selector["selector"].Choice
	if model := resp.Headers.Get("x-vsr-selected-model"); model != want {
		return "", fmt.Errorf("%q: Router selected model %q, the runtime chose %q (probabilities %v)",
			prompt, model, want, selector["selector"].Probabilities)
	}
	return fmt.Sprintf("request_kind=%s model=%s", kind, want), nil
}

func askDecisionRuntime(
	ctx context.Context,
	runtimeURL string,
	state string,
	questions map[string]decisionRuntimeQuestion,
) (map[string]decisionRuntimeAnswer, error) {
	body, err := json.Marshal(map[string]interface{}{"state": state, "questions": questions})
	if err != nil {
		return nil, err
	}
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, runtimeURL+"/v1/decisions", bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	request.Header.Set("Content-Type", "application/json")
	response, err := (&http.Client{Timeout: 30 * time.Second}).Do(request)
	if err != nil {
		return nil, fmt.Errorf("runtime request: %w", err)
	}
	defer response.Body.Close()
	payload, err := io.ReadAll(response.Body)
	if err != nil {
		return nil, err
	}
	if response.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("runtime returned %d: %s", response.StatusCode, payload)
	}
	var decoded struct {
		Answers map[string]decisionRuntimeAnswer `json:"answers"`
	}
	if err := json.Unmarshal(payload, &decoded); err != nil {
		return nil, fmt.Errorf("runtime response: %w", err)
	}
	for id := range questions {
		answer, ok := decoded.Answers[id]
		if !ok || answer.Error != "" {
			return nil, fmt.Errorf("runtime answer %q missing or failed: %+v", id, answer)
		}
	}
	return decoded.Answers, nil
}

func containsHeaderItem(values []string, want string) bool {
	for _, value := range values {
		if strings.TrimSpace(value) == want {
			return true
		}
	}
	return false
}
