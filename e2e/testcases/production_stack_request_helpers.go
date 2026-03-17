package testcases

import (
	"context"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const (
	vsrFastResponseHeader     = "x-vsr-fast-response"
	vsrSelectedDecisionHeader = "x-vsr-selected-decision"

	productionStackJailbreakDecision = "block_jailbreak"
	productionStackPIIDecision       = "block_pii"
)

var productionStackPromptTemplates = []string{
	"In computer science, explain how a hash table handles collisions and give one practical use case. Variation %d.",
	"In biology, summarize how photosynthesis converts light into stored chemical energy. Variation %d.",
	"In business, explain the difference between revenue and profit with a short example. Variation %d.",
	"In history, explain why the printing press accelerated the spread of knowledge. Variation %d.",
}

type productionStackRequestResult struct {
	StatusCode       int
	SelectedDecision string
	FastResponse     bool
	Body             []byte
}

func openProductionStackChatSession(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	timeout time.Duration,
) (*fixtures.ServiceSession, *fixtures.ChatCompletionsClient, error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, nil, err
	}
	return session, fixtures.NewChatCompletionsClient(session, timeout), nil
}

func productionStackPrompt(requestID int) string {
	template := productionStackPromptTemplates[(requestID-1)%len(productionStackPromptTemplates)]
	return fmt.Sprintf(template, requestID)
}

func sendProductionStackChatRequest(
	ctx context.Context,
	chatClient *fixtures.ChatCompletionsClient,
	requestID int,
) (*productionStackRequestResult, error) {
	resp, err := chatClient.Create(ctx, fixtures.ChatCompletionsRequest{
		Model: "MoM",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: productionStackPrompt(requestID)},
		},
	}, nil)
	if err != nil {
		return nil, err
	}

	result := &productionStackRequestResult{
		StatusCode:       resp.StatusCode,
		SelectedDecision: resp.Headers.Get(vsrSelectedDecisionHeader),
		FastResponse:     isProductionStackFastResponse(resp.Headers),
		Body:             resp.Body,
	}
	if result.StatusCode != http.StatusOK {
		return result, fmt.Errorf("status %d: %s", result.StatusCode, truncateString(string(resp.Body), 200))
	}
	return result, nil
}

func isProductionStackFastResponse(headers http.Header) bool {
	if strings.EqualFold(headers.Get(vsrFastResponseHeader), "true") {
		return true
	}

	switch headers.Get(vsrSelectedDecisionHeader) {
	case productionStackJailbreakDecision, productionStackPIIDecision:
		return true
	default:
		return false
	}
}

func recordDecisionCount(counts map[string]int, decision string) {
	if decision == "" {
		return
	}
	counts[decision]++
}

func formatDecisionCounts(counts map[string]int) string {
	if len(counts) == 0 {
		return "none"
	}

	decisions := make([]string, 0, len(counts))
	for decision := range counts {
		decisions = append(decisions, decision)
	}
	sort.Strings(decisions)

	parts := make([]string, 0, len(decisions))
	for _, decision := range decisions {
		parts = append(parts, fmt.Sprintf("%s=%d", decision, counts[decision]))
	}
	return strings.Join(parts, ", ")
}
