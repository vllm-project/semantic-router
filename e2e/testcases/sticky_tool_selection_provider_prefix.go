package testcases

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

func init() {
	pkgtestcases.Register("sticky-tool-selection-provider-prefix", pkgtestcases.TestCase{
		Description: "Verify sticky tool growth preserves the Anthropic provider prefix and cache-usage contract",
		Tags:        []string{"anthropic", "cache", "plugin", "sticky", "tool-selection"},
		Fn:          testStickyToolSelectionProviderPrefix,
	})
}

type stickyPrefixRequest struct {
	Model     string                `json:"model"`
	MaxTokens int                   `json:"max_tokens"`
	Metadata  map[string]string     `json:"metadata"`
	Messages  []anthropicMessage    `json:"messages"`
	Tools     []stickyAnthropicTool `json:"tools"`
}

type stickyAnthropicTool struct {
	Name         string                 `json:"name"`
	Description  string                 `json:"description"`
	InputSchema  json.RawMessage        `json:"input_schema"`
	CacheControl map[string]interface{} `json:"cache_control,omitempty"`
}

type stickyPrefixCycle struct {
	FirstUsage  anthropicCacheUsage
	SecondUsage anthropicCacheUsage
	FirstTools  stickyToolSnapshot
	SecondTools stickyToolSnapshot
}

const (
	stickyProviderPrefixFirstPrompt  = "What is the weather forecast?"
	stickyProviderPrefixGrowthPrompt = " "
)

func testStickyToolSelectionProviderPrefix(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) error {
	if opts.Verbose {
		fmt.Println("[Anthropic] Testing sticky tool provider-prefix retention")
	}

	sessions, err := openStickyAnthropicSessionPair(ctx, client, opts)
	if err != nil {
		return err
	}
	defer sessions.gateway.Close()
	defer sessions.backend.Close()

	trusted, err := runStickyProviderPrefixCycle(ctx, sessions, true)
	if err != nil {
		return fmt.Errorf("trusted sticky cycle: %w", err)
	}
	stateless, err := runStickyProviderPrefixCycle(ctx, sessions, false)
	if err != nil {
		return fmt.Errorf("stateless baseline cycle: %w", err)
	}
	if err := assertStickyProviderPrefixCycle(trusted, stateless); err != nil {
		return err
	}

	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"trusted_first_tools":             trusted.FirstTools.Names,
			"trusted_second_tools":            trusted.SecondTools.Names,
			"trusted_cache_creation_tokens":   trusted.FirstUsage.CacheCreationInputTokens,
			"trusted_cache_read_tokens":       trusted.SecondUsage.CacheReadInputTokens,
			"stateless_second_tools":          stateless.SecondTools.Names,
			"stateless_cache_creation_tokens": stateless.SecondUsage.CacheCreationInputTokens,
			"stateless_cache_read_tokens":     stateless.SecondUsage.CacheReadInputTokens,
		})
	}
	return nil
}

func openStickyAnthropicSessionPair(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (*stickySessionPair, error) {
	gateway, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return nil, err
	}
	backendOpts := opts
	backendOpts.ServiceConfig = pkgtestcases.ServiceConfig{
		Namespace:   "anthropic-backend-system",
		Name:        "anthropic-backend-qwen",
		ServicePort: "8080",
	}
	backend, err := fixtures.OpenServiceSession(ctx, client, backendOpts)
	if err != nil {
		gateway.Close()
		return nil, err
	}
	return &stickySessionPair{gateway: gateway, backend: backend}, nil
}

func runStickyProviderPrefixCycle(
	ctx context.Context,
	sessions *stickySessionPair,
	trusted bool,
) (stickyPrefixCycle, error) {
	sessionID := fmt.Sprintf("sticky-prefix-%t-%d", trusted, time.Now().UnixNano())
	headers := stickyRequestHeaders(sessionID, trusted)

	firstUsage, firstTools, err := runStickyProviderPrefixTurn(
		ctx,
		sessions,
		sessionID,
		headers,
		stickyProviderPrefixFirstPrompt,
		stickyProviderPrefixTools(),
	)
	if err != nil {
		return stickyPrefixCycle{}, err
	}
	secondUsage, secondTools, err := runStickyProviderPrefixTurn(
		ctx,
		sessions,
		sessionID,
		headers,
		stickyProviderPrefixGrowthPrompt,
		stickyProviderPrefixTools(),
	)
	if err != nil {
		return stickyPrefixCycle{}, err
	}
	return stickyPrefixCycle{
		FirstUsage:  firstUsage,
		SecondUsage: secondUsage,
		FirstTools:  firstTools,
		SecondTools: secondTools,
	}, nil
}

func runStickyProviderPrefixTurn(
	ctx context.Context,
	sessions *stickySessionPair,
	sessionID string,
	headers map[string]string,
	prompt string,
	tools []stickyAnthropicTool,
) (anthropicCacheUsage, stickyToolSnapshot, error) {
	body, err := sendProtocolMatrixRequestWithHeaders(
		ctx,
		sessions.gateway,
		"/v1/messages",
		stickyPrefixRequest{
			Model:     "MoM",
			MaxTokens: 8,
			Metadata:  map[string]string{"user_id": "sticky-prefix-contract"},
			Messages:  []anthropicMessage{{Role: "user", Content: prompt}},
			Tools:     tools,
		},
		false,
		headers,
	)
	if err != nil {
		return anthropicCacheUsage{}, stickyToolSnapshot{}, err
	}
	var response anthropicCacheResponse
	if err := json.Unmarshal(body, &response); err != nil {
		return anthropicCacheUsage{}, stickyToolSnapshot{}, fmt.Errorf("decode response: %w", err)
	}
	providerBody, err := lastProviderSimulatorRequest(ctx, sessions.backend, sessionID)
	if err != nil {
		return anthropicCacheUsage{}, stickyToolSnapshot{}, err
	}
	snapshot, err := decodeStickyProviderTools(providerBody)
	if err != nil {
		return anthropicCacheUsage{}, stickyToolSnapshot{}, err
	}
	return response.Usage, snapshot, nil
}

func assertStickyProviderPrefixCycle(trusted, stateless stickyPrefixCycle) error {
	if err := assertStickyTrustedProviderPrefix(trusted); err != nil {
		return err
	}
	return assertStickyStatelessProviderPrefix(stateless)
}

func assertStickyTrustedProviderPrefix(cycle stickyPrefixCycle) error {
	if len(cycle.FirstTools.Tools) != 1 || cycle.FirstTools.Names[0] != "get_weather" {
		return fmt.Errorf("trusted first provider tools = %v, want [get_weather]", cycle.FirstTools.Names)
	}
	if len(cycle.SecondTools.Tools) != 2 ||
		cycle.SecondTools.Names[0] != "get_weather" || cycle.SecondTools.Names[1] != "calculate" {
		return fmt.Errorf("trusted growth reordered the provider prefix: %v", cycle.SecondTools.Names)
	}
	if !bytes.Equal(cycle.FirstTools.Tools[0], cycle.SecondTools.Tools[0]) {
		return fmt.Errorf("retained provider definition changed bytes for %q", "get_weather")
	}
	if cycle.FirstUsage.CacheCreationInputTokens <= 0 || cycle.FirstUsage.CacheReadInputTokens != 0 {
		return fmt.Errorf("trusted first turn did not create a provider prefix: %+v", cycle.FirstUsage)
	}
	if cycle.SecondUsage.CacheReadInputTokens <= 0 || cycle.SecondUsage.CacheCreationInputTokens != 0 {
		return fmt.Errorf("trusted growth turn did not reuse the provider prefix: %+v", cycle.SecondUsage)
	}
	return nil
}

func assertStickyStatelessProviderPrefix(cycle stickyPrefixCycle) error {
	if len(cycle.FirstTools.Tools) != 1 || cycle.FirstTools.Names[0] != "get_weather" {
		return fmt.Errorf("stateless first provider tools = %v, want [get_weather]", cycle.FirstTools.Names)
	}
	if len(cycle.SecondTools.Tools) != 2 ||
		cycle.SecondTools.Names[0] != "calculate" || cycle.SecondTools.Names[1] != "get_weather" {
		return fmt.Errorf("stateless provider tools = %v, want [calculate get_weather]", cycle.SecondTools.Names)
	}
	if cycle.FirstUsage.CacheCreationInputTokens <= 0 || cycle.FirstUsage.CacheReadInputTokens != 0 {
		return fmt.Errorf("stateless first turn did not create its provider prefix: %+v", cycle.FirstUsage)
	}
	if cycle.SecondUsage.CacheCreationInputTokens <= 0 || cycle.SecondUsage.CacheReadInputTokens != 0 {
		return fmt.Errorf("stateless baseline unexpectedly reused the old provider prefix: %+v", cycle.SecondUsage)
	}
	return nil
}

func stickyProviderPrefixTools() []stickyAnthropicTool {
	return []stickyAnthropicTool{stickyCalculateTool(), stickyCachedWeatherTool()}
}

func stickyCachedWeatherTool() stickyAnthropicTool {
	return stickyAnthropicTool{
		Name:         "get_weather",
		Description:  "Get current weather information for a location",
		InputSchema:  json.RawMessage(`{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}`),
		CacheControl: map[string]interface{}{"type": "ephemeral"},
	}
}

func stickyCalculateTool() stickyAnthropicTool {
	return stickyAnthropicTool{
		Name:        "calculate",
		Description: "Perform a billing calculation",
		InputSchema: json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
	}
}
