package testcases

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/e2e/pkg/fixtures"
	pkgtestcases "github.com/vllm-project/semantic-router/e2e/pkg/testcases"
	"k8s.io/client-go/kubernetes"
)

const workflowRedisKeyPrefix = "vllm-sr:flow:state:"

func init() {
	pkgtestcases.Register("workflow-resume-restart-recovery", pkgtestcases.TestCase{
		Description: "Workflow tool pause/resume state persists in Redis across Semantic Router pod restarts",
		Tags:        []string{"workflow", "functional", "redis", "restart"},
		Fn:          testWorkflowResumeRestartRecovery,
	})
}

type workflowToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type workflowToolCallItem struct {
	ID       string                   `json:"id"`
	Type     string                   `json:"type"`
	Function workflowToolCallFunction `json:"function"`
}

type workflowMessagePayload struct {
	Role      string                 `json:"role"`
	Content   string                 `json:"content"`
	ToolCalls []workflowToolCallItem `json:"tool_calls"`
}

type workflowChoicePayload struct {
	Index        int                    `json:"index"`
	FinishReason string                 `json:"finish_reason"`
	Message      workflowMessagePayload `json:"message"`
}

type workflowChatResponsePayload struct {
	ID      string                  `json:"id"`
	Choices []workflowChoicePayload `json:"choices"`
}

func testWorkflowResumeRestartRecovery(ctx context.Context, client *kubernetes.Clientset, opts pkgtestcases.TestCaseOptions) error {
	if opts.Verbose {
		fmt.Println("[Test] Testing Workflow Resume: tool pause/resume restart recovery (Redis persistence)")
	}

	// 1. Send initial request with tools guaranteed to trigger the workflows algorithm and tool pause
	stateID, toolCallID, toolName, toolArgs, err := triggerWorkflowToolPauseBeforeRestart(ctx, client, opts)
	if err != nil {
		return err
	}

	if opts.Verbose {
		fmt.Printf("[Test] Triggered workflow pause (stateID=%s, toolCallID=%s). Restarting semantic-router pod...\n", stateID, toolCallID)
	}

	// 2. Kill the router pod
	if err := deleteSemanticRouterPod(ctx, client, opts); err != nil {
		return fmt.Errorf("delete router pod: %w", err)
	}

	// 3. Wait for the new pod to spin up and be ready
	if err := waitForSemanticRouterReady(ctx, client, opts); err != nil {
		return fmt.Errorf("wait for router ready: %w", err)
	}

	if opts.Verbose {
		fmt.Println("[Test] Pod restarted. Validating workflow state in Redis and resuming workflow...")
	}

	// 4. Verify Redis key exists under workflow-state prefix and resume workflow
	return resumeAndVerifyWorkflowStateAfterRestart(ctx, client, opts, stateID, toolCallID, toolName, toolArgs)
}

func triggerWorkflowToolPauseBeforeRestart(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
) (stateID string, toolCallID string, toolName string, toolArgs string, err error) {
	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return "", "", "", "", fmt.Errorf("open session for pre-restart workflow: %w", err)
	}
	defer session.Close()

	chatClient := fixtures.NewChatCompletionsClient(session, 45*time.Second)

	initReq := fixtures.ChatCompletionsRequest{
		Model: "openai/gpt-oss-20b",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Run workflow task: calculate sum of 2 and 2"},
		},
		Tools: []fixtures.ChatTool{
			{
				Type: "function",
				Function: fixtures.ChatToolFunc{
					Name:        "calculate",
					Description: "Calculate mathematical expressions",
					Parameters:  json.RawMessage(`{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}`),
				},
			},
		},
		ToolChoice: json.RawMessage(`"auto"`),
	}

	httpResp, err := chatClient.Create(ctx, initReq, map[string]string{"x-vsr-debug": "true"})
	if err != nil {
		return "", "", "", "", fmt.Errorf("initial workflow request failed: %w", err)
	}
	if httpResp.StatusCode != http.StatusOK {
		return "", "", "", "", fmt.Errorf("initial workflow request HTTP %d: %s", httpResp.StatusCode, string(httpResp.Body))
	}

	var chatResp workflowChatResponsePayload
	if err := httpResp.DecodeJSON(&chatResp); err != nil {
		return "", "", "", "", fmt.Errorf("decode initial workflow response: %w", err)
	}

	if len(chatResp.Choices) == 0 || len(chatResp.Choices[0].Message.ToolCalls) == 0 {
		return "", "", "", "", fmt.Errorf("expected workflow tool interruption but got response: %s", string(httpResp.Body))
	}

	toolCall := chatResp.Choices[0].Message.ToolCalls[0]
	toolCallID = toolCall.ID
	toolName = toolCall.Function.Name
	toolArgs = toolCall.Function.Arguments

	if !strings.HasPrefix(toolCallID, "flowtool_") {
		return "", "", "", "", fmt.Errorf("expected tool_call id prefix 'flowtool_', got %q", toolCallID)
	}

	trimmed := strings.TrimPrefix(toolCallID, "flowtool_")
	idx := strings.Index(trimmed, "__")
	if idx <= 0 {
		return "", "", "", "", fmt.Errorf("invalid workflow tool call id format %q", toolCallID)
	}
	stateID = trimmed[:idx]

	if err := assertWorkflowStateInRedis(ctx, client, stateID, opts); err != nil {
		return "", "", "", "", fmt.Errorf("assert workflow state before restart: %w", err)
	}

	return stateID, toolCallID, toolName, toolArgs, nil
}

func resumeAndVerifyWorkflowStateAfterRestart(
	ctx context.Context,
	client *kubernetes.Clientset,
	opts pkgtestcases.TestCaseOptions,
	stateID string,
	toolCallID string,
	toolName string,
	toolArgs string,
) error {
	if err := assertWorkflowStateInRedis(ctx, client, stateID, opts); err != nil {
		return fmt.Errorf("workflow state not found in redis after restart: %w", err)
	}

	session, err := fixtures.OpenServiceSession(ctx, client, opts)
	if err != nil {
		return fmt.Errorf("open session for workflow resume: %w", err)
	}
	defer session.Close()

	chatClient := fixtures.NewChatCompletionsClient(session, 45*time.Second)

	resumeReq := fixtures.ChatCompletionsRequest{
		Model: "openai/gpt-oss-20b",
		Messages: []fixtures.ChatMessage{
			{Role: "user", Content: "Run workflow task: calculate sum of 2 and 2"},
			{
				Role: "assistant",
				ToolCalls: []fixtures.ChatToolCall{
					{
						ID:   toolCallID,
						Type: "function",
						Function: fixtures.ChatToolCallFunc{
							Name:      toolName,
							Arguments: toolArgs,
						},
					},
				},
			},
			{
				Role:       "tool",
				ToolCallID: toolCallID,
				Content:    `{"result": 4}`,
			},
		},
	}

	httpResp, err := chatClient.Create(ctx, resumeReq, map[string]string{"x-vsr-debug": "true"})
	if err != nil {
		return fmt.Errorf("workflow resume request failed: %w", err)
	}
	if httpResp.StatusCode != http.StatusOK {
		return fmt.Errorf("workflow resume request HTTP %d: %s", httpResp.StatusCode, string(httpResp.Body))
	}

	var chatResp workflowChatResponsePayload
	if err := httpResp.DecodeJSON(&chatResp); err != nil {
		return fmt.Errorf("decode workflow resume response: %w", err)
	}

	if len(chatResp.Choices) == 0 || chatResp.Choices[0].Message.Content == "" {
		return fmt.Errorf("workflow resume returned empty content: %s", string(httpResp.Body))
	}

	if err := assertWorkflowStateConsumedInRedis(ctx, client, stateID, opts); err != nil {
		return fmt.Errorf("workflow state consumption verification failed: %w", err)
	}

	if opts.Verbose {
		fmt.Printf("[Test] Workflow successfully resumed, completed, and state %s consumed in Redis\n", stateID)
	}
	if opts.SetDetails != nil {
		opts.SetDetails(map[string]interface{}{
			"state_id": stateID,
			"resumed":  true,
			"consumed": true,
			"survived": true,
		})
	}
	return nil
}

func assertWorkflowStateInRedis(ctx context.Context, client *kubernetes.Clientset, stateID string, opts pkgtestcases.TestCaseOptions) error {
	podName, useCluster, found, err := getRedisPod(ctx, client)
	if err != nil {
		return fmt.Errorf("lookup redis pod: %w", err)
	}
	if !found {
		return fmt.Errorf("no redis pod found")
	}

	key := workflowRedisKeyPrefix + stateID
	output, err := execRedisCli(ctx, podName, useCluster, opts.Verbose, "EXISTS", key)
	if err != nil {
		return fmt.Errorf("redis EXISTS %s failed: %w", key, err)
	}

	trimmed := strings.TrimSpace(output)
	if trimmed != "1" && !strings.Contains(trimmed, "1") {
		return fmt.Errorf("expected redis key %s to exist, got output %q", key, output)
	}
	return nil
}

func assertWorkflowStateConsumedInRedis(ctx context.Context, client *kubernetes.Clientset, stateID string, opts pkgtestcases.TestCaseOptions) error {
	podName, useCluster, found, err := getRedisPod(ctx, client)
	if err != nil {
		return fmt.Errorf("lookup redis pod: %w", err)
	}
	if !found {
		return fmt.Errorf("no redis pod found")
	}

	key := workflowRedisKeyPrefix + stateID
	output, err := execRedisCli(ctx, podName, useCluster, opts.Verbose, "EXISTS", key)
	if err != nil {
		return fmt.Errorf("redis EXISTS %s failed: %w", key, err)
	}

	trimmed := strings.TrimSpace(output)
	if trimmed != "0" && !strings.Contains(trimmed, "0") {
		return fmt.Errorf("expected redis key %s to be consumed, got output %q", key, output)
	}
	return nil
}
