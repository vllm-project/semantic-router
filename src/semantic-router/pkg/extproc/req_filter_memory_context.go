package extproc

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FormatMemoriesAsContext formats retrieved memories as a context block
// for injection into the LLM request.
func FormatMemoriesAsContext(memories []*memory.RetrieveResult) string {
	if len(memories) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("The following is relevant context from previous conversations with the user:\n\n")

	for _, result := range memories {
		if result.Memory != nil && result.Memory.Content != "" {
			sb.WriteString(fmt.Sprintf("- %s\n", result.Memory.Content))
		}
	}

	sb.WriteString("\nUse this context to personalize your response when relevant. Do not repeat it verbatim unless asked.")

	return sb.String()
}

// injectMemoryMessages inserts memory context as a separate message in the
// conversation, following the openai-agents-python pattern where context is
// injected as conversation items rather than appended to the system prompt.
func injectMemoryMessages(requestBody []byte, content string) ([]byte, error) {
	var request map[string]interface{}
	if err := json.Unmarshal(requestBody, &request); err != nil {
		return nil, fmt.Errorf("failed to parse request body: %w", err)
	}

	messages, ok := request["messages"].([]interface{})
	if !ok {
		messages = []interface{}{}
	}

	memoryMessage := map[string]interface{}{
		"role":    "user",
		"content": content,
	}

	insertIdx := 0
	for i, msg := range messages {
		msgMap, ok := msg.(map[string]interface{})
		if !ok {
			continue
		}
		if role, ok := msgMap["role"].(string); ok && (role == "system" || role == "developer") {
			insertIdx = i + 1
		}
	}

	newMessages := make([]interface{}, 0, len(messages)+1)
	newMessages = append(newMessages, messages[:insertIdx]...)
	newMessages = append(newMessages, memoryMessage)
	newMessages = append(newMessages, messages[insertIdx:]...)

	request["messages"] = newMessages

	modifiedBody, err := json.Marshal(request)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal modified request: %w", err)
	}

	logging.Debugf("Memory: Injected memory as separate message at position %d", insertIdx)
	return modifiedBody, nil
}
