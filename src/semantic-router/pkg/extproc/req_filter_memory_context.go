package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/memory"
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
