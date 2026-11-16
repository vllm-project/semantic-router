package codexreview

import (
	"bytes"
	"encoding/json"
	"path"
	"slices"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay/store"
)

var (
	riskLevels         = []string{"low", "medium", "high", "critical"}
	userAuthorizations = []string{"unknown", "low", "medium", "high"}
)

// Verdict reads the reviewer's final answer. An answer that asks for a tool
// check is not a verdict yet; an error response is a failed review, never a
// deny; any other answer that is not the verdict schema is invalid. Only
// outcome is required, as in Codex's schema.
func Verdict(response *llmprotocol.Response) (string, *store.ReviewAssessment) {
	if response == nil {
		return store.ReviewStatusInvalid, nil
	}
	if response.Error != nil {
		return store.ReviewStatusFailed, nil
	}
	text, toolCall := finalAnswer(response.Output)
	if toolCall {
		return store.ReviewStatusToolCall, nil
	}
	var answer struct {
		Outcome           string `json:"outcome"`
		RiskLevel         string `json:"risk_level"`
		UserAuthorization string `json:"user_authorization"`
		Rationale         string `json:"rationale"`
	}
	if json.Unmarshal([]byte(strings.TrimSpace(text)), &answer) != nil {
		return store.ReviewStatusInvalid, nil
	}
	if answer.Outcome != store.ReviewOutcomeAllow && answer.Outcome != store.ReviewOutcomeDeny ||
		!optionalOneOf(answer.RiskLevel, riskLevels) || !optionalOneOf(answer.UserAuthorization, userAuthorizations) {
		return store.ReviewStatusInvalid, nil
	}
	// The rationale stays whole here: whoever keeps it redacts it first and
	// bounds it after, with BoundRationale, so no cut hides a secret.
	return store.ReviewStatusCompleted, &store.ReviewAssessment{
		Outcome: answer.Outcome, RiskLevel: answer.RiskLevel, UserAuthorization: answer.UserAuthorization,
		Rationale: strings.TrimSpace(answer.Rationale),
	}
}

// BoundRationale cuts a redacted rationale to what a record keeps.
func BoundRationale(rationale string) (string, bool) {
	return truncate(rationale, maxRationaleBytes)
}

func optionalOneOf(value string, allowed []string) bool {
	return value == "" || slices.Contains(allowed, value)
}

// finalAnswer returns the text of the last output item that has any, and
// whether the reviewer called a tool instead of answering.
func finalAnswer(output []llmprotocol.OutputItem) (string, bool) {
	text := ""
	for _, item := range output {
		for _, content := range item.Content {
			if content.Kind == llmprotocol.ContentToolCall {
				return "", true
			}
		}
		if itemText := contentText(item.Content); itemText != "" {
			text = itemText
		}
	}
	return text, false
}

// plannedAction reads the planned action JSON in an approval request block.
// The summary is a short, single-line view of what would run; the arguments
// keep the action as Codex sent it, compacted and bounded.
func plannedAction(block string) *store.ReviewAction {
	_, raw, ok := strings.Cut(block, plannedActionLabel)
	raw = strings.TrimSpace(raw)
	var fields map[string]any
	if !ok || json.Unmarshal([]byte(raw), &fields) != nil {
		return nil
	}
	tool, _ := fields["tool"].(string)
	cwd, _ := fields["cwd"].(string)
	action := &store.ReviewAction{Tool: boundedID(tool), Summary: actionSummary(fields)}
	action.Cwd, _ = truncate(cwd, maxCwdBytes)
	var compact bytes.Buffer
	if json.Compact(&compact, []byte(raw)) == nil {
		action.Arguments, action.ArgumentsTruncated = truncate(compact.String(), maxArgumentBytes)
	}
	return action
}

// actionSummary names what the action does: a shell script or command, the
// files a patch touches, a network target, or an MCP tool.
func actionSummary(fields map[string]any) string {
	summary := ""
	switch command := fields["command"].(type) {
	case string:
		summary = command
	case []any:
		summary = commandSummary(stringItems(command))
	}
	if summary == "" {
		summary = strings.Join(stringItems(asSlice(fields["files"])), ", ")
	}
	if summary == "" {
		summary = firstString(fields, "target", "host", "url")
	}
	if server, tool := firstString(fields, "server"), firstString(fields, "tool_name"); summary == "" && tool != "" {
		summary = strings.TrimPrefix(server+"."+tool, ".")
	}
	summary = strings.Join(strings.Fields(summary), " ")
	if utf8.RuneCountInString(summary) > maxSummaryRunes {
		summary = string([]rune(summary)[:maxSummaryRunes-1]) + "…"
	}
	return summary
}

// commandSummary shows a shell's script, as in bash -lc "script", and any
// other command as its words.
func commandSummary(words []string) string {
	if len(words) == 3 && strings.HasPrefix(words[1], "-") && strings.HasSuffix(words[1], "c") &&
		strings.HasSuffix(path.Base(words[0]), "sh") {
		return words[2]
	}
	return strings.Join(words, " ")
}

func asSlice(value any) []any {
	items, _ := value.([]any)
	return items
}

func stringItems(values []any) []string {
	items := make([]string, 0, len(values))
	for _, value := range values {
		if text, ok := value.(string); ok {
			items = append(items, text)
		}
	}
	return items
}

func firstString(fields map[string]any, keys ...string) string {
	for _, key := range keys {
		if value, ok := fields[key].(string); ok && strings.TrimSpace(value) != "" {
			return value
		}
	}
	return ""
}
