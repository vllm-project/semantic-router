package agentruntime

import (
	"bytes"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

const (
	contextCompactionHighWaterPercent = int64(85)
	contextCompactionTargetPercent    = int64(70)
	maximumContextSynopsisBytes       = 64 << 10
	maximumConstraintMemoryBytes      = 512 << 10
	maximumContextReferences          = 512
	maximumContextFailures            = 128
	maximumContextDecisions           = 512
)

// fitExecutionContext bounds the model-visible projection using the Profile's
// pinned token budget. The compactor is deliberately extractive: user text is
// retained exactly as a constraint, while older assistant prose and completed
// tool exchanges become a bounded synopsis. The durable events remain the
// audit/history source through CompactedThroughSequence.
func fitExecutionContext(
	projection executionContext, tokenBudget int64, tools []llmprotocol.Tool,
) (executionContext, bool, error) {
	if tokenBudget < 1 {
		return executionContext{}, false, agentmanagement.ErrInvalid
	}
	projection.Instructions = rebuildMemoryInstruction(projection.Instructions, projection.Memory)
	estimated, estimateContextTokensErr := estimateContextTokens(projection.Instructions, projection.Messages, tools)
	if estimateContextTokensErr != nil {
		return executionContext{}, false, estimateContextTokensErr
	}
	if estimated*100 <= tokenBudget*contextCompactionHighWaterPercent {
		return projection, false, nil
	}

	latestUser := latestUserMessage(projection.Messages)
	changed := false
	for estimated*100 > tokenBudget*contextCompactionTargetPercent {
		candidates := oldestCompactableMessages(projection, latestUser)
		if len(candidates) == 0 {
			break
		}
		remove := make(map[int]struct{}, len(candidates))
		removedBeforeLatest := 0
		for _, candidate := range candidates {
			if err := absorbCompactedMessage(&projection.Memory, projection.Messages[candidate]); err != nil {
				return executionContext{}, false, err
			}
			remove[candidate] = struct{}{}
			if candidate < latestUser {
				removedBeforeLatest++
			}
		}
		messages := make([]llmprotocol.Message, 0, len(projection.Messages)-len(remove))
		for index, message := range projection.Messages {
			if _, compact := remove[index]; !compact {
				messages = append(messages, message)
			}
		}
		projection.Messages = messages
		latestUser -= removedBeforeLatest
		projection.Memory.CompactedThroughSequence = projection.ThroughSequence
		projection.Instructions = rebuildMemoryInstruction(projection.Instructions, projection.Memory)
		estimated, estimateContextTokensErr = estimateContextTokens(projection.Instructions, projection.Messages, tools)
		if estimateContextTokensErr != nil {
			return executionContext{}, false, estimateContextTokensErr
		}
		changed = true
	}
	if estimated*100 > tokenBudget*contextCompactionTargetPercent && projection.Memory.Synopsis != "" {
		// The durable transcript and CompactedThroughSequence retain the full
		// historical detail. If extractive notes themselves become the largest
		// context item, collapse them to a stable history pointer before ever
		// sacrificing active constraints or unresolved work.
		projection.Memory.Synopsis = "Earlier assistant and completed tool detail remains in durable history."
		projection.Instructions = rebuildMemoryInstruction(projection.Instructions, projection.Memory)
		estimated, estimateContextTokensErr = estimateContextTokens(projection.Instructions, projection.Messages, tools)
		if estimateContextTokensErr != nil {
			return executionContext{}, false, estimateContextTokensErr
		}
		changed = true
	}
	if estimated > tokenBudget {
		return executionContext{}, false, fmt.Errorf(
			"%w: Agent context budget cannot retain the current user input, pending work, and required continuity state",
			agentmanagement.ErrConflict,
		)
	}
	return projection, changed, nil
}

func estimateContextTokens(
	instructions []llmprotocol.InstructionBlock,
	messages []llmprotocol.Message,
	tools []llmprotocol.Tool,
) (int64, error) {
	encoded, err := json.Marshal(struct {
		Instructions []llmprotocol.InstructionBlock `json:"instructions"`
		Messages     []llmprotocol.Message          `json:"messages"`
		Tools        []llmprotocol.Tool             `json:"tools"`
	}{instructions, messages, tools})
	if err != nil {
		return 0, fmt.Errorf("estimate Agent context: %w", err)
	}
	// Three UTF-8 bytes per token is intentionally conservative for Latin
	// text while remaining safe for CJK-heavy sessions (roughly one rune per
	// token). The public inference path remains the final tokenizer authority.
	return int64((len(encoded)+2)/3 + 32), nil
}

func latestUserMessage(messages []llmprotocol.Message) int {
	for index := len(messages) - 1; index >= 0; index-- {
		if messages[index].Role == llmprotocol.RoleUser {
			return index
		}
	}
	return -1
}

func oldestCompactableMessages(projection executionContext, latestUser int) []int {
	for index, message := range projection.Messages {
		if index == latestUser || messageContainsPendingCall(message, projection.Pending) {
			continue
		}
		invocations := messageInvocationIDs(message)
		if len(invocations) == 0 {
			return []int{index}
		}
		indices := make([]int, 0, len(invocations)+1)
		for candidateIndex, candidate := range projection.Messages {
			if candidateIndex == latestUser || messageContainsPendingCall(candidate, projection.Pending) {
				continue
			}
			for invocationID := range messageInvocationIDs(candidate) {
				if _, sameExchange := invocations[invocationID]; sameExchange {
					indices = append(indices, candidateIndex)
					break
				}
			}
		}
		return indices
	}
	return nil
}

func messageInvocationIDs(message llmprotocol.Message) map[string]struct{} {
	result := make(map[string]struct{})
	for _, content := range message.Content {
		if content.ToolCall != nil && content.ToolCall.ID != "" {
			result[content.ToolCall.ID] = struct{}{}
		}
		if content.ToolResult != nil && content.ToolResult.CallID != "" {
			result[content.ToolResult.CallID] = struct{}{}
		}
	}
	return result
}

func messageContainsPendingCall(message llmprotocol.Message, pending []pendingToolCall) bool {
	if len(pending) == 0 {
		return false
	}
	ids := make(map[string]struct{}, len(pending))
	for _, call := range pending {
		ids[call.InvocationID] = struct{}{}
	}
	for _, content := range message.Content {
		if content.ToolCall != nil {
			if _, found := ids[content.ToolCall.ID]; found {
				return true
			}
		}
		if content.ToolResult != nil {
			if _, found := ids[content.ToolResult.CallID]; found {
				return true
			}
		}
	}
	return false
}

func absorbCompactedMessage(memory *executionMemory, message llmprotocol.Message) error {
	switch message.Role {
	case llmprotocol.RoleUser:
		constraint := flattenMessage(message)
		if constraint != "" {
			if err := appendUserConstraint(memory, constraint); err != nil {
				return err
			}
		}
	case llmprotocol.RoleAssistant:
		appendSynopsis(memory, "Assistant: "+boundedMessageNote(message, 1024))
	case llmprotocol.RoleTool:
		appendSynopsis(memory, "Tool result retained in the durable transcript.")
	}
	return nil
}

func flattenMessage(message llmprotocol.Message) string {
	parts := make([]string, 0, len(message.Content))
	for _, content := range message.Content {
		switch content.Kind {
		case llmprotocol.ContentText:
			parts = append(parts, strings.TrimSpace(content.Text))
		case llmprotocol.ContentImage:
			parts = append(parts, "image="+content.URL)
		case llmprotocol.ContentFile:
			parts = append(parts, "file="+content.FileID)
		}
	}
	return strings.TrimSpace(strings.Join(parts, "\n"))
}

func boundedMessageNote(message llmprotocol.Message, maximum int) string {
	value := flattenMessage(message)
	if value == "" {
		for _, content := range message.Content {
			if content.ToolCall != nil {
				value = "requested " + content.ToolCall.Name
				break
			}
		}
	}
	return truncateUTF8(value, maximum)
}

func appendUserConstraint(memory *executionMemory, value string) error {
	value = strings.TrimSpace(value)
	if value == "" {
		return nil
	}
	for _, existing := range memory.UserConstraints {
		if existing == value {
			return nil
		}
	}
	total := len(value)
	for _, existing := range memory.UserConstraints {
		total += len(existing)
	}
	if total > maximumConstraintMemoryBytes {
		return fmt.Errorf(
			"%w: required user constraints exceed the bounded Agent checkpoint",
			agentmanagement.ErrConflict,
		)
	}
	memory.UserConstraints = append(memory.UserConstraints, value)
	return nil
}

func appendSynopsis(memory *executionMemory, value string) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	for _, existing := range strings.Split(memory.Synopsis, "\n") {
		if existing == value {
			return
		}
	}
	if memory.Synopsis != "" {
		memory.Synopsis += "\n"
	}
	memory.Synopsis += value
	if len(memory.Synopsis) > maximumContextSynopsisBytes {
		memory.Synopsis = "Earlier assistant detail remains in durable history.\n" +
			truncateUTF8(utf8Suffix(memory.Synopsis, maximumContextSynopsisBytes), maximumContextSynopsisBytes-64)
	}
}

func executionInstructions(
	stable llmprotocol.InstructionBlock, memory executionMemory,
) []llmprotocol.InstructionBlock {
	return rebuildMemoryInstruction([]llmprotocol.InstructionBlock{stable}, memory)
}

func rebuildMemoryInstruction(
	instructions []llmprotocol.InstructionBlock, memory executionMemory,
) []llmprotocol.InstructionBlock {
	base := append([]llmprotocol.InstructionBlock(nil), instructions...)
	if len(base) > 1 && instructionIsCheckpointMemory(base[len(base)-1]) {
		base = base[:len(base)-1]
	}
	if executionMemoryEmpty(memory) {
		return base
	}
	encoded, _ := json.Marshal(memory)
	base = append(base, llmprotocol.InstructionBlock{
		Role: llmprotocol.RoleSystem,
		Content: []llmprotocol.Content{{
			Kind: llmprotocol.ContentText,
			Text: "Authoritative checkpoint memory. Preserve every unresolved constraint, failure, approval, and pinned resource revision. Historical detail remains auditable through the checkpoint sequence.\n" + string(encoded),
		}},
	})
	return base
}

func instructionIsCheckpointMemory(value llmprotocol.InstructionBlock) bool {
	return len(value.Content) == 1 && value.Content[0].Kind == llmprotocol.ContentText &&
		strings.HasPrefix(value.Content[0].Text, "Authoritative checkpoint memory.")
}

func executionMemoryEmpty(memory executionMemory) bool {
	return memory.Synopsis == "" && len(memory.UserConstraints) == 0 &&
		len(memory.ToolFailures) == 0 && memory.PendingApproval == nil &&
		len(memory.ResourceReferences) == 0 && len(memory.ToolResultReferences) == 0 &&
		len(memory.Decisions) == 0 && memory.CompactedThroughSequence == 0
}

func cloneExecutionMemory(memory executionMemory) executionMemory {
	result := memory
	result.UserConstraints = append([]string(nil), memory.UserConstraints...)
	result.ToolFailures = append([]toolFailureMemory(nil), memory.ToolFailures...)
	result.ResourceReferences = append(
		[]agentmanagement.ResourceReference(nil), memory.ResourceReferences...,
	)
	result.ToolResultReferences = append([]string(nil), memory.ToolResultReferences...)
	result.Decisions = append([]string(nil), memory.Decisions...)
	if memory.PendingApproval != nil {
		copy := *memory.PendingApproval
		copy.Summary.Topology = append(json.RawMessage(nil), copy.Summary.Topology...)
		copy.Summary.Assignments = append(json.RawMessage(nil), copy.Summary.Assignments...)
		copy.Summary.GateResults = append(json.RawMessage(nil), copy.Summary.GateResults...)
		copy.Summary.ChangedResources = append([]string(nil), copy.Summary.ChangedResources...)
		copy.Summary.Warnings = append([]string(nil), copy.Summary.Warnings...)
		result.PendingApproval = &copy
	}
	return result
}

func checkpointSummary(projection executionContext) string {
	if projection.Memory.CompactedThroughSequence == 0 {
		return fmt.Sprintf("Conversation state through event %d.", projection.ThroughSequence)
	}
	return fmt.Sprintf(
		"Compacted context through event %d; durable history remains authoritative through event %d.",
		projection.Memory.CompactedThroughSequence, projection.ThroughSequence,
	)
}

func appendToolFailure(memory *executionMemory, payload agentmanagement.ToolResultEvent) {
	if payload.Error == nil {
		return
	}
	value := toolFailureMemory{
		InvocationID: payload.InvocationID, ToolName: payload.ToolName, Failure: *payload.Error,
	}
	for index := range memory.ToolFailures {
		if memory.ToolFailures[index].InvocationID == value.InvocationID {
			memory.ToolFailures[index] = value
			return
		}
	}
	memory.ToolFailures = append(memory.ToolFailures, value)
	if len(memory.ToolFailures) > maximumContextFailures {
		memory.ToolFailures = memory.ToolFailures[len(memory.ToolFailures)-maximumContextFailures:]
	}
}

func appendToolResultReference(memory *executionMemory, payload agentmanagement.ToolResultEvent) {
	value := payload.ArtifactID
	if value == "" && payload.Status != "completed" {
		value = payload.InvocationID
	}
	appendUniqueBounded(&memory.ToolResultReferences, value, maximumContextReferences)
}

func appendDecision(memory *executionMemory, value string) {
	appendUniqueBounded(&memory.Decisions, value, maximumContextDecisions)
}

func appendUniqueBounded(values *[]string, value string, maximum int) {
	value = strings.TrimSpace(value)
	if value == "" {
		return
	}
	for _, existing := range *values {
		if existing == value {
			return
		}
	}
	*values = append(*values, value)
	if len(*values) > maximum {
		*values = (*values)[len(*values)-maximum:]
	}
}

func observeResourceReferences(memory *executionMemory, raw json.RawMessage, toolName ...string) {
	if len(raw) == 0 {
		return
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var value any
	if decoder.Decode(&value) != nil {
		return
	}
	defaultKind := "resource"
	if len(toolName) > 0 {
		defaultKind = resourceKindForTool(toolName[0])
	}
	walkResourceValue(memory, value, defaultKind, "", 0)
}

func walkResourceValue(
	memory *executionMemory, value any, inheritedKind, inheritedRevision string, depth int,
) {
	if depth > 16 {
		return
	}
	switch typed := value.(type) {
	case []any:
		for _, item := range typed {
			walkResourceValue(memory, item, inheritedKind, inheritedRevision, depth+1)
		}
	case map[string]any:
		kind, revision := inheritedKind, inheritedRevision
		if candidate := firstString(typed, "kind", "resourceKind", "type"); candidate != "" {
			kind = normalizeReferenceKind(candidate, inheritedKind)
		}
		if candidate := firstScalarString(
			typed, "etag", "resourceRevision", "contentRevision", "catalogRevision", "revision", "digest",
		); candidate != "" {
			revision = candidate
		}
		id, idKind := referenceIdentity(typed, kind)
		if id != "" && revision != "" {
			appendResourceReference(memory, agentmanagement.ResourceReference{
				Kind: idKind, ID: id, Revision: revision,
			})
		}
		keys := make([]string, 0, len(typed))
		for key := range typed {
			keys = append(keys, key)
		}
		sort.Strings(keys)
		for _, key := range keys {
			walkResourceValue(memory, typed[key], kind, revision, depth+1)
		}
	}
}

func referenceIdentity(value map[string]any, fallbackKind string) (string, string) {
	for _, candidate := range []struct{ key, kind string }{
		{"recipeId", "recipe"},
		{"entrypointId", "entrypoint"},
		{"modelId", "model"},
		{"planId", "publication_plan"},
		{"resourceId", fallbackKind},
		{"id", fallbackKind},
	} {
		if id := scalarString(value[candidate.key]); id != "" {
			return id, candidate.kind
		}
	}
	return "", fallbackKind
}

func appendResourceReference(memory *executionMemory, value agentmanagement.ResourceReference) {
	if value.Kind == "" || value.ID == "" || value.Revision == "" {
		return
	}
	for index := range memory.ResourceReferences {
		current := &memory.ResourceReferences[index]
		if current.Kind == value.Kind && current.ID == value.ID {
			*current = value
			return
		}
	}
	memory.ResourceReferences = append(memory.ResourceReferences, value)
	sort.Slice(memory.ResourceReferences, func(left, right int) bool {
		if memory.ResourceReferences[left].Kind == memory.ResourceReferences[right].Kind {
			return memory.ResourceReferences[left].ID < memory.ResourceReferences[right].ID
		}
		return memory.ResourceReferences[left].Kind < memory.ResourceReferences[right].Kind
	})
	if len(memory.ResourceReferences) > maximumContextReferences {
		memory.ResourceReferences = memory.ResourceReferences[len(memory.ResourceReferences)-maximumContextReferences:]
	}
}

func resourceKindForTool(name string) string {
	switch {
	case strings.Contains(name, "entrypoint"):
		return "entrypoint"
	case strings.Contains(name, "recipe"):
		return "recipe"
	case strings.Contains(name, "model"):
		return "model"
	case strings.Contains(name, "publish"):
		return "publication_plan"
	default:
		return "resource"
	}
}

func normalizeReferenceKind(value, fallback string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	switch value {
	case "recipe", "entrypoint", "model", "publication_plan":
		return value
	default:
		return fallback
	}
}

func firstString(value map[string]any, keys ...string) string {
	for _, key := range keys {
		if candidate, ok := value[key].(string); ok && strings.TrimSpace(candidate) != "" {
			return candidate
		}
	}
	return ""
}

func firstScalarString(value map[string]any, keys ...string) string {
	for _, key := range keys {
		if candidate := scalarString(value[key]); candidate != "" {
			return candidate
		}
	}
	return ""
}

func scalarString(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case json.Number:
		return typed.String()
	default:
		return ""
	}
}

func truncateUTF8(value string, maximum int) string {
	if len(value) <= maximum {
		return value
	}
	end := maximum
	for end > 0 && !utf8.RuneStart(value[end]) {
		end--
	}
	return strings.TrimSpace(value[:end]) + "…"
}

func utf8Suffix(value string, maximum int) string {
	if len(value) <= maximum {
		return value
	}
	start := len(value) - maximum
	for start < len(value) && !utf8.RuneStart(value[start]) {
		start++
	}
	return value[start:]
}
