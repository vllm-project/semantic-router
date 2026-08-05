package looper

import (
	"encoding/json"
	"sort"
	"strconv"
	"strings"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

const (
	maxRequestOutputContractChars = 2000
	jsonActionMinCommandDuration  = 1.0
	requiredOutputContractMarker  = "Required output contract:"
)

func applyFinalOutputContract(spec *config.OutputContractSpec, resp *ModelResponse) {
	if resp == nil || resp.HasToolCalls {
		return
	}
	if !requestsSingleChoice(spec) {
		return
	}
	if answer, ok := extractSingleChoiceAnswerFromResponse(resp, spec); ok {
		resp.Content = renderSingleChoiceAnswer(answer, spec)
	}
}

func applyJSONActionOutputContract(spec *config.OutputContractSpec, resp *ModelResponse, candidates []*ModelResponse) {
	if resp == nil {
		return
	}
	if !requestsJSONAction(spec) {
		return
	}
	for _, source := range outputContractExtractSources(spec) {
		if content, ok := extractJSONActionFromSource(resp, candidates, source, spec); ok {
			replaceModelResponseContent(resp, content)
			return
		}
	}
}

func applyReferenceSelectionOutputContract(spec *config.OutputContractSpec, resp *ModelResponse, candidates []*ModelResponse) {
	if resp == nil || !requestsReferenceSelection(spec) || !shouldDereferenceReferenceSelection(spec) {
		return
	}
	index, ok := extractReferenceSelectionIndexFromResponse(resp, spec)
	if !ok || index < 1 || index > len(candidates) {
		return
	}
	candidate := candidates[index-1]
	if candidate == nil {
		return
	}
	content := strings.TrimSpace(candidate.Content)
	if content == "" {
		content = strings.TrimSpace(candidate.ReasoningContent)
	}
	if content == "" {
		return
	}
	replaceModelResponseContent(resp, content)
}

func extractJSONActionFromSource(resp *ModelResponse, candidates []*ModelResponse, source string, spec *config.OutputContractSpec) (string, bool) {
	switch source {
	case config.OutputContractExtractSourceContent:
		return extractJSONActionObject(resp.Content, spec)
	case config.OutputContractExtractSourceReasoningContent:
		return extractJSONActionObject(resp.ReasoningContent, spec)
	case config.OutputContractExtractSourceCandidateResponses:
		for _, candidate := range candidates {
			if candidate == nil {
				continue
			}
			if content, ok := extractJSONActionObject(candidate.Content, spec); ok {
				return content, true
			}
			if content, ok := extractJSONActionObject(candidate.ReasoningContent, spec); ok {
				return content, true
			}
		}
	}
	return "", false
}

func extractReferenceSelectionIndexFromResponse(resp *ModelResponse, spec *config.OutputContractSpec) (int, bool) {
	for _, source := range outputContractExtractSources(spec) {
		content, ok := responseContentForOutputContractSource(resp, source)
		if !ok {
			continue
		}
		if index, ok := extractReferenceSelectionIndex(content, spec); ok {
			return index, true
		}
	}
	return 0, false
}

func extractReferenceSelectionIndex(content string, spec *config.OutputContractSpec) (int, bool) {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return 0, false
	}
	idFormat := config.OutputContractReferenceIDFormatIndex
	if spec != nil && spec.Reference != nil && strings.TrimSpace(spec.Reference.IDFormat) != "" {
		idFormat = strings.TrimSpace(spec.Reference.IDFormat)
	}
	switch idFormat {
	case config.OutputContractReferenceIDFormatIndex:
		index, err := strconv.Atoi(trimmed)
		return index, err == nil
	case config.OutputContractReferenceIDFormatReferenceNumber:
		parts := strings.Fields(trimmed)
		if len(parts) != 2 || !strings.EqualFold(parts[0], "reference") {
			return 0, false
		}
		index, err := strconv.Atoi(parts[1])
		return index, err == nil
	default:
		return 0, false
	}
}

func shouldDereferenceReferenceSelection(spec *config.OutputContractSpec) bool {
	if spec == nil {
		return false
	}
	for _, postprocess := range spec.Postprocess {
		if strings.TrimSpace(postprocess.Type) == config.OutputContractPostprocessDereferenceSelectedReference {
			return true
		}
	}
	return false
}

func replaceModelResponseContent(resp *ModelResponse, content string) {
	resp.Content = content
	resp.HasToolCalls = false
	resp.Raw = nil
}

func extractJSONActionObject(content string, spec *config.OutputContractSpec) (string, bool) {
	candidates := []string{strings.TrimSpace(content)}
	if outputContractExtractMode(spec) == config.OutputContractExtractModeJSONObject {
		candidates = jsonObjectParseCandidates(content)
	}
	for _, candidate := range candidates {
		action, ok := parseJSONActionObject(candidate)
		if !ok {
			continue
		}
		body, err := formatJSONActionObject(action)
		if err != nil {
			continue
		}
		return body, true
	}
	return "", false
}

func outputContractExtractMode(spec *config.OutputContractSpec) string {
	if spec == nil || spec.Extract == nil || strings.TrimSpace(spec.Extract.Mode) == "" {
		return config.OutputContractExtractModeExact
	}
	return strings.TrimSpace(spec.Extract.Mode)
}

func parseJSONActionObject(candidate string) (map[string]interface{}, bool) {
	var action map[string]interface{}
	if err := json.Unmarshal([]byte(candidate), &action); err != nil {
		return nil, false
	}
	_, hasCommands := action["commands"]
	_, hasTaskComplete := action["task_complete"]
	if !hasCommands && !hasTaskComplete {
		return nil, false
	}
	if hasCommands && !jsonActionCommandsValid(action["commands"]) {
		return nil, false
	}
	normalizeJSONActionCommands(action)
	if _, ok := action["analysis"].(string); !ok {
		action["analysis"] = "Continue from the current terminal state."
	}
	if _, ok := action["plan"].(string); !ok {
		action["plan"] = "Run the listed terminal commands and inspect the result."
	}
	if !hasCommands {
		action["commands"] = []interface{}{}
	}
	if !hasTaskComplete {
		action["task_complete"] = false
	}
	return action, true
}

func jsonActionCommandsValid(value interface{}) bool {
	commands, ok := value.([]interface{})
	if !ok {
		return false
	}
	for _, raw := range commands {
		command, ok := raw.(map[string]interface{})
		if !ok {
			return false
		}
		if _, ok := command["keystrokes"].(string); !ok {
			return false
		}
	}
	return true
}

func normalizeJSONActionCommands(action map[string]interface{}) {
	commands, ok := action["commands"].([]interface{})
	if !ok {
		return
	}
	for _, raw := range commands {
		command, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		duration, ok := command["duration"].(float64)
		if !ok || duration < jsonActionMinCommandDuration {
			command["duration"] = jsonActionMinCommandDuration
		}
	}
}

func formatJSONActionObject(action map[string]interface{}) (string, error) {
	orderedKeys := []string{"analysis", "plan", "commands", "task_complete"}
	ordered := map[string]bool{}
	for _, key := range orderedKeys {
		ordered[key] = true
	}

	var builder strings.Builder
	builder.WriteByte('{')
	wrote := false
	for _, key := range orderedKeys {
		value, ok := action[key]
		if !ok {
			continue
		}
		if wrote {
			builder.WriteByte(',')
		}
		if err := writeJSONActionField(&builder, key, value); err != nil {
			return "", err
		}
		wrote = true
	}

	extraKeys := make([]string, 0, len(action))
	for key := range action {
		if !ordered[key] {
			extraKeys = append(extraKeys, key)
		}
	}
	sort.Strings(extraKeys)
	for _, key := range extraKeys {
		if wrote {
			builder.WriteByte(',')
		}
		if err := writeJSONActionField(&builder, key, action[key]); err != nil {
			return "", err
		}
		wrote = true
	}
	builder.WriteByte('}')
	return builder.String(), nil
}

func writeJSONActionField(builder *strings.Builder, key string, value interface{}) error {
	keyBytes, err := json.Marshal(key)
	if err != nil {
		return err
	}
	valueBytes, err := json.Marshal(value)
	if err != nil {
		return err
	}
	builder.Write(keyBytes)
	builder.WriteByte(':')
	builder.Write(valueBytes)
	return nil
}

func applyWorkflowSingleChoiceFallback(spec *config.OutputContractSpec, resp *ModelResponse, stepResults []workflowStepResult) {
	if resp == nil || resp.HasToolCalls || !requestsSingleChoice(spec) {
		return
	}
	if _, ok := extractSingleChoiceAnswer(resp.Content, spec); ok {
		return
	}
	answer, ok := workflowMajoritySingleChoiceAnswer(stepResults, spec)
	if ok {
		resp.Content = renderSingleChoiceAnswer(answer, spec)
	}
}

func workflowMajoritySingleChoiceAnswer(stepResults []workflowStepResult, spec *config.OutputContractSpec) (string, bool) {
	counts := collectWorkflowSingleChoiceAnswers(stepResults, spec)
	return chooseWorkflowSingleChoiceAnswer(counts)
}

type workflowSingleChoiceTally struct {
	count int
	first int
}

func collectWorkflowSingleChoiceAnswers(
	stepResults []workflowStepResult,
	spec *config.OutputContractSpec,
) map[string]workflowSingleChoiceTally {
	counts := map[string]workflowSingleChoiceTally{}
	for _, step := range stepResults {
		for _, resp := range step.responses {
			recordWorkflowSingleChoiceAnswer(counts, resp, spec)
		}
	}
	return counts
}

func recordWorkflowSingleChoiceAnswer(
	counts map[string]workflowSingleChoiceTally,
	resp *ModelResponse,
	spec *config.OutputContractSpec,
) {
	answer, ok := workflowSingleChoiceAnswer(resp, spec)
	if !ok {
		return
	}
	entry := counts[answer]
	if entry.count == 0 {
		entry.first = len(counts)
	}
	entry.count++
	counts[answer] = entry
}

func workflowSingleChoiceAnswer(resp *ModelResponse, spec *config.OutputContractSpec) (string, bool) {
	if resp == nil {
		return "", false
	}
	answer, ok := extractSingleChoiceAnswer(resp.Content, spec)
	if !ok && strings.TrimSpace(resp.Content) == "" {
		answer, ok = extractSingleChoiceAnswer(resp.ReasoningContent, spec)
	}
	return answer, ok
}

func chooseWorkflowSingleChoiceAnswer(counts map[string]workflowSingleChoiceTally) (string, bool) {
	best := ""
	bestCount := 0
	bestFirst := int(^uint(0) >> 1)
	for answer, entry := range counts {
		if entry.count > bestCount || (entry.count == bestCount && entry.first < bestFirst) {
			best = answer
			bestCount = entry.count
			bestFirst = entry.first
		}
	}
	return best, best != ""
}

func requestTextWithOutputContract(original string, req *openai.ChatCompletionNewParams, configuredContracts ...string) string {
	return textWithOutputContract(original, requestOutputContract(req, configuredContracts...))
}

func requestOutputContract(req *openai.ChatCompletionNewParams, configuredContracts ...string) string {
	contracts := make([]string, 0, 1+len(configuredContracts))
	contracts = append(contracts, extractRequestOutputContract(req))
	contracts = append(contracts, configuredContracts...)
	return mergeOutputContracts(contracts...)
}

func mergeOutputContracts(contracts ...string) string {
	seen := map[string]bool{}
	merged := make([]string, 0, len(contracts))
	for _, contract := range contracts {
		contract = strings.TrimSpace(contract)
		if contract == "" || seen[contract] {
			continue
		}
		seen[contract] = true
		merged = append(merged, contract)
	}
	return truncateOutputContract(strings.Join(merged, "\n\n"))
}

func textWithOutputContract(original string, outputContract string) string {
	outputContract = strings.TrimSpace(outputContract)
	if outputContract == "" {
		return original
	}
	return strings.TrimSpace(original + "\n\n" + requiredOutputContractMarker + "\n" + outputContract)
}

func extractRequestOutputContract(req *openai.ChatCompletionNewParams) string {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return ""
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return ""
	}

	seen := map[string]bool{}
	var contracts []string
	for _, raw := range messages {
		message, ok := raw.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		if role == "assistant" || role == "tool" {
			continue
		}
		contract := embeddedOutputContract(requestContentText(message["content"]))
		if contract == "" || seen[contract] {
			continue
		}
		seen[contract] = true
		contracts = append(contracts, contract)
	}
	return strings.Join(contracts, "\n\n")
}

func requestContentText(content interface{}) string {
	switch typed := content.(type) {
	case string:
		return typed
	case []interface{}:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			if text := requestContentText(item); strings.TrimSpace(text) != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	case map[string]interface{}:
		for _, key := range []string{"text", "content"} {
			if text, ok := typed[key].(string); ok {
				return text
			}
		}
	}
	return ""
}

func truncateOutputContract(text string) string {
	runes := []rune(text)
	if len(runes) <= maxRequestOutputContractChars {
		return text
	}
	return strings.TrimSpace(string(runes[:maxRequestOutputContractChars]))
}

func formatOutputContractForPrompt(outputContract string) string {
	outputContract = strings.TrimSpace(outputContract)
	if outputContract == "" {
		return ""
	}
	return "\n" + requiredOutputContractMarker + "\n" + outputContract + "\n"
}

func embeddedOutputContract(text string) string {
	idx := strings.LastIndex(text, requiredOutputContractMarker)
	if idx < 0 {
		return ""
	}
	return strings.TrimSpace(text[idx+len(requiredOutputContractMarker):])
}

func appendOutputContractForPrompt(prompt string, outputContract string) string {
	outputContract = strings.TrimSpace(outputContract)
	if outputContract == "" {
		return prompt
	}
	return strings.TrimSpace(prompt) + formatOutputContractForPrompt(outputContract)
}

func requestsJSONAction(spec *config.OutputContractSpec) bool {
	return spec != nil &&
		strings.TrimSpace(spec.Type) == config.OutputContractTypeStructuredJSON &&
		spec.JSONSchema != nil &&
		strings.TrimSpace(spec.JSONSchema.SchemaRef) == config.OutputContractJSONTerminalActionV1
}

func requestsSingleChoice(spec *config.OutputContractSpec) bool {
	return spec != nil && strings.TrimSpace(spec.Type) == config.OutputContractTypeChoice
}

func requestsReferenceSelection(spec *config.OutputContractSpec) bool {
	return spec != nil && strings.TrimSpace(spec.Type) == config.OutputContractTypeReferenceSelect
}

func extractSingleChoiceAnswerFromResponse(resp *ModelResponse, spec *config.OutputContractSpec) (string, bool) {
	for _, source := range outputContractExtractSources(spec) {
		content, ok := responseContentForOutputContractSource(resp, source)
		if !ok {
			continue
		}
		if answer, ok := extractSingleChoiceAnswer(content, spec); ok {
			return answer, true
		}
	}
	return "", false
}

func extractSingleChoiceAnswer(content string, spec *config.OutputContractSpec) (string, bool) {
	choices := outputContractChoiceValues(spec)
	if len(choices) == 0 {
		return "", false
	}
	trimmed := strings.TrimSpace(content)
	for _, choice := range choices {
		if trimmed == choice {
			return choice, true
		}
	}
	return "", false
}

func outputContractChoiceValues(spec *config.OutputContractSpec) []string {
	if spec == nil || spec.ChoiceSet == nil {
		return nil
	}
	values := make([]string, 0, len(spec.ChoiceSet.Values))
	for _, value := range spec.ChoiceSet.Values {
		value = strings.TrimSpace(value)
		if value != "" {
			values = append(values, value)
		}
	}
	return values
}

func renderSingleChoiceAnswer(answer string, spec *config.OutputContractSpec) string {
	if spec == nil || spec.Render == nil {
		return answer
	}
	if strings.TrimSpace(spec.Render.Mode) != config.OutputContractRenderModeTemplate {
		return answer
	}
	rendered := strings.ReplaceAll(spec.Render.Template, "{{choice}}", answer)
	rendered = strings.ReplaceAll(rendered, "{choice}", answer)
	return strings.TrimSpace(rendered)
}

func outputContractExtractSources(spec *config.OutputContractSpec) []string {
	if spec == nil || spec.Extract == nil || len(spec.Extract.Sources) == 0 {
		return []string{config.OutputContractExtractSourceContent}
	}
	sources := make([]string, 0, len(spec.Extract.Sources))
	for _, source := range spec.Extract.Sources {
		source = strings.TrimSpace(source)
		if source != "" {
			sources = append(sources, source)
		}
	}
	if len(sources) == 0 {
		return []string{config.OutputContractExtractSourceContent}
	}
	return sources
}

func responseContentForOutputContractSource(resp *ModelResponse, source string) (string, bool) {
	if resp == nil {
		return "", false
	}
	switch source {
	case config.OutputContractExtractSourceContent:
		return resp.Content, true
	case config.OutputContractExtractSourceReasoningContent:
		return resp.ReasoningContent, true
	default:
		return "", false
	}
}
