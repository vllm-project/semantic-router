package classification

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// Parsing extracts a declared verdict. Parse failure is an error, never a safe
// verdict, and the presence of a label never implies a confidence value.
func (v *VLLMJailbreakInference) parseSafetyOutput(output string) (tasks.LabelDecision, error) {
	var result tasks.LabelDecision
	var ok bool
	switch v.determineParserType() {
	case "qwen3guard":
		result, ok = v.parseQwen3GuardFormat(output)
	case "json":
		result, ok = v.parseJSONFormat(output)
	case "simple":
		result, ok = v.parseSimpleFormat(output)
	default:
		result, ok = v.parseQwen3GuardFormat(output)
		if !ok {
			result, ok = v.parseJSONFormat(output)
		}
		if !ok {
			result, ok = v.parseSimpleFormat(output)
		}
	}
	if !ok {
		return tasks.LabelDecision{}, fmt.Errorf("guard response has no recognized safety verdict")
	}
	return result, nil
}

func (v *VLLMJailbreakInference) determineParserType() string {
	if v.parserType != "" && v.parserType != "auto" {
		return v.parserType
	}
	model := strings.ToLower(v.modelName)
	if strings.Contains(model, "qwen3guard") || strings.Contains(model, "qwen_guard") {
		return "qwen3guard"
	}
	if strings.Contains(model, "json") {
		return "json"
	}
	return "auto"
}

var guardSafetyField = regexp.MustCompile(`(?im)^\s*(?:safety|severity\s+level):\s*(safe|unsafe|controversial)\b`)

func (v *VLLMJailbreakInference) parseQwen3GuardFormat(output string) (tasks.LabelDecision, bool) {
	match := guardSafetyField.FindStringSubmatch(output)
	if len(match) < 2 {
		return tasks.LabelDecision{}, false
	}
	return tasks.LabelDecision{Label: strings.ToLower(match[1]), Categories: v.extractCategories(output)}, true
}

func (v *VLLMJailbreakInference) extractCategories(output string) []string {
	matches := regexp.MustCompile(`(?im)^\s*categories?:\s*([^\n]+)`).FindStringSubmatch(output)
	if len(matches) < 2 {
		return nil
	}
	var categories []string
	for _, part := range strings.Split(matches[1], ",") {
		part = strings.TrimSpace(part)
		if part != "" && !strings.EqualFold(part, "none") {
			categories = append(categories, part)
		}
	}
	return categories
}

func (v *VLLMJailbreakInference) parseJSONFormat(output string) (tasks.LabelDecision, bool) {
	var payload struct {
		Safety      string   `json:"safety"`
		IsJailbreak *bool    `json:"is_jailbreak"`
		IsUnsafe    *bool    `json:"is_unsafe"`
		Categories  []string `json:"categories"`
	}
	raw := strings.TrimSpace(output)
	if strings.HasPrefix(raw, "```") {
		raw = strings.TrimPrefix(raw, "```json")
		raw = strings.TrimPrefix(raw, "```")
		raw = strings.TrimSuffix(strings.TrimSpace(raw), "```")
	}
	if json.Unmarshal([]byte(raw), &payload) != nil {
		return tasks.LabelDecision{}, false
	}
	label := strings.ToLower(strings.TrimSpace(payload.Safety))
	if label != "safe" && label != "unsafe" && label != "controversial" {
		verdict := payload.IsJailbreak
		if verdict == nil {
			verdict = payload.IsUnsafe
		}
		if verdict == nil {
			return tasks.LabelDecision{}, false
		}
		label = "safe"
		if *verdict {
			label = "unsafe"
		}
	}
	return tasks.LabelDecision{Label: label, Categories: payload.Categories}, true
}

// Accept an unambiguous verdict token, not substring matches such as "not
// unsafe" or a quoted jailbreak instruction embedded in an explanation.
func (v *VLLMJailbreakInference) parseSimpleFormat(output string) (tasks.LabelDecision, bool) {
	raw := strings.ToLower(strings.TrimSpace(output))
	raw = strings.Trim(raw, " .!\n\r\t")
	for _, prefix := range []string{"this content is ", "the request is "} {
		raw = strings.TrimPrefix(raw, prefix)
	}
	switch raw {
	case "safe", "unsafe", "controversial":
		return tasks.LabelDecision{Label: raw}, true
	case "jailbreak":
		return tasks.LabelDecision{Label: "unsafe"}, true
	}
	return tasks.LabelDecision{}, false
}
