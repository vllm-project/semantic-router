package handlers

import (
	"bytes"
	"encoding/json"
	"strings"
)

var openAIChatAllowedContentPartTypes = map[string]struct{}{
	"":            {},
	"text":        {},
	"output_text": {},
}

func extractOpenAIChatChoiceContent(choice openAIChatChoice) string {
	if content := extractOpenAIChatContent(choice.Message.Content); content != "" {
		return content
	}
	return strings.TrimSpace(choice.Text)
}

func extractOpenAIChatContent(raw json.RawMessage) string {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 || bytes.Equal(trimmed, []byte("null")) {
		return ""
	}

	var asString string
	if err := json.Unmarshal(trimmed, &asString); err == nil {
		return strings.TrimSpace(asString)
	}

	var asStrings []string
	if err := json.Unmarshal(trimmed, &asStrings); err == nil {
		return strings.TrimSpace(strings.Join(asStrings, " "))
	}

	var parts []map[string]any
	if err := json.Unmarshal(trimmed, &parts); err == nil {
		textParts := make([]string, 0, len(parts))
		for _, part := range parts {
			partType, _ := part["type"].(string)
			if _, ok := openAIChatAllowedContentPartTypes[strings.TrimSpace(partType)]; !ok {
				continue
			}
			if text := extractOpenAIChatPartText(part); text != "" {
				textParts = append(textParts, text)
			}
		}
		return strings.TrimSpace(strings.Join(textParts, " "))
	}

	return ""
}

func extractOpenAIChatPartText(part map[string]any) string {
	for _, key := range []string{"text", "output_text", "content"} {
		if text := extractOpenAIChatStringish(part[key]); text != "" {
			return text
		}
	}
	return ""
}

func extractOpenAIChatStringish(value any) string {
	switch typed := value.(type) {
	case string:
		return strings.TrimSpace(typed)
	case []any:
		parts := make([]string, 0, len(typed))
		for _, item := range typed {
			if text := extractOpenAIChatStringish(item); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.TrimSpace(strings.Join(parts, " "))
	case map[string]any:
		for _, key := range []string{"value", "text", "content"} {
			if text := extractOpenAIChatStringish(typed[key]); text != "" {
				return text
			}
		}
	}
	return ""
}
