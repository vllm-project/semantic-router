package graph

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"strings"
)

// Request is the chat completions request a call step sends, as a JSON
// object. Fields a step does not touch pass through byte for byte.
type Request struct {
	fields map[string]json.RawMessage
}

// ParseRequest reads a chat completions request body.
func ParseRequest(body []byte) (Request, error) {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return Request{}, fmt.Errorf("graph: the request is not a JSON object: %w", err)
	}
	if fields == nil {
		return Request{}, errors.New("graph: the request is not a JSON object")
	}
	return Request{fields: fields}, nil
}

// Clone returns a copy that changes independently of r.
func (r Request) Clone() Request {
	return Request{fields: maps.Clone(r.fields)}
}

// Body encodes the request.
func (r Request) Body() ([]byte, error) {
	if r.fields == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(r.fields)
}

// Model returns the request's model field.
func (r Request) Model() string {
	var model string
	_ = json.Unmarshal(r.fields["model"], &model)
	return model
}

// Set encodes value into the named field.
func (r *Request) Set(name string, value any) error {
	raw, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("graph: encode request field %q: %w", name, err)
	}
	r.SetRaw(name, raw)
	return nil
}

// SetRaw stores an encoded value in the named field.
func (r *Request) SetRaw(name string, raw json.RawMessage) {
	if r.fields == nil {
		r.fields = map[string]json.RawMessage{}
	}
	r.fields[name] = raw
}

// Messages returns the request's messages, each a JSON object.
func (r Request) Messages() ([]map[string]json.RawMessage, error) {
	raw, ok := r.fields["messages"]
	if !ok {
		return nil, nil
	}
	var messages []map[string]json.RawMessage
	if err := json.Unmarshal(raw, &messages); err != nil {
		return nil, fmt.Errorf("graph: the request's messages are not a list of objects: %w", err)
	}
	return messages, nil
}

// SetMessages replaces the request's messages.
func (r *Request) SetMessages(messages []map[string]json.RawMessage) error {
	return r.Set("messages", messages)
}

// Message builds a text message for role.
func Message(role, content string) map[string]json.RawMessage {
	roleJSON, _ := json.Marshal(role)
	contentJSON, _ := json.Marshal(content)
	return map[string]json.RawMessage{"role": roleJSON, "content": contentJSON}
}

// MessageRole returns a message's role.
func MessageRole(message map[string]json.RawMessage) string {
	var role string
	_ = json.Unmarshal(message["role"], &role)
	return role
}

// MessageText returns a message's text: its string content, or the text
// parts of its content list joined by newlines.
func MessageText(message map[string]json.RawMessage) string {
	raw := message["content"]
	var text string
	if json.Unmarshal(raw, &text) == nil {
		return text
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if json.Unmarshal(raw, &parts) != nil {
		return ""
	}
	texts := make([]string, 0, len(parts))
	for _, part := range parts {
		if part.Type == "text" {
			texts = append(texts, part.Text)
		}
	}
	return strings.Join(texts, "\n")
}

// completionText returns the first choice's text of a chat completion or of
// its event stream.
func completionText(body []byte) string {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return ""
	}
	if body[0] == '{' {
		var completion struct {
			Choices []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
				Text string `json:"text"`
			} `json:"choices"`
		}
		if json.Unmarshal(body, &completion) != nil || len(completion.Choices) == 0 {
			return ""
		}
		choice := completion.Choices[0]
		if choice.Message.Content != "" {
			return choice.Message.Content
		}
		return choice.Text
	}
	var text strings.Builder
	for _, data := range eventData(body) {
		var chunk struct {
			Choices []struct {
				Index int `json:"index"`
				Delta struct {
					Content string `json:"content"`
				} `json:"delta"`
			} `json:"choices"`
		}
		if json.Unmarshal(data, &chunk) != nil {
			continue
		}
		for _, choice := range chunk.Choices {
			if choice.Index == 0 {
				text.WriteString(choice.Delta.Content)
			}
		}
	}
	return text.String()
}

// completionUsage returns the usage a chat completion, or the last usage
// chunk of its event stream, reports.
func completionUsage(body []byte) Usage {
	body = bytes.TrimSpace(body)
	if len(body) == 0 {
		return Usage{}
	}
	payloads := [][]byte{body}
	if body[0] != '{' {
		payloads = eventData(body)
	}
	var usage Usage
	for _, payload := range payloads {
		var carrier struct {
			Usage *Usage `json:"usage"`
		}
		if json.Unmarshal(payload, &carrier) == nil && carrier.Usage != nil {
			usage = *carrier.Usage
		}
	}
	return usage
}

// eventData returns the data payloads of a server-sent event stream, without
// the terminating [DONE].
func eventData(body []byte) [][]byte {
	var payloads [][]byte
	for _, line := range bytes.Split(body, []byte("\n")) {
		line = bytes.TrimSpace(line)
		data, ok := bytes.CutPrefix(line, []byte("data:"))
		if !ok {
			continue
		}
		data = bytes.TrimSpace(data)
		if len(data) == 0 || bytes.Equal(data, []byte("[DONE]")) {
			continue
		}
		payloads = append(payloads, data)
	}
	return payloads
}
