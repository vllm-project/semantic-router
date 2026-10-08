package llmprotocol

import "encoding/json"

// SpeechGenerationOptions carries a Speech API (/v1/audio/speech) request in
// the neutral IR. Its presence is what makes a request require
// CapabilitySpeechGeneration.
//
// The text to speak is the last user message, not a field here. Fields holds
// every other request field by its wire name, with the value kept as raw JSON
// so the backend receives it unchanged.
type SpeechGenerationOptions struct {
	Fields map[string]json.RawMessage
}
