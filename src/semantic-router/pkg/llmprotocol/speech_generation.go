package llmprotocol

// SpeechGenerationOptions carries a Speech API (/v1/audio/speech) request in
// the neutral IR. Its presence is what makes a request require
// CapabilitySpeechGeneration.
type SpeechGenerationOptions struct {
	Input string
	Voice string
}
