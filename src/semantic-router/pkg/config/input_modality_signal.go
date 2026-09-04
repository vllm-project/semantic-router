package config

import "strings"

const SignalTypeInputModality = "input_modality"

// Input modalities detectable from request structure. A request may contain
// several at once; each InputModalityRule matches the presence of one.
const (
	InputModalityText  = "text"
	InputModalityImage = "image"
	InputModalityAudio = "audio"
	InputModalityVideo = "video"
)

// InputModalityRule deterministically matches when the parsed request contains
// at least one content part of the declared input modality. Detection is
// structural — no classifier, embedding model, or other ML inference runs —
// and is independent of the intended output modality (the modality signal)
// and of which payload gets embedded (embedding query_modality).
type InputModalityRule struct {
	Name        string `yaml:"name"`
	Description string `yaml:"description,omitempty"`
	Modality    string `yaml:"modality"`
}

// SupportedInputModalities lists the modality values an InputModalityRule may
// declare.
func SupportedInputModalities() []string {
	return []string{InputModalityText, InputModalityImage, InputModalityAudio, InputModalityVideo}
}

// InputModalityForContentPartType maps a raw message content-part type to the
// input modality it carries. The data plane decodes every wire protocol into
// the neutral request and counts modalities from its content kinds; this table
// serves the classify/eval HTTP APIs, which walk raw messages[].content parts,
// so part-type aliases stay in one place. Only Chat Completions and Response
// API part types are listed: those APIs accept OpenAI-shaped messages, and the
// image set matches what the conversation signal has always counted there.
// The type is normalized (trimmed, lowercased) before matching. Returns false
// for non-modality part types such as tool_use or refusal.
func InputModalityForContentPartType(partType string) (string, bool) {
	switch strings.ToLower(strings.TrimSpace(partType)) {
	case "text", "input_text":
		return InputModalityText, true
	case "image_url", "input_image":
		return InputModalityImage, true
	case "input_audio", "audio_url":
		return InputModalityAudio, true
	case "video_url", "input_video":
		return InputModalityVideo, true
	default:
		return "", false
	}
}
