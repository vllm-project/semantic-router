package llmprotocol

import "strings"

// ModelCapabilities projects model declarations onto the protocol capability
// vocabulary. Catalog input-modality aliases are accepted here, while labels
// such as long_context or coding remain descriptive metadata. Such labels must
// not erase recognized declarations in the same model card.
//
// The bool distinguishes a recognized declaration from an unannotated model.
// ParseCapabilities remains strict for protocol contracts; this projection is
// only for the open vocabulary of model metadata. Generated media capabilities
// require explicit declarations; vision/audio/video only describe input.
func ModelCapabilities(names []string) (CapabilitySet, bool) {
	var result CapabilitySet
	annotated := false
	for _, name := range names {
		canonical := strings.ToLower(strings.TrimSpace(name))
		switch canonical {
		case "vision":
			canonical = "image_input"
		case "audio":
			canonical = "audio_input"
		case "video":
			canonical = "video_input"
		case "structured_output":
			canonical = "structured_json"
		case "tool_use":
			canonical = "tools"
		}
		parsed, err := ParseCapabilities([]string{canonical})
		if err != nil {
			continue
		}
		annotated = true
		result.bits |= parsed.bits
	}
	return result, annotated
}
