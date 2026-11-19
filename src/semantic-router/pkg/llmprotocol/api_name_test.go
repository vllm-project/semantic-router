package llmprotocol

import (
	"errors"
	"strings"
	"testing"
)

// A refusal names the API as clients call it, never the wire format's
// identifier.
func TestRequireCapabilitiesNamesTheAPI(t *testing.T) {
	for format, name := range map[WireFormat]string{
		OpenAIChatV1: "Chat Completions", OpenAIResponsesV1: "Responses",
		AnthropicMessagesV1: "Anthropic Messages", OpenAIImagesV1: "Images",
	} {
		if format.APIName() != name {
			t.Fatalf("%s.APIName() = %q, want %q", format, format.APIName(), name)
		}
		err := RequireCapabilities(format, Capabilities(CapabilityText), Capabilities(CapabilityText, CapabilityAudioInput))
		var protocolError *ProtocolError
		if !errors.As(err, &protocolError) || protocolError.Message != "the "+name+" API does not support: audio_input" ||
			strings.Contains(protocolError.Message, string(format)) {
			t.Fatalf("%s refusal = %v", format, err)
		}
	}
}
