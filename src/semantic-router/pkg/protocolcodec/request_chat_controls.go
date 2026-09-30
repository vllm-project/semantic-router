package protocolcodec

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"

// Provider controls must never disappear when a request changes wire format.
func rejectChatOnlyControls(request llmprotocol.Request) error {
	for _, field := range []struct {
		name    string
		present bool
	}{
		{"min_p", request.Sampling.MinP != nil},
		{"repetition_penalty", request.Sampling.RepetitionPenalty != nil},
		{"cache_salt", request.CacheSalt != nil},
	} {
		if field.present {
			return llmprotocol.NewError(llmprotocol.ErrorUnsupportedFeature, "unsupported_"+field.name, "target protocol cannot preserve "+field.name, nil)
		}
	}
	return rejectUnsupportedRequestField("chat_template_kwargs", request.ChatTemplateKwargs)
}
