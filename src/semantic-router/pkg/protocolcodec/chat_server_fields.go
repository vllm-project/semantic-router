package protocolcodec

import "encoding/json"

// chatServerFields are the members a model server adds about itself: its build
// (vLLM's system_fingerprint names its version, commit and parallel layout),
// the rendered prompt and its token ids, expert routing, KV and encoder-cache
// transfer, and per-request metrics. A Chat client never receives them,
// whatever their value, though the neutral decode accepts and reports them.
var chatServerFields = []string{
	"system_fingerprint", "prompt_logprobs", "prompt_token_ids", "prompt_text", "prompt_routed_experts",
	"kv_transfer_params", "ec_transfer_params", "metrics",
	"do_remote_decode", "do_remote_prefill", "remote_block_ids", "remote_engine_id", "remote_host", "remote_port",
}

// chatChoiceServerFields are the members of the same kind on a choice.
var chatChoiceServerFields = []string{"token_ids", "routed_experts", "weight_versions"}

// publicChatResponseBody is a buffered Chat reply as a Chat client receives it
// from the same format, without the server's own members. It reports whether
// that differs from body.
func publicChatResponseBody(body []byte) ([]byte, bool, error) {
	var object map[string]json.RawMessage
	if err := json.Unmarshal(body, &object); err != nil {
		return nil, false, err
	}
	changed, err := deleteChatServerMembers(object)
	if err != nil || !changed {
		return body, false, err
	}
	encoded, err := json.Marshal(object)
	return encoded, err == nil, err
}

// deleteChatServerMembers removes the server's own members from a Chat reply or
// stream chunk, at its top level and on its choices.
func deleteChatServerMembers(object map[string]json.RawMessage) (bool, error) {
	changed := false
	for _, name := range chatServerFields {
		if _, present := object[name]; present {
			delete(object, name)
			changed = true
		}
	}
	raw, present := object["choices"]
	if !present || !jsonFieldPresent(raw) {
		return changed, nil
	}
	var choices []json.RawMessage
	if err := json.Unmarshal(raw, &choices); err != nil {
		return false, err
	}
	choicesChanged := false
	for index, choice := range choices {
		stripped, removed, err := removeChatJSONFields(choice, chatChoiceServerFields...)
		if err != nil {
			return false, err
		}
		if removed {
			choices[index], choicesChanged = stripped, true
		}
	}
	if !choicesChanged {
		return changed, nil
	}
	encoded, err := json.Marshal(choices)
	if err != nil {
		return false, err
	}
	object["choices"] = encoded
	return true, nil
}
