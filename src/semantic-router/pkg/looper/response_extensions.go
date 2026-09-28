package looper

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// ResponseExtension is optional router-generated evidence. Its contents are
// private: provider decoding and callers constructing Response literals cannot
// manufacture provenance by choosing a JSON field name.
type ResponseExtension struct {
	name  string
	value json.RawMessage
}

func (extension ResponseExtension) Name() string          { return extension.name }
func (extension ResponseExtension) JSON() json.RawMessage { return bytes.Clone(extension.value) }

// ProtocolBody returns an independent snapshot with no router-added evidence.
// Without an emitter-owned snapshot, every original field remains subject to
// strict provider validation; field names are never treated as provenance.
func (response *Response) ProtocolBody() []byte {
	if response.protocolBody != nil {
		return bytes.Clone(response.protocolBody)
	}
	return bytes.Clone(response.Body)
}

// RouterExtensions returns detached evidence; changing it cannot change the
// original transport snapshot or the direct Go caller's compatibility body.
func (response *Response) RouterExtensions() []ResponseExtension {
	extensions := make([]ResponseExtension, len(response.routerExtensions))
	for i, extension := range response.routerExtensions {
		extensions[i] = ResponseExtension{name: extension.name, value: bytes.Clone(extension.value)}
	}
	return extensions
}

func withFusionExtension(response *Response, cfg fusionExecutionConfig, trace *FusionTrace) (*Response, error) {
	if !shouldIncludeFusionTrace(cfg, trace) {
		return response, nil
	}
	return withRouterExtension(response, "fusion", projectFusionPublicTrace(trace))
}

func withWorkflowExtension(response *Response, cfg workflowsExecutionConfig, trace *workflowTrace) (*Response, error) {
	if trace == nil || (!cfg.IncludeIntermediateResponses && len(trace.FailedModels) == 0) {
		return response, nil
	}
	return withRouterExtension(response, "flow", trace)
}

func withReMoMExtension(response *Response, include bool, rounds []RoundResponse) (*Response, error) {
	if !include {
		return response, nil
	}
	return withRouterExtension(response, "reasoning_mom_responses", rounds)
}

// Only the three typed emitter adapters above call this helper. Capture the
// original bytes before serialization, never by stripping a completed body.
func withRouterExtension(response *Response, name string, value interface{}) (*Response, error) {
	encoded, err := json.Marshal(value)
	if err != nil {
		return nil, fmt.Errorf("marshal router extension: %w", err)
	}
	original := bytes.Clone(response.Body)
	body, err := appendCompatibilityExtension(original, name, encoded, response.ContentType == "text/event-stream")
	if err != nil {
		return nil, err
	}
	response.protocolBody = original
	response.routerExtensions = []ResponseExtension{{name: name, value: bytes.Clone(encoded)}}
	response.Body = body
	return response, nil
}

// appendCompatibilityExtension is only the direct Go API view. ExtProc never
// sends it: final protocol encoding, limits and diagnostics belong to transport.
func appendCompatibilityExtension(body []byte, name string, value json.RawMessage, streaming bool) ([]byte, error) {
	if streaming {
		lines := bytes.Split(body, []byte("\n"))
		for i, line := range lines {
			payload, ok := bytes.CutPrefix(line, []byte("data: "))
			if !ok || len(payload) == 0 || payload[0] != '{' {
				continue
			}
			encoded, err := appendCompatibilityExtension(payload, name, value, false)
			if err != nil {
				return nil, err
			}
			lines[i] = append([]byte("data: "), encoded...)
			return bytes.Join(lines, []byte("\n")), nil
		}
		return nil, fmt.Errorf("router stream has no JSON data frame")
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(body, &fields); err != nil {
		return nil, err
	}
	fields[name] = value
	return json.Marshal(fields)
}
