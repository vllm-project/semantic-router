package extproc

import (
	"fmt"
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/proto"
)

const (
	redactedHeaderValue = "[REDACTED]"
	// redactedBodyValue replaces request/response body content in log dumps.
	// It is distinct from redactedHeaderValue so a log reader can tell a masked
	// credential from withheld content.
	redactedBodyValue = "[REDACTED BODY]"
)

// sensitiveHeaderKeys lists header names whose values are credentials and must
// never be written to logs. The credential resolver injects the upstream
// provider key under one of these (Authorization for OpenAI, x-api-key for
// Anthropic, custom names via provider profiles), so a verbatim dump of the
// ext_proc header mutation would otherwise leak it (CWE-532).
var sensitiveHeaderKeys = map[string]bool{
	"authorization":       true,
	"proxy-authorization": true,
	"x-api-key":           true,
	"api-key":             true,
	"x-goog-api-key":      true,
}

// isSensitiveHeaderKey reports whether a header's value should be redacted from
// logs. It matches the explicit credential headers above and, defensively, any
// header whose name carries a credential by convention (a "*-key" suffix or a
// name containing "api-key", "token", or "secret") so custom provider auth
// headers are covered without enumerating every one.
func isSensitiveHeaderKey(key string) bool {
	k := strings.ToLower(strings.TrimSpace(key))
	if sensitiveHeaderKeys[k] {
		return true
	}
	return strings.HasSuffix(k, "-key") ||
		strings.Contains(k, "api-key") ||
		strings.Contains(k, "token") ||
		strings.Contains(k, "secret")
}

// redactResponseForLog returns a deep copy of response with credential header
// values masked and all request/response body content replaced by a size-only
// placeholder, so a debug-level dump of the ext_proc mutation never writes the
// upstream provider credential or user content to the log (CWE-532). The
// original response (the one actually sent to Envoy) is left untouched.
//
// Bodies are redacted, not dropped: the rewritten request body carries the
// prompt (including any RAG/memory context injected by the router) and an
// immediate response carries generated or cached completion text, so neither
// belongs in normal diagnostics. The byte count is preserved because size is
// what makes the log useful for debugging mutation and truncation problems.
func redactResponseForLog(response *ext_proc.ProcessingResponse) *ext_proc.ProcessingResponse {
	if response == nil {
		return nil
	}
	clone, ok := proto.Clone(response).(*ext_proc.ProcessingResponse)
	if !ok || clone == nil {
		// Cloning failed: never fall back to the unredacted original.
		return nil
	}
	redactHeaderMutation(responseHeaderMutation(clone))
	redactBodyMutation(responseCommonResponse(clone).GetBodyMutation())
	if imm := clone.GetImmediateResponse(); imm != nil {
		redactHeaderMutation(imm.GetHeaders())
		imm.Body = redactedBodyPlaceholder(imm.GetBody())
	}
	return clone
}

// redactedBodyPlaceholder replaces body bytes with a size-only marker. An empty
// body stays empty so the log distinguishes "no body" from "body withheld".
func redactedBodyPlaceholder(body []byte) []byte {
	if len(body) == 0 {
		return body
	}
	return []byte(fmt.Sprintf("%s %d bytes", redactedBodyValue, len(body)))
}

// redactBodyMutation masks body content in place on the clone. Only the inline
// body variants carry content; ClearBody is a boolean directive with nothing to
// redact.
func redactBodyMutation(bm *ext_proc.BodyMutation) {
	if bm == nil {
		return
	}
	switch m := bm.Mutation.(type) {
	case *ext_proc.BodyMutation_Body:
		m.Body = redactedBodyPlaceholder(m.Body)
	case *ext_proc.BodyMutation_StreamedResponse:
		if m.StreamedResponse != nil {
			m.StreamedResponse.Body = redactedBodyPlaceholder(m.StreamedResponse.GetBody())
		}
	}
}

// responseCommonResponse returns the CommonResponse carried by whichever
// request/response phase variant the ProcessingResponse holds, or nil. Header
// and body mutations both hang off it, so the phase switch lives in one place.
func responseCommonResponse(r *ext_proc.ProcessingResponse) *ext_proc.CommonResponse {
	switch v := r.Response.(type) {
	case *ext_proc.ProcessingResponse_RequestHeaders:
		return v.RequestHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_ResponseHeaders:
		return v.ResponseHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_RequestBody:
		return v.RequestBody.GetResponse()
	case *ext_proc.ProcessingResponse_ResponseBody:
		return v.ResponseBody.GetResponse()
	}
	return nil
}

// responseHeaderMutation returns the HeaderMutation carried by whichever
// request/response phase variant the ProcessingResponse holds, or nil.
func responseHeaderMutation(r *ext_proc.ProcessingResponse) *ext_proc.HeaderMutation {
	return responseCommonResponse(r).GetHeaderMutation()
}

// redactHeaderMutation masks the value of every sensitive set-header in place.
// It operates on a clone, so the mutation sent to Envoy is unaffected.
func redactHeaderMutation(hm *ext_proc.HeaderMutation) {
	if hm == nil {
		return
	}
	for _, opt := range hm.SetHeaders {
		redactHeaderValueOption(opt)
	}
}

func redactHeaderValueOption(opt *core.HeaderValueOption) {
	if opt == nil || opt.Header == nil {
		return
	}
	if !isSensitiveHeaderKey(opt.Header.Key) {
		return
	}
	if opt.Header.Value != "" {
		opt.Header.Value = redactedHeaderValue
	}
	if len(opt.Header.RawValue) > 0 {
		opt.Header.RawValue = []byte(redactedHeaderValue)
	}
}
