package extproc

import (
	"strings"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/protobuf/proto"
)

const redactedHeaderValue = "[REDACTED]"

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

// redactResponseForLog returns a deep copy of response with the
// values of sensitive headers masked, so a debug-level dump of the ext_proc
// mutation never writes the upstream provider credential to the log. The
// original response (the one actually sent to Envoy) is left untouched.
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
	if imm := clone.GetImmediateResponse(); imm != nil {
		redactHeaderMutation(imm.GetHeaders())
	}
	return clone
}

// responseHeaderMutation returns the HeaderMutation carried by whichever
// request/response phase variant the ProcessingResponse holds, or nil.
func responseHeaderMutation(r *ext_proc.ProcessingResponse) *ext_proc.HeaderMutation {
	switch v := r.Response.(type) {
	case *ext_proc.ProcessingResponse_RequestHeaders:
		return v.RequestHeaders.GetResponse().GetHeaderMutation()
	case *ext_proc.ProcessingResponse_ResponseHeaders:
		return v.ResponseHeaders.GetResponse().GetHeaderMutation()
	case *ext_proc.ProcessingResponse_RequestBody:
		return v.RequestBody.GetResponse().GetHeaderMutation()
	case *ext_proc.ProcessingResponse_ResponseBody:
		return v.ResponseBody.GetResponse().GetHeaderMutation()
	}
	return nil
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
