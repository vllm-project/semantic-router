package extproc

import (
	"fmt"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// These canaries pin the content-free diagnostics contract from issue #2464.
//
// utils.go logs the ext_proc mutation with `%+v`, so the assertions below format
// the redacted response exactly the way the logger does rather than walking a
// subset of fields. A header-only dump helper cannot observe a body leak, so it
// would pass while prompt or response content still reached the log.

// bodyMutationResponse builds a request-body ProcessingResponse whose body
// mutation carries body, mirroring the rewritten request the router sends to
// Envoy after model substitution or RAG/memory injection.
func bodyMutationResponse(body string) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_Body{Body: []byte(body)},
					},
				},
			},
		},
	}
}

// immediateResponseWithBody builds an immediate ProcessingResponse carrying
// body, mirroring a cache hit, jailbreak block, or budget rejection.
func immediateResponseWithBody(body string) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Body: []byte(body),
			},
		},
	}
}

// logFormat renders r the way utils.go renders it when debug logging is on.
func logFormat(r *ext_proc.ProcessingResponse) string {
	return fmt.Sprintf("%+v", r)
}

func TestRedactResponseForLogRemovesBodyMutationContent(t *testing.T) {
	const canary = "my social security number is 123-45-6789 body-canary-001"

	resp := bodyMutationResponse(`{"messages":[{"role":"user","content":"` + canary + `"}]}`)
	dump := logFormat(redactResponseForLog(resp))

	if strings.Contains(dump, "body-canary-001") {
		t.Fatalf("request body content reached the log dump: %s", dump)
	}

	// The response actually sent to Envoy must keep the real body.
	orig := resp.GetRequestBody().GetResponse().GetBodyMutation().GetBody()
	if !strings.Contains(string(orig), "body-canary-001") {
		t.Fatalf("original body was mutated; the redaction must operate on a clone")
	}
}

func TestRedactResponseForLogRemovesImmediateResponseBody(t *testing.T) {
	const canary = "cached completion text immediate-canary-002"

	resp := immediateResponseWithBody(`{"choices":[{"message":{"content":"` + canary + `"}}]}`)
	dump := logFormat(redactResponseForLog(resp))

	if strings.Contains(dump, "immediate-canary-002") {
		t.Fatalf("immediate response body reached the log dump: %s", dump)
	}

	orig := resp.GetImmediateResponse().GetBody()
	if !strings.Contains(string(orig), "immediate-canary-002") {
		t.Fatalf("original immediate body was mutated; redaction must operate on a clone")
	}
}

// TestRedactResponseForLogKeepsBodySizeMetadata pins the positive half of the
// contract: content is removed, but the operator still gets enough to debug.
func TestRedactResponseForLogKeepsBodySizeMetadata(t *testing.T) {
	body := strings.Repeat("x", 4096)
	dump := logFormat(redactResponseForLog(bodyMutationResponse(body)))

	if !strings.Contains(dump, "4096") {
		t.Errorf("redacted dump should report the body byte count, got: %s", dump)
	}
}
