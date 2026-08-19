package extproc

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

func TestContextCompressionFakeUpstreamReceivesCompressedBody(t *testing.T) {
	largeTool := strings.Repeat("irrelevant inventory ", 300) +
		"authentication validator failed " +
		strings.Repeat("irrelevant billing ", 300)
	original := []byte(`{"model":"auto","messages":[` +
		`{"role":"user","content":"fix authentication validator"},` +
		`{"role":"tool","tool_call_id":"call_1","content":` + mustJSONString(t, largeTool) + `}]}`)
	ctx := &RequestContext{
		OriginalRequestBody: original,
		VSRSelectedDecision: contextCompressionTestDecision(false),
	}
	router := &OpenAIRouter{}
	working := router.applyContextCompression(ctx, original)
	ctx.setWorkingRequestBody(working)

	response := attachWorkingBodyMutation(&ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE},
			},
		},
	}, ctx)
	outbound := response.GetRequestBody().GetResponse().GetBodyMutation().GetBody()

	upstream, captured := newCompressionCaptureServer(t)
	defer upstream.Close()
	postCompressionBody(t, upstream, outbound)
	assertCompressionCapture(t, *captured, original, working, ctx)
}

func newCompressionCaptureServer(t *testing.T) (*httptest.Server, *[]byte) {
	t.Helper()
	var captured []byte
	server := httptest.NewServer(http.HandlerFunc(func(writer http.ResponseWriter, request *http.Request) {
		defer request.Body.Close()
		var err error
		captured, err = io.ReadAll(request.Body)
		if err != nil {
			t.Errorf("read upstream request: %v", err)
			writer.WriteHeader(http.StatusInternalServerError)
			return
		}
		if !json.Valid(captured) {
			t.Errorf("upstream received invalid JSON: %s", captured)
			writer.WriteHeader(http.StatusBadRequest)
			return
		}
		writer.WriteHeader(http.StatusOK)
	}))
	return server, &captured
}

func postCompressionBody(t *testing.T, upstream *httptest.Server, outbound []byte) {
	t.Helper()
	httpResponse, err := upstream.Client().Post(
		upstream.URL,
		"application/json",
		bytes.NewReader(outbound),
	)
	if err != nil {
		t.Fatalf("post to fake upstream: %v", err)
	}
	httpResponse.Body.Close()
	if httpResponse.StatusCode != http.StatusOK {
		t.Fatalf("fake upstream rejected request: %s", httpResponse.Status)
	}
}

func assertCompressionCapture(
	t *testing.T,
	captured []byte,
	original []byte,
	working []byte,
	ctx *RequestContext,
) {
	t.Helper()
	if !bytes.Equal(captured, working) {
		t.Fatal("fake upstream did not receive the working request body")
	}
	if bytes.Equal(captured, original) || len(captured) >= len(original) {
		t.Fatal("fake upstream received an uncompressed request")
	}
	if !strings.Contains(string(captured), "authentication validator") {
		t.Fatal("fake upstream body lost query-relevant content")
	}
	if !ctx.ContextCompressionApplied ||
		ctx.ContextCompressionBefore <= ctx.ContextCompressionAfter ||
		ctx.ContextCompressionOmitted == 0 {
		t.Fatalf("compression diagnostics do not match emitted body: %#v", ctx)
	}
	diagnostics := buildReplayRouteDiagnostics(ctx, "auto", "test", "route", 1, 10)
	if !diagnostics.ContextCompressionApplied ||
		diagnostics.ContextCompressionFormat != ctx.ContextCompressionFormat ||
		diagnostics.ContextCompressionOmitted != ctx.ContextCompressionOmitted {
		t.Fatalf("replay diagnostics do not match emitted body: %#v", diagnostics)
	}
}
