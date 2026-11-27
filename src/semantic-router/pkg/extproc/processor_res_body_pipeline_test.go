package extproc

import (
	"bytes"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

func TestSetResponseBodyMutationRemovesStaleContentLength(t *testing.T) {
	response := buildResponseBodyContinueResponse(nil, nil)

	setResponseBodyMutation(response, []byte(`{"object":"response"}`))

	common := response.GetResponseBody().GetResponse()
	if got := string(common.GetBodyMutation().GetBody()); got != `{"object":"response"}` {
		t.Fatalf("body mutation = %q", got)
	}
	if !containsStringForTest(common.GetHeaderMutation().GetRemoveHeaders(), "content-length") {
		t.Fatalf("content-length was not removed: %#v", common.GetHeaderMutation())
	}
}

func TestSetResponseBodyMutationPreservesExistingHeaderMutation(t *testing.T) {
	response := buildResponseBodyContinueResponse(nil, &ext_proc.HeaderMutation{
		RemoveHeaders: []string{"x-obsolete"},
	})

	setResponseBodyMutation(response, []byte(`{"object":"response"}`))

	removed := response.GetResponseBody().GetResponse().GetHeaderMutation().GetRemoveHeaders()
	if !containsStringForTest(removed, "x-obsolete") || !containsStringForTest(removed, "content-length") {
		t.Fatalf("header removals = %#v", removed)
	}
}

func TestHandleNonStreamingResponseBodyMutatesWhenBodyReplaced(t *testing.T) {
	body := extProcResponseFixture(llmprotocol.OpenAIChatV1)
	sameFormatChat := func() *RequestContext {
		return &RequestContext{
			SourceFormat: llmprotocol.OpenAIChatV1,
			TargetFormat: llmprotocol.OpenAIChatV1,
			TraceContext: t.Context(),
		}
	}

	unchanged := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, sameFormatChat(), 0, false)
	if mutation := unchanged.GetResponseBody().GetResponse().GetBodyMutation(); mutation != nil {
		t.Fatalf("same-format Chat forwarded a body mutation: %#v", mutation)
	}

	replaced := (&OpenAIRouter{}).handleNonStreamingResponseBody(body, sameFormatChat(), 0, true)
	common := replaced.GetResponseBody().GetResponse()
	if got := common.GetBodyMutation().GetBody(); !bytes.Equal(got, body) {
		t.Fatalf("replaced body mutation = %q, want original client body %q", got, body)
	}
	if !containsStringForTest(common.GetHeaderMutation().GetRemoveHeaders(), "content-length") {
		t.Fatalf("content-length was not removed: %#v", common.GetHeaderMutation())
	}
}

func TestSetResponseContentTypeOverwritesExistingValue(t *testing.T) {
	response := buildResponseBodyContinueResponse(nil, &ext_proc.HeaderMutation{
		SetHeaders: []*core.HeaderValueOption{{
			Header: &core.HeaderValue{Key: "content-type", Value: "text/plain"},
		}},
	})

	setResponseContentType(response, "application/json")

	option := response.GetResponseBody().GetResponse().GetHeaderMutation().GetSetHeaders()[0]
	if !bytes.Equal(option.GetHeader().GetRawValue(), []byte("application/json")) ||
		option.GetAppendAction() != core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
		t.Fatalf("content-type option = %#v", option)
	}
}
