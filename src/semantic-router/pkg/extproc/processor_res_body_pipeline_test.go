package extproc

import (
	"bytes"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
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
