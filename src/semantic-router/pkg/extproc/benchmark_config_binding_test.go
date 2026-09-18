package extproc

import (
	"context"
	"net/http"
	"strings"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func benchmarkHeaderRequest(expected string) *ext_proc.ProcessingRequest_RequestHeaders {
	return &ext_proc.ProcessingRequest_RequestHeaders{RequestHeaders: &ext_proc.HttpHeaders{Headers: &core.HeaderMap{Headers: []*core.HeaderValue{
		{Key: ":method", Value: "POST"}, {Key: ":path", Value: "/v1/chat/completions"}, {Key: headers.SRBenchExpectedConfigHash, Value: expected},
	}}}}
}

func TestBenchmarkConfigPreconditionRejectsBeforeGeneration(t *testing.T) {
	active := strings.Repeat("a", 64)
	for _, tc := range []struct {
		name, expected, actual string
		status                 int
	}{
		{"match", active, active, 0}, {"mismatch", strings.Repeat("b", 64), active, http.StatusPreconditionFailed},
		{"unavailable", active, "", http.StatusServiceUnavailable}, {"malformed", "bad", active, http.StatusBadRequest},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: tc.actual}}
			ctx := &RequestContext{Headers: map[string]string{}, TraceContext: context.Background()}
			response, err := router.handleRequestHeaders(benchmarkHeaderRequest(tc.expected), ctx)
			if err != nil {
				t.Fatal(err)
			}
			if tc.status == 0 {
				if response.GetRequestHeaders() == nil {
					t.Fatal("matching request did not continue")
				}
			} else if got := int(response.GetImmediateResponse().GetStatus().GetCode()); got != tc.status {
				t.Fatalf("status=%d want=%d", got, tc.status)
			}
		})
	}
}

func TestBenchmarkReceiptUsesRequestGenerationAndOverwritesProviderValue(t *testing.T) {
	hash := strings.Repeat("a", 64)
	router := &OpenAIRouter{Config: &config.RouterConfig{DocumentHash: hash}}
	ctx := &RequestContext{Headers: map[string]string{headers.SRBenchExpectedConfigHash: hash}}
	for _, immediate := range []bool{false, true} {
		mutation := &ext_proc.HeaderMutation{SetHeaders: []*core.HeaderValueOption{newHeaderValueOption(headers.VSRConfigHash, "forged")}}
		response := &ext_proc.ProcessingResponse{}
		if immediate {
			response.Response = &ext_proc.ProcessingResponse_ImmediateResponse{ImmediateResponse: &ext_proc.ImmediateResponse{Headers: mutation}}
		} else {
			response.Response = &ext_proc.ProcessingResponse_ResponseHeaders{ResponseHeaders: &ext_proc.HeadersResponse{Response: &ext_proc.CommonResponse{HeaderMutation: mutation}}}
		}
		router.bindBenchmarkConfigResponse(response, ctx)
		count := 0
		for _, option := range mutation.SetHeaders {
			if option.GetHeader().GetKey() == headers.VSRConfigHash {
				count++
				if string(option.GetHeader().GetRawValue()) != hash || option.AppendAction != core.HeaderValueOption_OVERWRITE_IF_EXISTS_OR_ADD {
					t.Fatal("receipt was not overwritten with request generation")
				}
			}
		}
		if count != 1 {
			t.Fatalf("receipt count=%d", count)
		}
	}
	response := newContinueRequestHeadersResponse()
	router.bindBenchmarkConfigResponse(response, ctx)
	removed := response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
	if len(removed) != 5 || removed[0] != headers.SRBenchExpectedConfigHash {
		t.Fatalf("binding leaked to provider: %v", removed)
	}
}
