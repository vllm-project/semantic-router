package extproc

import (
	"testing"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/require"
)

// repliedPath is the :path a reply sets on the request, and whether it sets one.
func repliedPath(t *testing.T, mutation *ext_proc.HeaderMutation) (string, bool) {
	t.Helper()
	var paths []string
	for _, header := range mutation.GetSetHeaders() {
		if header.GetHeader().GetKey() == ":path" {
			paths = append(paths, string(header.GetHeader().GetRawValue()))
		}
	}
	require.LessOrEqual(t, len(paths), 1, "one reply sets :path at most once")
	if len(paths) == 0 {
		return "", false
	}
	return paths[0], true
}

// A request under a listener's API prefix leaves the request-header phase
// with the API's own path and its query, whether the Router goes on to route
// it or the client opted out of processing. A request at the API's own path
// keeps the path the client sent.
func TestRequestHeaderReplyForwardsTheAPIPath(t *testing.T) {
	for _, tc := range []struct {
		name     string
		target   string
		skip     bool
		wantPath string
	}{
		{name: "routed", target: "/anthropic/v1/messages?beta=true", wantPath: "/v1/messages?beta=true"},
		{name: "routed_without_query", target: "/anthropic/v1/messages", wantPath: "/v1/messages"},
		{name: "skipped", target: "/anthropic/v1/messages?beta=true", skip: true, wantPath: "/v1/messages?beta=true"},
		{name: "skipped_count_tokens", target: "/anthropic/v1/messages/count_tokens?a=1&b=2", skip: true, wantPath: "/v1/messages/count_tokens?a=1&b=2"},
		{name: "routed_at_the_api_path", target: "/v1/messages?beta=true"},
		{name: "skipped_at_the_api_path", target: "/v1/messages?beta=true", skip: true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := &OpenAIRouter{Config: prefixedMessagesConfig(t)}
			router.Config.SkipProcessing.Enabled = true
			request := newSkipProcessingRequestHeaders("POST", tc.target, "false")
			if tc.skip {
				request = newSkipProcessingRequestHeaders("POST", tc.target, "true")
			}
			ctx := &RequestContext{Headers: make(map[string]string)}

			response, err := router.handleRequestHeaders(request, ctx)

			require.NoError(t, err)
			require.Equal(t, tc.skip, ctx.SkipProcessing)
			common := response.GetRequestHeaders().GetResponse()
			require.Equal(t, ext_proc.CommonResponse_CONTINUE, common.GetStatus(), "reply %v", response)
			path, set := repliedPath(t, common.GetHeaderMutation())
			require.Equal(t, tc.wantPath != "", set, "reply sets :path %q", path)
			require.Equal(t, tc.wantPath, path)
		})
	}
}

// A skipped request gets no provider path from the request-body phase, so the
// request-header reply is the only one that keeps the prefix from the upstream.
func TestSkippedRequestUnderAnAPIPrefixIsForwardedAtTheAPIPath(t *testing.T) {
	router := &OpenAIRouter{Config: prefixedMessagesConfig(t)}
	router.Config.SkipProcessing.Enabled = true
	stream := NewMockStream([]*ext_proc.ProcessingRequest{
		{Request: newSkipProcessingRequestHeaders("POST", "/anthropic/v1/messages?beta=true", "true")},
		{Request: &ext_proc.ProcessingRequest_RequestBody{RequestBody: &ext_proc.HttpBody{
			Body:        []byte(`{"model":"m","max_tokens":8,"messages":[{"role":"user","content":"hi"}]}`),
			EndOfStream: true,
		}}},
	})

	require.NoError(t, router.Process(stream))

	require.Len(t, stream.Responses, 2)
	path, set := repliedPath(t, stream.Responses[0].GetRequestHeaders().GetResponse().GetHeaderMutation())
	require.True(t, set, "header reply %v", stream.Responses[0])
	require.Equal(t, "/v1/messages?beta=true", path)
	body := stream.Responses[1].GetRequestBody().GetResponse()
	require.Equal(t, ext_proc.CommonResponse_CONTINUE, body.GetStatus(), "body reply %v", stream.Responses[1])
	_, set = repliedPath(t, body.GetHeaderMutation())
	require.False(t, set, "body reply %v", stream.Responses[1])
	require.False(t, body.GetClearRouteCache())
}
