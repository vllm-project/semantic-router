package extproc

import (
	"slices"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
)

func TestRequestHeadersStripProtectionReleaseHeader(t *testing.T) {
	for _, tc := range []struct {
		name       string
		skip       string
		header     *string
		disabled   bool
		protection bool
		want       string
	}{
		{name: "normal_path", skip: "false", protection: true, want: "x-vsr-protection-release"},
		{name: "skip_processing", skip: "true", protection: true, want: "x-vsr-protection-release"},
		{name: "release_disabled", skip: "false", protection: true, disabled: true, want: "x-vsr-protection-release"},
		{name: "protection_disabled", skip: "false", want: "x-vsr-protection-release"},
		{name: "custom_header", skip: "false", protection: true, header: stringPtr("X-My-Release"), want: "x-my-release"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			router := newRouterWithSkipProcessingGate(true)
			router.Config.RouterLearning.Enabled = tc.protection
			router.Config.RouterLearning.Protection.Release.Header = tc.header
			if tc.disabled {
				router.Config.RouterLearning.Protection.Release.Enabled = extprocBoolPtr(false)
			}
			request := newSkipProcessingRequestHeaders("POST", "/v1/chat/completions", tc.skip)
			request.RequestHeaders.Headers.Headers = append(request.RequestHeaders.Headers.Headers,
				&core.HeaderValue{Key: tc.want, Value: "1"})

			response, err := router.handleRequestHeaders(request, &RequestContext{Headers: make(map[string]string)})
			if err != nil {
				t.Fatal(err)
			}
			removed := response.GetRequestHeaders().GetResponse().GetHeaderMutation().GetRemoveHeaders()
			if !slices.Contains(removed, tc.want) {
				t.Fatalf("release header %q not removed: %v", tc.want, removed)
			}
		})
	}
}
