package extproc

import (
	"strings"
	"testing"

	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
)

func TestPublicListenerFailsClosedForRouterReplayBeforeSkipProcessing(t *testing.T) {
	enabledRouter, recordID := newReplayAPITestRouter(t)
	tests := []struct {
		name          string
		router        *OpenAIRouter
		requestTarget string
		skip          bool
	}{
		{name: "disabled list", router: &OpenAIRouter{}, requestTarget: "/v1/router_replay"},
		{name: "enabled detail", router: enabledRouter, requestTarget: "/v1/router_replay/" + recordID},
		{name: "aggregate with skip opt-out", router: newRouterWithSkipProcessingGate(true), requestTarget: "/v1/router_replay/aggregate", skip: true},
		{name: "reserved prefix variant", router: &OpenAIRouter{}, requestTarget: "/v1/router_replay-export"},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := newRequestHeaders("GET", test.requestTarget)
			if test.skip {
				request = newSkipProcessingRequestHeaders("GET", test.requestTarget, "true")
			}
			ctx := &RequestContext{Headers: make(map[string]string)}
			response, err := test.router.handleRequestHeaders(request, ctx)
			if err != nil {
				t.Fatalf("handleRequestHeaders() error = %v", err)
			}
			immediate := response.GetImmediateResponse()
			if immediate == nil || immediate.Status.GetCode() != typev3.StatusCode_NotFound {
				t.Fatalf("public replay response = %#v, want immediate 404", response)
			}
			if !strings.Contains(string(immediate.Body), "endpoint not found") {
				t.Fatalf("public replay body = %s", immediate.Body)
			}
			if test.skip && !ctx.SkipProcessing {
				t.Fatal("skip-processing header was not captured before replay denial")
			}
		})
	}
}

func TestManagementReplayAdapterPreservesExistingReplayResponse(t *testing.T) {
	router, recordID := newReplayAPITestRouter(t)

	response, handled := router.HandleReplayRequest("GET", "/v1/router_replay/"+recordID)
	if !handled {
		t.Fatal("management replay request was not handled")
	}
	if response.StatusCode != 200 {
		t.Fatalf("management replay status = %d, want 200", response.StatusCode)
	}
	if !strings.Contains(string(response.Body), recordID) {
		t.Fatalf("management replay body does not contain %q: %s", recordID, response.Body)
	}
}
