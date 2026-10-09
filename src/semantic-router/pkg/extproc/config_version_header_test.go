package extproc

import (
	"context"
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
)

func headerValue(options []*core.HeaderValueOption, name string) (string, bool) {
	for _, option := range options {
		if option.GetHeader().GetKey() == name {
			return string(option.GetHeader().GetRawValue()), true
		}
	}
	return "", false
}

func TestRoutedResponsesNameTheServingConfigVersion(t *testing.T) {
	mutation := buildResponseHeaderMutation(&RequestContext{ConfigVersion: 7}, true)
	if got, ok := headerValue(mutation.GetSetHeaders(), headers.VSRConfigVersion); !ok || got != "7" {
		t.Fatalf("%s = %q, %v", headers.VSRConfigVersion, got, ok)
	}
	failed := buildResponseHeaderMutation(&RequestContext{ConfigVersion: 7}, false)
	if got, ok := headerValue(failed.GetSetHeaders(), headers.VSRConfigVersion); !ok || got != "7" {
		t.Fatalf("an upstream error lacks %s: %q, %v", headers.VSRConfigVersion, got, ok)
	}
	unversioned := buildResponseHeaderMutation(&RequestContext{}, true)
	if _, ok := headerValue(unversioned.GetSetHeaders(), headers.VSRConfigVersion); ok {
		t.Fatal("a router without a snapshot named a version")
	}
}

func TestRouterAnswersNameTheServingConfigVersion(t *testing.T) {
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ImmediateResponse{
		ImmediateResponse: &ext_proc.ImmediateResponse{Status: &typev3.HttpStatus{Code: typev3.StatusCode_Forbidden}},
	}}
	appendConfigVersionToImmediateResponse(response, &RequestContext{ConfigVersion: 3})
	if got, ok := headerValue(response.GetImmediateResponse().GetHeaders().GetSetHeaders(), headers.VSRConfigVersion); !ok || got != "3" {
		t.Fatalf("%s = %q, %v", headers.VSRConfigVersion, got, ok)
	}
	continued := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestHeaders{
		RequestHeaders: &ext_proc.HeadersResponse{},
	}}
	appendConfigVersionToImmediateResponse(continued, &RequestContext{ConfigVersion: 3})
	if continued.GetImmediateResponse() != nil || continued.GetRequestHeaders().GetResponse() != nil {
		t.Fatal("a request that continues upstream carried the version to the backend")
	}
}

// A request reads the version of the generation that opened it, for its
// whole life, even when a reload swaps another generation in meanwhile.
func TestRequestsKeepTheVersionTheyStartedOn(t *testing.T) {
	manager := configsnapshot.NewManager(configsnapshot.Options{})
	first, err := manager.Install(context.Background(), configsnapshot.Update{
		Origin: configsnapshot.Origin{Source: configsnapshot.SourceStartup}, Config: &config.RouterConfig{},
	})
	if err != nil {
		t.Fatal(err)
	}
	router := &OpenAIRouter{Config: &config.RouterConfig{}, resources: newResourceScope()}
	service := NewRouterServiceForSnapshot(router, first)
	t.Cleanup(func() { _ = service.Close() })
	session, err := service.Open(t.Context())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(nil)
	if err := service.Swap(&OpenAIRouter{Config: &config.RouterConfig{}, resources: newResourceScope()}, nil); err != nil {
		t.Fatal(err)
	}
	if got := session.(*routingSession).ctx.ConfigVersion; got != first.Version() {
		t.Fatalf("session version = %d, want %d", got, first.Version())
	}
}
