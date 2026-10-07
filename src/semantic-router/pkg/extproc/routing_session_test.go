package extproc

import (
	"context"
	"errors"
	"testing"
	"time"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/protobuf/types/known/wrapperspb"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

func TestRoutingSessionHoldsTheGenerationUntilClose(t *testing.T) {
	closed := make(chan struct{})
	resources := newResourceScope()
	resources.add(func() error {
		close(closed)
		return nil
	})
	first := (&routerComponents{resources: resources}).buildRouter()
	service := NewRouterService(first)

	session, err := service.Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if err := service.Swap((&routerComponents{resources: newResourceScope()}).buildRouter(), nil); err != nil {
		t.Fatal(err)
	}
	select {
	case <-closed:
		t.Fatal("a reload closed the router while a routing session still used it")
	case <-time.After(50 * time.Millisecond):
	}
	session.Close(nil)
	session.Close(errors.New("second close is ignored"))
	select {
	case <-closed:
	case <-time.After(5 * time.Second):
		t.Fatal("the retired router was not closed after its last session ended")
	}
}

func TestRoutingSessionOpenFailsAfterShutdown(t *testing.T) {
	service := NewRouterService((&routerComponents{resources: newResourceScope()}).buildRouter())
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := service.Shutdown(ctx); err != nil {
		t.Fatal(err)
	}
	if _, err := service.Open(context.Background()); err == nil {
		t.Fatal("Open must fail once the service is shutting down")
	}
}

func TestRoutingSessionEndsOnAPanickingPhase(t *testing.T) {
	session := (&OpenAIRouter{}).newRoutingSession(context.Background(), nil)
	effect, err := session.phase(func() (*ext_proc.ProcessingResponse, error) {
		panic("boom")
	}, func(*ext_proc.ProcessingResponse) {})
	if effect != nil || err == nil {
		t.Fatalf("a panicking phase must fail, got %v %v", effect, err)
	}
	if _, err := session.ResponseBody(nil, true); !errors.Is(err, errSessionEnded) {
		t.Fatalf("phases after a failure must report the ended session, got %v", err)
	}
	session.Close(nil)
}

func TestRoutingSessionAnswersAnUnroutableRequestImmediately(t *testing.T) {
	session, err := (&OpenAIRouter{}).Open(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	defer session.Close(nil)
	effect, err := session.RequestHeaders(routing.Header{
		{Name: ":method", Value: "GET"},
		{Name: ":path", Value: "/v1/router-replay"},
	}, true)
	if err != nil {
		t.Fatal(err)
	}
	if effect.Immediate == nil || effect.Immediate.Status != 404 {
		t.Fatalf("the replay API must stay off the inference path, got %+v", effect)
	}
}

func TestRoutingEffectDecodesWhatEnvoyApplies(t *testing.T) {
	response := &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ResponseHeaders{ResponseHeaders: &ext_proc.HeadersResponse{
			Response: &ext_proc.CommonResponse{
				HeaderMutation: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{Header: &core.HeaderValue{Key: "x-raw", RawValue: []byte("r")}},
						{Header: &core.HeaderValue{Key: "x-value-only", Value: "ignored by Envoy"}},
						{Header: &core.HeaderValue{Key: "x-append", RawValue: []byte("a")}, Append: wrapperspb.Bool(true)},
						{AppendAction: core.HeaderValueOption_ADD_IF_ABSENT},
					},
					RemoveHeaders: []string{"content-length"},
				},
				ClearRouteCache: true,
			},
		}},
		ModeOverride: &http_ext.ProcessingMode{ResponseBodyMode: http_ext.ProcessingMode_STREAMED},
	}
	effect, err := routingEffect(response)
	if err != nil {
		t.Fatal(err)
	}
	want := []routing.HeaderOption{{Name: "x-raw", Value: "r"}, {Name: "x-value-only"}, {Name: "x-append", Value: "a", Append: true}}
	if len(effect.Header.Set) != len(want) {
		t.Fatalf("sets = %+v", effect.Header.Set)
	}
	for i := range want {
		if effect.Header.Set[i] != want[i] {
			t.Fatalf("set %d = %+v, want %+v", i, effect.Header.Set[i], want[i])
		}
	}
	if !effect.ClearRouteCache || effect.ResponseBodyMode != routing.BodyModeStreamed || effect.Header.Remove[0] != "content-length" {
		t.Fatalf("effect = %+v", effect)
	}

	immediate, err := routingEffect(&ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_ImmediateResponse{
		ImmediateResponse: &ext_proc.ImmediateResponse{Status: &typev3.HttpStatus{Code: typev3.StatusCode_TooManyRequests}, Body: []byte("slow down"), Details: "rate"},
	}})
	if err != nil || immediate.Immediate.Status != 429 || string(immediate.Immediate.Body) != "slow down" || immediate.Immediate.Details != "rate" {
		t.Fatalf("immediate = %+v, %v", immediate, err)
	}
}

func TestRoutingEffectFailsClosedOnFieldsTheContractLacks(t *testing.T) {
	body := func(common *ext_proc.CommonResponse) *ext_proc.ProcessingResponse {
		return &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{RequestBody: &ext_proc.BodyResponse{Response: common}}}
	}
	tests := map[string]*ext_proc.ProcessingResponse{
		"continue and replace": body(&ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE_AND_REPLACE}),
		"trailers":             body(&ext_proc.CommonResponse{Trailers: &core.HeaderMap{}}),
		"streamed response": body(&ext_proc.CommonResponse{BodyMutation: &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_StreamedResponse{StreamedResponse: &ext_proc.StreamedBodyResponse{}},
		}}),
		"clear body false": body(&ext_proc.CommonResponse{BodyMutation: &ext_proc.BodyMutation{
			Mutation: &ext_proc.BodyMutation_ClearBody{ClearBody: false},
		}}),
		"partial mode": {
			Response:     &ext_proc.ProcessingResponse_ResponseHeaders{ResponseHeaders: &ext_proc.HeadersResponse{}},
			ModeOverride: &http_ext.ProcessingMode{ResponseBodyMode: http_ext.ProcessingMode_BUFFERED_PARTIAL},
		},
		"immediate grpc status": {Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{GrpcStatus: &ext_proc.GrpcStatus{Status: 7}},
		}},
		"trailers reply": {Response: &ext_proc.ProcessingResponse_RequestTrailers{RequestTrailers: &ext_proc.TrailersResponse{}}},
	}
	for name, response := range tests {
		t.Run(name, func(t *testing.T) {
			if _, err := routingEffect(response); !errors.Is(err, routing.ErrUnsupported) {
				t.Fatalf("err = %v, want ErrUnsupported", err)
			}
		})
	}
	if effect, err := routingEffect(nil); err != nil || effect == nil || effect.Header != nil {
		t.Fatalf("a nil reply is a plain continue, got %+v %v", effect, err)
	}
}
