package extproc

import (
	"testing"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestFullDuplex_FinalResponseMovesHeaderMutationToHold(t *testing.T) {
	hold := &fullDuplexHeaderHold{}
	h := &StreamedBodyHandler{ctx: &RequestContext{FullDuplexRequestBody: true, fullDuplexHold: hold}}
	h.buf.WriteString(fullDuplexTestBody)
	route := &ext_proc.HeaderMutation{RemoveHeaders: []string{"content-length"}}
	response := &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestBody{
		RequestBody: &ext_proc.BodyResponse{Response: &ext_proc.CommonResponse{
			Status: ext_proc.CommonResponse_CONTINUE, HeaderMutation: route, ClearRouteCache: true,
		}},
	}}

	common := h.finalizeResponse(response).GetRequestBody().GetResponse()
	assert.Nil(t, common.GetHeaderMutation())
	assert.False(t, common.GetClearRouteCache())
	assert.Same(t, route, hold.routeMutation)
	assert.True(t, hold.clearRouteCache)
}

func TestFullDuplex_FlushCarriesTheBodyStageRouteCacheClear(t *testing.T) {
	ctx := &RequestContext{fullDuplexHold: &fullDuplexHeaderHold{
		response: &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{},
		}},
		routeMutation:   &ext_proc.HeaderMutation{RemoveHeaders: []string{"authorization"}},
		clearRouteCache: true,
	}}
	stream := NewMockStream(nil)

	require.NoError(t, flushHeldRequestHeaderReply(stream, newContinueRequestBodyResponse(), ctx))

	require.Len(t, stream.Responses, 1)
	common := stream.Responses[0].GetRequestHeaders().GetResponse()
	assert.True(t, common.GetClearRouteCache(), "only the body stage asked to clear the route cache")
	assert.Equal(t, []string{"authorization"}, common.GetHeaderMutation().GetRemoveHeaders())
	assert.Nil(t, ctx.fullDuplexHold)
}

func TestFullDuplex_BodyEndingWithoutReplyIsAnError(t *testing.T) {
	ctx := &RequestContext{FullDuplexRequestBody: true, fullDuplexHold: &fullDuplexHeaderHold{
		response: &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{},
		}},
	}}
	stream := NewMockStream(nil)

	_, err := fullDuplexRoutingRouter().sendRequestBodyResult(stream, nil, nil, ctx)

	require.Error(t, err, "a held header reply must not be left waiting")
	assert.Empty(t, stream.Responses)
}

func TestSequenceHeaderMutations(t *testing.T) {
	set := func(key, value string) *core.HeaderValueOption {
		return &core.HeaderValueOption{Header: &core.HeaderValue{Key: key, RawValue: []byte(value)}}
	}
	earlier := &ext_proc.HeaderMutation{
		SetHeaders:    []*core.HeaderValueOption{set("x-kept", "a"), set("X-Dropped", "b"), set("x-overwritten", "c")},
		RemoveHeaders: []string{"x-early-remove"},
	}
	later := &ext_proc.HeaderMutation{
		SetHeaders:    []*core.HeaderValueOption{set("x-overwritten", "d"), set("x-early-remove", "e")},
		RemoveHeaders: []string{"x-dropped"},
	}

	merged := sequenceHeaderMutations(earlier, later)

	var keys []string
	for _, header := range merged.GetSetHeaders() {
		keys = append(keys, header.GetHeader().GetKey()+"="+string(header.GetHeader().GetRawValue()))
	}
	assert.Equal(t, []string{"x-kept=a", "x-overwritten=c", "x-overwritten=d", "x-early-remove=e"}, keys,
		"a later remove drops an earlier set; later sets follow earlier ones so they win")
	assert.Equal(t, []string{"x-early-remove", "x-dropped"}, merged.GetRemoveHeaders())
}

func TestFullDuplex_HeldHeaderReplyBeforeErrorRecoversSendPanic(t *testing.T) {
	ctx := &RequestContext{fullDuplexHold: &fullDuplexHeaderHold{
		response: &ext_proc.ProcessingResponse{Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{},
		}},
	}}
	stream := &panicOnSendStream{MockStream: *NewMockStream(nil), panicMsg: "send failed"}

	assert.NotPanics(t, func() { sendHeldHeaderReplyBeforeError(stream, ctx) })
	assert.Nil(t, ctx.fullDuplexHold)
}
