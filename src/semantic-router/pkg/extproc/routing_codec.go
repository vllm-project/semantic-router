package extproc

import (
	"encoding/json"
	"fmt"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	http_ext "github.com/envoyproxy/go-control-plane/envoy/extensions/filters/http/ext_proc/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// The pipeline still builds ext_proc messages as its internal representation.
// This codec is the boundary between them and the transport-agnostic routing
// contract. Decoding reads exactly the fields Envoy applies (for example a set
// header's raw_value, never its value) and fails closed on any field the
// contract does not carry, so no ext_proc behavior is silently dropped.

func extprocHeaderMap(header routing.Header) *core.HeaderMap {
	out := &core.HeaderMap{Headers: make([]*core.HeaderValue, 0, len(header))}
	for _, field := range header {
		out.Headers = append(out.Headers, &core.HeaderValue{Key: field.Name, RawValue: []byte(field.Value)})
	}
	return out
}

// routingEffect decodes the reply to one phase.
func routingEffect(response *ext_proc.ProcessingResponse) (*routing.Effect, error) {
	if response == nil {
		// sendResponse answers a nil reply with a plain CONTINUE.
		return &routing.Effect{}, nil
	}
	if response.GetDynamicMetadata() != nil || response.GetOverrideMessageTimeout() != nil {
		return nil, unsupportedReply("dynamic metadata or a message timeout override")
	}
	var common *ext_proc.CommonResponse
	switch reply := response.GetResponse().(type) {
	case *ext_proc.ProcessingResponse_ImmediateResponse:
		immediate, err := routingImmediate(reply.ImmediateResponse)
		if err != nil {
			return nil, err
		}
		return &routing.Effect{Immediate: immediate}, nil
	case *ext_proc.ProcessingResponse_RequestHeaders:
		common = reply.RequestHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_ResponseHeaders:
		common = reply.ResponseHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_RequestBody:
		common = reply.RequestBody.GetResponse()
	case *ext_proc.ProcessingResponse_ResponseBody:
		common = reply.ResponseBody.GetResponse()
	default:
		return nil, unsupportedReply(fmt.Sprintf("reply %T", reply))
	}
	effect, err := routingCommonEffect(common)
	if err != nil {
		return nil, err
	}
	if mode := response.GetModeOverride(); mode != nil {
		if effect.ResponseBodyMode, err = routingBodyMode(mode); err != nil {
			return nil, err
		}
	}
	return effect, nil
}

func routingCommonEffect(common *ext_proc.CommonResponse) (*routing.Effect, error) {
	effect := &routing.Effect{}
	if common == nil {
		return effect, nil
	}
	if common.GetStatus() != ext_proc.CommonResponse_CONTINUE {
		return nil, unsupportedReply("status " + common.GetStatus().String())
	}
	if common.GetTrailers() != nil {
		return nil, unsupportedReply("trailer mutations")
	}
	effect.Header = routingHeaderMutation(common.GetHeaderMutation())
	effect.ClearRouteCache = common.GetClearRouteCache()
	if mutation := common.GetBodyMutation(); mutation != nil {
		switch body := mutation.GetMutation().(type) {
		case *ext_proc.BodyMutation_Body:
			effect.Body = &routing.BodyMutation{Body: body.Body}
		case *ext_proc.BodyMutation_ClearBody:
			if !body.ClearBody {
				return nil, unsupportedReply("clear_body set to false")
			}
			effect.Body = &routing.BodyMutation{Clear: true}
		default:
			return nil, unsupportedReply(fmt.Sprintf("body mutation %T", body))
		}
	}
	return effect, nil
}

func routingHeaderMutation(mutation *ext_proc.HeaderMutation) *routing.HeaderMutation {
	if mutation == nil {
		return nil
	}
	out := &routing.HeaderMutation{Remove: append([]string(nil), mutation.GetRemoveHeaders()...)}
	for _, option := range mutation.GetSetHeaders() {
		if option.GetHeader() == nil {
			continue
		}
		out.Set = append(out.Set, routing.HeaderOption{
			Name:   option.GetHeader().GetKey(),
			Value:  string(option.GetHeader().GetRawValue()),
			Append: option.GetAppend().GetValue(),
		})
	}
	return out
}

func routingImmediate(response *ext_proc.ImmediateResponse) (*routing.ImmediateResponse, error) {
	if response.GetGrpcStatus() != nil {
		return nil, unsupportedReply("an immediate gRPC status")
	}
	return &routing.ImmediateResponse{
		Status:  int(response.GetStatus().GetCode()),
		Header:  routingHeaderMutation(response.GetHeaders()),
		Body:    response.GetBody(),
		Details: response.GetDetails(),
	}, nil
}

func routingBodyMode(mode *http_ext.ProcessingMode) (routing.BodyMode, error) {
	if mode.GetRequestHeaderMode() != http_ext.ProcessingMode_DEFAULT ||
		mode.GetResponseHeaderMode() != http_ext.ProcessingMode_DEFAULT ||
		mode.GetRequestTrailerMode() != http_ext.ProcessingMode_DEFAULT ||
		mode.GetResponseTrailerMode() != http_ext.ProcessingMode_DEFAULT {
		return "", unsupportedReply("a header or trailer mode override")
	}
	switch mode.GetResponseBodyMode() {
	case http_ext.ProcessingMode_STREAMED:
		return routing.BodyModeStreamed, nil
	case http_ext.ProcessingMode_BUFFERED:
		return routing.BodyModeBuffered, nil
	default:
		return "", unsupportedReply("response body mode " + mode.GetResponseBodyMode().String())
	}
}

func unsupportedReply(what string) error {
	return fmt.Errorf("%w: %s", routing.ErrUnsupported, what)
}

// routingEvidence reports the routing facts the request context holds.
func routingEvidence(ctx *RequestContext) routing.Evidence {
	evidence := routing.Evidence{
		Recipe:          string(ctx.Routing.RecipeName()),
		Decision:        ctx.VSRSelectedDecisionName,
		Confidence:      ctx.VSRSelectedDecisionConfidence,
		Category:        ctx.VSRSelectedCategory,
		Model:           ctx.VSRSelectedModel,
		SelectionMethod: ctx.VSRSelectionMethod,
		ReasoningMode:   ctx.VSRReasoningMode,
		CacheHit:        ctx.VSRCacheHit,
	}
	if encoded, err := json.Marshal(replaySignalState(ctx)); err == nil {
		var signals map[string][]string
		if json.Unmarshal(encoded, &signals) == nil && len(signals) > 0 {
			evidence.Signals = signals
		}
	}
	return evidence
}
