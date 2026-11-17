package extproc

import (
	"crypto/sha256"
	"errors"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

const preparedDispatchReceiptVersion = 1

// preparedProviderRequest is the immutable result of provider projection. It
// owns the exact bytes that admission will observe and dispatch returns to Envoy.
// Metadata is deliberately small and additive so later accounting revisions,
// context limits, and admission evidence can extend this contract without
// changing the prepare -> admit -> dispatch lifecycle.
type preparedProviderRequest struct {
	body     []byte
	metadata preparedProviderRequestMetadata
}

type preparedProviderRequestMetadata struct {
	wireFormat llmprotocol.WireFormat
}

// newPreparedProviderRequest transfers ownership of body to the returned
// artifact. The codec and provider adapter both return fresh slices, so callers
// must not retain or mutate body after this boundary.
func newPreparedProviderRequest(body []byte, metadata preparedProviderRequestMetadata) *preparedProviderRequest {
	return &preparedProviderRequest{body: body, metadata: metadata}
}

func (request *preparedProviderRequest) byteLength() int {
	return len(request.body)
}

func (request *preparedProviderRequest) payloadSHA256() string {
	digest := sha256.Sum256(request.body)
	return fmt.Sprintf("%x", digest)
}

// materializePreparedProviderRequest freezes the final provider-bound payload
// after semantic plugins, protocol projection, and provider adaptation. It is
// intentionally behavior-neutral: admission will consume this artifact in a
// later slice, while dispatch already consumes the exact same artifact today.
func (r *OpenAIRouter) materializePreparedProviderRequest(
	dispatch *providerDispatch,
	ctx *RequestContext,
) (*preparedProviderRequest, error) {
	body, err := r.encodeDispatchRequest(ctx)
	if err != nil {
		var protocolError *llmprotocol.ProtocolError
		if errors.As(err, &protocolError) &&
			protocolError.Code == promptCacheErrorTargetUnsupported {
			if ctx != nil {
				ctx.ImmediateProtocolError = protocolError
			}
			return nil, err
		}
		metrics.RecordRequestError(dispatch.logicalModel, "serialization_error")
		return nil, dispatchWireError(err, ctx, "encode provider request")
	}
	body, err = r.adaptProviderRequest(body, dispatch, ctx)
	if err != nil {
		metrics.RecordRequestError(dispatch.logicalModel, "provider_adapter_error")
		return nil, dispatchWireError(err, ctx, "adapt provider request")
	}
	return newPreparedProviderRequest(body, preparedProviderRequestMetadata{
		wireFormat: dispatch.targetFormat,
	}), nil
}
