package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

func (r *OpenAIRouter) validationResponseFromRequestError(err error) *ext_proc.ProcessingResponse {
	if err == nil {
		return nil
	}

	st, ok := status.FromError(err)
	if !ok || st.Code() != codes.InvalidArgument {
		return nil
	}

	return r.createErrorResponse(400, st.Message())
}
