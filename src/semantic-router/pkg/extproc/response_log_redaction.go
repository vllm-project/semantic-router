package extproc

import (
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
)

// processingResponseLogFields deliberately exposes only transport metadata.
// ProcessingResponse can contain request/response bodies and credential-bearing
// header mutations, so it must never be formatted or marshaled for logs.
func processingResponseLogFields(
	stage string,
	response *ext_proc.ProcessingResponse,
) map[string]interface{} {
	fields := map[string]interface{}{
		"stage":         stage,
		"response_type": "none",
	}
	if response == nil {
		return fields
	}

	var common *ext_proc.CommonResponse
	switch typed := response.Response.(type) {
	case *ext_proc.ProcessingResponse_RequestHeaders:
		fields["response_type"] = "request_headers"
		common = typed.RequestHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_RequestBody:
		fields["response_type"] = "request_body"
		common = typed.RequestBody.GetResponse()
	case *ext_proc.ProcessingResponse_ResponseHeaders:
		fields["response_type"] = "response_headers"
		common = typed.ResponseHeaders.GetResponse()
	case *ext_proc.ProcessingResponse_ResponseBody:
		fields["response_type"] = "response_body"
		common = typed.ResponseBody.GetResponse()
	case *ext_proc.ProcessingResponse_ImmediateResponse:
		fields["response_type"] = "immediate_response"
		appendImmediateResponseLogFields(fields, typed.ImmediateResponse)
		return fields
	}

	appendCommonResponseLogFields(fields, common)
	return fields
}

func appendCommonResponseLogFields(fields map[string]interface{}, common *ext_proc.CommonResponse) {
	if common == nil {
		return
	}
	fields["status"] = common.GetStatus().String()
	fields["clear_route_cache"] = common.GetClearRouteCache()
	appendHeaderMutationLogFields(fields, common.GetHeaderMutation())
	if trailers := common.GetTrailers(); trailers != nil {
		fields["trailer_count"] = len(trailers.GetHeaders())
	}
	appendBodyMutationLogFields(fields, common.GetBodyMutation())
}

func appendImmediateResponseLogFields(
	fields map[string]interface{},
	immediate *ext_proc.ImmediateResponse,
) {
	if immediate == nil {
		return
	}
	fields["body_bytes"] = len(immediate.GetBody())
	if status := immediate.GetStatus(); status != nil {
		fields["http_status"] = status.GetCode().String()
	}
	if grpcStatus := immediate.GetGrpcStatus(); grpcStatus != nil {
		fields["grpc_status"] = grpcStatus.GetStatus()
	}
	appendHeaderMutationLogFields(fields, immediate.GetHeaders())
}

func appendHeaderMutationLogFields(fields map[string]interface{}, mutation *ext_proc.HeaderMutation) {
	if mutation == nil {
		return
	}
	fields["set_header_count"] = len(mutation.GetSetHeaders())
	fields["remove_header_count"] = len(mutation.GetRemoveHeaders())
}

func appendBodyMutationLogFields(fields map[string]interface{}, mutation *ext_proc.BodyMutation) {
	if mutation == nil {
		return
	}
	switch typed := mutation.Mutation.(type) {
	case *ext_proc.BodyMutation_Body:
		fields["body_mutation"] = "replace"
		fields["body_bytes"] = len(typed.Body)
	case *ext_proc.BodyMutation_ClearBody:
		fields["body_mutation"] = "clear"
		fields["body_bytes"] = 0
	case *ext_proc.BodyMutation_StreamedResponse:
		fields["body_mutation"] = "streamed_response"
		fields["body_bytes"] = len(typed.StreamedResponse.GetBody())
		fields["end_of_stream"] = typed.StreamedResponse.GetEndOfStream()
	}
}
