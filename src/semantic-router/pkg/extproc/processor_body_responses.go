package extproc

import ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

func newContinueRequestBodyResponse() *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{Status: ext_proc.CommonResponse_CONTINUE},
			},
		},
	}
}

func newFullDuplexRequestBodyResponse(body []byte, endOfStream bool) *ext_proc.ProcessingResponse {
	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestBody{
			RequestBody: &ext_proc.BodyResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
					BodyMutation: &ext_proc.BodyMutation{
						Mutation: &ext_proc.BodyMutation_StreamedResponse{
							StreamedResponse: &ext_proc.StreamedBodyResponse{
								Body: body, EndOfStream: endOfStream,
							},
						},
					},
				},
			},
		},
	}
}
