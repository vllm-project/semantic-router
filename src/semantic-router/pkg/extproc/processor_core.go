package extproc

import (
	"context"
	"errors"
	"io"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/headers"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/metrics"
)

// Process implements the ext_proc calls
func (r *OpenAIRouter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	logging.Infof("Processing at stage [init]")

	// Initialize request context once per stream
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}
	logging.Infof("Created new RequestContext at %p for new stream", ctx)

	for {
		req, err := stream.Recv()
		if err != nil {
			// Mark streaming as aborted if it was a streaming response
			// This prevents caching incomplete responses
			if ctx.IsStreamingResponse && !ctx.StreamingComplete {
				ctx.StreamingAborted = true
				logging.Infof("Streaming response aborted before completion, will not cache")
			}

			// Handle EOF - this indicates the client has closed the stream gracefully
			if errors.Is(err, io.EOF) {
				logging.Infof("Stream ended gracefully")
				return nil
			}

			// Handle gRPC status-based cancellations/timeouts
			if s, ok := status.FromError(err); ok {
				switch s.Code() {
				case codes.Canceled:
					return nil
				case codes.DeadlineExceeded:
					logging.Infof("Stream deadline exceeded")
					metrics.RecordRequestError(ctx.RequestModel, "timeout")
					return nil
				}
			}

			// Handle context cancellation from the server-side context
			if errors.Is(err, context.Canceled) {
				logging.Infof("Stream canceled gracefully")
				return nil
			}
			if errors.Is(err, context.DeadlineExceeded) {
				logging.Infof("Stream deadline exceeded")
				metrics.RecordRequestError(ctx.RequestModel, "timeout")
				return nil
			}

			logging.Errorf("Error receiving request: %v", err)
			return err
		}

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			response, err := r.handleRequestHeaders(v, ctx)
			if err != nil {
				logging.Errorf("handleRequestHeaders failed: %v", err)
				return err
			}
			if r.Config.IsAgentGateway() {
				// AgentGateway: defer header response until body EndOfStream
				ctx.DeferredHeaderResponse = response
				logging.Infof("AgentGateway mode: deferring header response until body EndOfStream")
			} else {
				if err := sendResponse(stream, response, "request header"); err != nil {
					logging.Errorf("sendResponse for headers failed: %v", err)
					return err
				}
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			if r.Config.IsAgentGateway() && !v.RequestBody.EndOfStream {
				// Accumulate intermediate body chunks, suppress response
				ctx.AccumulatedBody = append(ctx.AccumulatedBody, v.RequestBody.Body...)
				logging.Debugf("AgentGateway mode: accumulated body chunk (%d bytes, total %d bytes)",
					len(v.RequestBody.Body), len(ctx.AccumulatedBody))
				continue // Do NOT send a response for intermediate chunks
			}
			if r.Config.IsAgentGateway() && v.RequestBody.EndOfStream {
				// Final chunk: append to accumulated body and replace
				ctx.AccumulatedBody = append(ctx.AccumulatedBody, v.RequestBody.Body...)
				v.RequestBody.Body = ctx.AccumulatedBody
				logging.Infof("AgentGateway mode: final body chunk, total accumulated %d bytes", len(ctx.AccumulatedBody))
			}

			response, err := r.handleRequestBody(v, ctx)
			if err != nil {
				logging.Errorf("handleRequestBody failed: %v", err)
				return err
			}

			// AgentGateway: send deferred header response (updated with routing headers)
			// then send the body response sequentially.
			if r.Config.IsAgentGateway() && ctx.DeferredHeaderResponse != nil && !ctx.HeaderResponseSent {
				// Update routing headers in deferred header response with the final decision
				if ctx.VSRSelectedModel != "" {
					headerResp := ctx.DeferredHeaderResponse.GetRequestHeaders()
					if headerResp != nil && headerResp.Response != nil {
						if headerResp.Response.HeaderMutation == nil {
							headerResp.Response.HeaderMutation = &ext_proc.HeaderMutation{}
						}

						// Force Content-Type: application/json so upstream always parses our body mutation correctly.
						// We explicitly remove existing ones to avoid case-sensitive duplicate header issues.
						headerResp.Response.HeaderMutation.RemoveHeaders = append(headerResp.Response.HeaderMutation.RemoveHeaders, "content-type", "Content-Type")
						headerResp.Response.HeaderMutation.SetHeaders = append(headerResp.Response.HeaderMutation.SetHeaders, &core.HeaderValueOption{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						})

						// Replace the initial default model with the actual selected model
						found := false
						for _, h := range headerResp.Response.HeaderMutation.SetHeaders {
							if h.Header.Key == headers.VSRSelectedModel {
								h.Header.RawValue = []byte(ctx.VSRSelectedModel)
								found = true
								break
							}
						}
						if !found {
							headerResp.Response.HeaderMutation.SetHeaders = append(headerResp.Response.HeaderMutation.SetHeaders, &core.HeaderValueOption{
								Header: &core.HeaderValue{
									Key:      headers.VSRSelectedModel,
									RawValue: []byte(ctx.VSRSelectedModel),
								},
							})
						}
						logging.Infof("Updated deferred header response for AgentGateway: model=%s, content-type=application/json (removed old ones)", ctx.VSRSelectedModel)
					}
				}

				if err := sendResponse(stream, ctx.DeferredHeaderResponse, "deferred request header"); err != nil {
					logging.Errorf("sendResponse for deferred headers failed: %v", err)
					return err
				}
				ctx.HeaderResponseSent = true
				logging.Infof("Sent deferred header response for AgentGateway")
			}

			if err := sendResponse(stream, response, "request body"); err != nil {
				logging.Errorf("sendResponse for body failed: %v", err)
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			response, err := r.handleResponseHeaders(v, ctx)
			if err != nil {
				return err
			}
			if err := sendResponse(stream, response, "response header"); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseBody:
			// For non-streaming responses in AgentGateway mode, buffer chunks until
			// EndOfStream so handleResponseBody always receives the full JSON body.
			if !ctx.IsStreamingResponse && r.Config.IsAgentGateway() {
				ctx.AccumulatedResponseBody = append(ctx.AccumulatedResponseBody, v.ResponseBody.Body...)

				if !v.ResponseBody.EndOfStream {
					// Suppress intermediate chunk — drain it with an empty StreamedResponse.
					response := &ext_proc.ProcessingResponse{
						Response: &ext_proc.ProcessingResponse_ResponseBody{
							ResponseBody: &ext_proc.BodyResponse{
								Response: &ext_proc.CommonResponse{
									BodyMutation: &ext_proc.BodyMutation{
										Mutation: &ext_proc.BodyMutation_StreamedResponse{
											StreamedResponse: &ext_proc.StreamedBodyResponse{
												Body: []byte{}, EndOfStream: false,
											},
										},
									},
								},
							},
						},
					}
					if err := sendResponse(stream, response, "intermediate response body"); err != nil {
						return err
					}
					continue
				}

				// Final chunk — replace body with the full accumulated buffer.
				v.ResponseBody.Body = ctx.AccumulatedResponseBody
				logging.Debugf("AgentGateway: buffered response body ready, total %d bytes", len(ctx.AccumulatedResponseBody))
			}

			response, err := r.handleResponseBody(v, ctx)
			if err != nil {
				return err
			}
			if err := sendResponse(stream, response, "response body"); err != nil {
				return err
			}

		default:
			logging.Warnf("Unknown request type: %v", v)

			// For unknown message types, create a body response with CONTINUE status
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}

			if err := sendResponse(stream, response, "unknown"); err != nil {
				return err
			}
		}
	}
}
