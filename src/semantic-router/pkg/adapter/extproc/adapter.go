package extproc

import (
	"context"
	"encoding/json"
	"fmt"
	"net"

	core "github.com/envoyproxy/go-control-plane/envoy/config/core/v3"
	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	typev3 "github.com/envoyproxy/go-control-plane/envoy/type/v3"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/router/engine"
)

type Adapter struct {
	engine   *engine.RouterEngine
	server   *grpc.Server
	port     int
	secure   bool
	certPath string
}

// RequestContext tracks request state across ExtProc phases
type RequestContext struct {
	Headers   map[string]string
	RequestID string
}

func NewAdapter(eng *engine.RouterEngine, configPath string, port int, tlsConfig *config.TLSConfig) (*Adapter, error) {
	secure := false
	certPath := ""
	if tlsConfig != nil && tlsConfig.Enabled {
		secure = true
		certPath = tlsConfig.CertFile
		logging.Infof("ExtProc adapter TLS enabled with cert: %s", certPath)
	}

	return &Adapter{
		engine:   eng,
		port:     port,
		secure:   secure,
		certPath: certPath,
	}, nil
}

func (a *Adapter) Start() error {
	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", a.port))
	if err != nil {
		return fmt.Errorf("failed to listen on port %d: %w", a.port, err)
	}

	var opts []grpc.ServerOption

	if a.secure {
		logging.Infof("Starting ExtProc adapter on port %d (TLS enabled)", a.port)
		creds, err := credentials.NewServerTLSFromFile(a.certPath, a.certPath)
		if err != nil {
			return fmt.Errorf("failed to load TLS credentials: %w", err)
		}
		opts = append(opts, grpc.Creds(creds))
	} else {
		logging.Infof("Starting ExtProc adapter on port %d", a.port)
	}

	a.server = grpc.NewServer(opts...)
	ext_proc.RegisterExternalProcessorServer(a.server, a)

	logging.Infof("ExtProc adapter listening on port %d", a.port)
	return a.server.Serve(lis)
}

func (a *Adapter) Stop() error {
	logging.Infof("Stopping ExtProc adapter")
	if a.server != nil {
		a.server.GracefulStop()
	}
	return nil
}

func (a *Adapter) Process(stream ext_proc.ExternalProcessor_ProcessServer) error {
	ctx := &RequestContext{
		Headers: make(map[string]string),
	}

	for {
		req, err := stream.Recv()
		if err != nil {
			return err
		}

		switch v := req.Request.(type) {
		case *ext_proc.ProcessingRequest_RequestHeaders:
			response, err := a.handleRequestHeaders(v, ctx)
			if err != nil {
				return err
			}
			if err := stream.Send(response); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_RequestBody:
			response, err := a.handleRequestBody(v, ctx, stream.Context())
			if err != nil {
				return err
			}
			if err := stream.Send(response); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseHeaders:
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseHeaders{
					ResponseHeaders: &ext_proc.HeadersResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}
			if err := stream.Send(response); err != nil {
				return err
			}

		case *ext_proc.ProcessingRequest_ResponseBody:
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_ResponseBody{
					ResponseBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}
			if err := stream.Send(response); err != nil {
				return err
			}

		default:
			response := &ext_proc.ProcessingResponse{
				Response: &ext_proc.ProcessingResponse_RequestBody{
					RequestBody: &ext_proc.BodyResponse{
						Response: &ext_proc.CommonResponse{
							Status: ext_proc.CommonResponse_CONTINUE,
						},
					},
				},
			}
			if err := stream.Send(response); err != nil {
				return err
			}
		}
	}
}

func (a *Adapter) handleRequestHeaders(v *ext_proc.ProcessingRequest_RequestHeaders, ctx *RequestContext) (*ext_proc.ProcessingResponse, error) {
	for _, h := range v.RequestHeaders.GetHeaders().GetHeaders() {
		ctx.Headers[h.Key] = string(h.RawValue)
	}

	if reqID, ok := ctx.Headers["x-request-id"]; ok {
		ctx.RequestID = reqID
	}

	method := ctx.Headers[":method"]
	path := ctx.Headers[":path"]

	if method == "GET" && (path == "/v1/router_replay" || len(path) > len("/v1/router_replay/") && path[:len("/v1/router_replay/")] == "/v1/router_replay/") {
		logging.Infof("ExtProc: Handling router replay request %s", ctx.RequestID)
		return a.handleReplayRequest(path)
	}

	logging.Infof("ExtProc: Processing request %s", ctx.RequestID)

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_RequestHeaders{
			RequestHeaders: &ext_proc.HeadersResponse{
				Response: &ext_proc.CommonResponse{
					Status: ext_proc.CommonResponse_CONTINUE,
				},
			},
		},
	}, nil
}

func (a *Adapter) handleRequestBody(v *ext_proc.ProcessingRequest_RequestBody, ctx *RequestContext, grpcCtx context.Context) (*ext_proc.ProcessingResponse, error) {
	body := v.RequestBody.GetBody()
	var chatReq struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
		User string `json:"user,omitempty"`
	}

	if err := json.Unmarshal(body, &chatReq); err != nil {
		logging.Errorf("ExtProc: Failed to parse request body: %v", err)
		return nil, status.Errorf(codes.InvalidArgument, "invalid request body")
	}

	messages := make([]engine.Message, len(chatReq.Messages))
	for i, msg := range chatReq.Messages {
		messages[i] = engine.Message{
			Role:    msg.Role,
			Content: msg.Content,
		}
	}

	routeReq := &engine.RouteRequest{
		Model:    chatReq.Model,
		Messages: messages,
		User:     chatReq.User,
		Headers:  ctx.Headers,
		Context:  grpcCtx,
	}

	routeResp, err := a.engine.Route(grpcCtx, routeReq)
	if err != nil {
		logging.Errorf("ExtProc: Routing failed: %v", err)
		return nil, status.Errorf(codes.Internal, "routing failed: %v", err)
	}

	if routeResp.CacheHit {
		logging.Infof("ExtProc: Cache hit for request %s", ctx.RequestID)
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &ext_proc.ImmediateResponse{
					Status: &typev3.HttpStatus{Code: typev3.StatusCode_OK},
					Body:   []byte(routeResp.CachedResponse),
					Headers: &ext_proc.HeaderMutation{
						SetHeaders: []*core.HeaderValueOption{
							{
								Header: &core.HeaderValue{
									Key:      "X-Cache",
									RawValue: []byte("HIT"),
								},
							},
						},
					},
				},
			},
		}, nil
	}

	if routeResp.Blocked {
		logging.Warnf("ExtProc: Request blocked: %s", routeResp.BlockReason)
		return &ext_proc.ProcessingResponse{
			Response: &ext_proc.ProcessingResponse_ImmediateResponse{
				ImmediateResponse: &ext_proc.ImmediateResponse{
					Status: &typev3.HttpStatus{Code: typev3.StatusCode_Forbidden},
					Body:   []byte(fmt.Sprintf(`{"error": {"message": "%s", "type": "forbidden"}}`, routeResp.BlockReason)),
				},
			},
		}, nil
	}

	setHeaders := []*core.HeaderValueOption{
		{
			Header: &core.HeaderValue{
				Key:      "x-vsr-selected-model",
				RawValue: []byte(routeResp.SelectedModel),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      "x-router-decision",
				RawValue: []byte(routeResp.DecisionName),
			},
		},
		{
			Header: &core.HeaderValue{
				Key:      "content-type",
				RawValue: []byte("application/json"),
			},
		},
	}

	if routeResp.ReplayID != "" {
		setHeaders = append(setHeaders, &core.HeaderValueOption{
			Header: &core.HeaderValue{
				Key:      "x-replay-id",
				RawValue: []byte(routeResp.ReplayID),
			},
		})
	}

	logging.Infof("ExtProc: Returning backend response (status=%d, size=%d bytes)", routeResp.StatusCode, len(routeResp.ResponseBody))

	statusCode := typev3.StatusCode_OK
	if routeResp.StatusCode >= 0 && routeResp.StatusCode <= 599 {
		statusCode = typev3.StatusCode(routeResp.StatusCode)
	}

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: statusCode,
				},
				Body: routeResp.ResponseBody,
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: setHeaders,
				},
			},
		},
	}, nil
}

func (a *Adapter) handleReplayRequest(path string) (*ext_proc.ProcessingResponse, error) {
	var responseBody []byte
	var statusCode int32 = 200

	if path == "/v1/router_replay" {
		hasRecorders := len(a.engine.ReplayRecorders) > 0
		if !hasRecorders {
			response := map[string]interface{}{
				"object":  "router_replay.list",
				"count":   0,
				"data":    []interface{}{},
				"message": "Router replay not configured",
			}
			responseBody, _ = json.Marshal(response)
		} else {
			var allRecords []interface{}
			for _, recorder := range a.engine.ReplayRecorders {
				if recorder != nil {
					records := recorder.ListAllRecords()
					for _, r := range records {
						allRecords = append(allRecords, r)
					}
				}
			}
			response := map[string]interface{}{
				"object": "router_replay.list",
				"count":  len(allRecords),
				"data":   allRecords,
			}
			responseBody, _ = json.Marshal(response)
		}
	} else if len(path) > len("/v1/router_replay/") {
		replayID := path[len("/v1/router_replay/"):]
		found := false

		for _, recorder := range a.engine.ReplayRecorders {
			if recorder != nil {
				if rec, ok := recorder.GetRecord(replayID); ok {
					responseBody, _ = json.Marshal(rec)
					found = true
					break
				}
			}
		}

		if !found {
			statusCode = 404
			errorResponse := map[string]interface{}{
				"error": map[string]interface{}{
					"message": "replay record not found",
					"type":    "not_found_error",
				},
			}
			responseBody, _ = json.Marshal(errorResponse)
		}
	} else {
		statusCode = 400
		errorResponse := map[string]interface{}{
			"error": map[string]interface{}{
				"message": "invalid replay request",
				"type":    "invalid_request_error",
			},
		}
		responseBody, _ = json.Marshal(errorResponse)
	}

	logging.Infof("ExtProc: Returning replay response (status=%d, size=%d bytes)", statusCode, len(responseBody))

	return &ext_proc.ProcessingResponse{
		Response: &ext_proc.ProcessingResponse_ImmediateResponse{
			ImmediateResponse: &ext_proc.ImmediateResponse{
				Status: &typev3.HttpStatus{
					Code: typev3.StatusCode(statusCode),
				},
				Body: responseBody,
				Headers: &ext_proc.HeaderMutation{
					SetHeaders: []*core.HeaderValueOption{
						{
							Header: &core.HeaderValue{
								Key:      "content-type",
								RawValue: []byte("application/json"),
							},
						},
					},
				},
			},
		},
	}, nil
}

func (a *Adapter) GetEngine() *engine.RouterEngine {
	return a.engine
}
