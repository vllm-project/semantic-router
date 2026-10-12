package extproc

import (
	"context"
	"errors"
	"io"
	"runtime/debug"
	"strconv"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routerreplay"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// A routing session runs the phase logic an ext_proc stream runs and ends
// through the same lifecycle (replay, in-flight admission, traces), so the
// native gateway and Envoy get identical behavior from one pipeline.

var (
	_ routing.Processor = (*RouterService)(nil)
	_ routing.Processor = (*OpenAIRouter)(nil)
)

var errSessionEnded = errors.New("routing session has ended")

// Open starts a routing session on the current router generation. The
// generation stays leased until the session closes, as it does for an ext_proc
// stream, so a reload drains the session before closing the old router.
func (rs *RouterService) Open(ctx context.Context) (routing.Session, error) {
	router, release, err := rs.lease()
	if err != nil {
		return nil, err
	}
	return router.newRoutingSession(ctx, release), nil
}

// Open starts a routing session on this router.
func (r *OpenAIRouter) Open(ctx context.Context) (routing.Session, error) {
	return r.newRoutingSession(ctx, nil), nil
}

// routingSession is not safe for concurrent use: one request drives its
// phases in order.
type routingSession struct {
	router  *OpenAIRouter
	ctx     *RequestContext
	release func()
	ended   bool
	closed  bool
	// clientHeader is the request as the header phase left it, before any
	// provider dispatch: the base of a fallback candidate's request.
	clientHeader routing.Header
	// fallback is set once the caller takes over the request's fallback.
	fallback *sessionFallback
}

func (r *OpenAIRouter) newRoutingSession(ctx context.Context, release func()) *routingSession {
	if ctx == nil {
		ctx = context.Background()
	}
	session := &routingSession{
		router:  r,
		release: release,
		// Request bodies arrive whole, as Envoy delivers them in BUFFERED mode.
		ctx: &RequestContext{
			Headers: make(map[string]string), TraceContext: ctx, BufferedRequestBody: true,
			ConfigVersion: r.configVersion.Load(),
		},
	}
	if hop, ok := routing.HopFrom(ctx); ok {
		session.ctx.Hop = &hop
	} else if models, ok := routing.ListenerModelsFrom(ctx); ok {
		// A hop's context derives from its client request's, so only a
		// client request takes the listener's restriction.
		session.ctx.ListenerModels = models
	}
	return session
}

func (s *routingSession) RequestHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	v := &ext_proc.ProcessingRequest_RequestHeaders{
		RequestHeaders: &ext_proc.HttpHeaders{Headers: extprocHeaderMap(header), EndOfStream: endOfStream},
	}
	effect, err := s.phase(func() (*ext_proc.ProcessingResponse, error) {
		return s.router.requestHeadersReply(v, s.ctx)
	}, func(response *ext_proc.ProcessingResponse) {
		finishImmediateResponseTrace(s.ctx, response)
	})
	if err == nil {
		s.clientHeader = header.Clone()
		if effect.Header != nil {
			err = routing.ApplyHeaderMutation(&s.clientHeader, effect.Header, false, routing.DefaultLimits)
		}
	}
	return effect, err
}

func (s *routingSession) RequestBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	v := &ext_proc.ProcessingRequest_RequestBody{
		RequestBody: &ext_proc.HttpBody{Body: body, EndOfStream: endOfStream},
	}
	return s.phase(func() (*ext_proc.ProcessingResponse, error) {
		response, err := s.router.handleRequestBodyDispatch(v, s.ctx)
		return s.router.requestBodyReply(response, err, s.ctx)
	}, func(response *ext_proc.ProcessingResponse) {
		finishImmediateResponseTrace(s.ctx, response)
	})
}

func (s *routingSession) ResponseHeaders(header routing.Header, endOfStream bool) (*routing.Effect, error) {
	if s.fallback != nil {
		// The caller asks the fallback only about failures, so a fallback
		// candidate's success is recorded here, before its headers are built.
		if status, err := strconv.Atoi(header.Get(":status")); err == nil && status >= 200 && status < 300 {
			s.fallback.succeeded(status)
		}
	}
	v := &ext_proc.ProcessingRequest_ResponseHeaders{
		ResponseHeaders: &ext_proc.HttpHeaders{Headers: extprocHeaderMap(header), EndOfStream: endOfStream},
	}
	return s.phase(func() (*ext_proc.ProcessingResponse, error) {
		return s.router.responseHeadersReply(v, s.ctx)
	}, func(response *ext_proc.ProcessingResponse) {
		responseHeadersReplySent(s.ctx, response, endOfStream)
	})
}

func (s *routingSession) ResponseBody(body []byte, endOfStream bool) (*routing.Effect, error) {
	v := &ext_proc.ProcessingRequest_ResponseBody{
		ResponseBody: &ext_proc.HttpBody{Body: body, EndOfStream: endOfStream},
	}
	return s.phase(func() (*ext_proc.ProcessingResponse, error) {
		return s.router.responseBodyReply(v, s.ctx)
	}, func(response *ext_proc.ProcessingResponse) {
		responseBodyReplySent(s.ctx, response, endOfStream)
	})
}

func (s *routingSession) Evidence() routing.Evidence {
	return routingEvidence(s.ctx)
}

// Close ends the request the way the end of its ext_proc stream would: EOF
// after a delivered response, cancellation or a deadline when the transport
// gave up early.
func (s *routingSession) Close(err error) {
	if s.closed {
		return
	}
	s.closed = true
	if s.fallback != nil {
		// A chain the caller abandoned still leaves one audit record.
		s.router.recordFallbackExecution(s.ctx)
	}
	if s.ended {
		return
	}
	transportErr := io.EOF
	switch {
	case err == nil:
	case errors.Is(err, context.DeadlineExceeded):
		transportErr = context.DeadlineExceeded
	default:
		transportErr = context.Canceled
	}
	s.end(s.router.handleProcessReceiveError(s.ctx, transportErr))
}

// phase runs one reply builder with Process's failure semantics: a panic or
// an error finalizes replay and ends the request.
func (s *routingSession) phase(
	reply func() (*ext_proc.ProcessingResponse, error),
	sent func(*ext_proc.ProcessingResponse),
) (effect *routing.Effect, err error) {
	if s.ended || s.closed {
		return nil, errSessionEnded
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			s.router.finalizeRouterReplay(s.ctx, routerreplay.LifecycleFailed, "processor_panic")
			logging.Errorf("routing session: recovered panic: %v\n%s", recovered, debug.Stack())
			effect, err = nil, status.Errorf(codes.Internal, "internal error: %v", recovered)
			s.end(err)
		}
	}()
	response, err := reply()
	if err == nil {
		sent(response)
		effect, err = routingEffect(response)
	}
	if err != nil {
		state, reason := replayLifecycleForProcessError(err)
		s.router.finalizeRouterReplay(s.ctx, state, reason)
		s.end(err)
		return nil, err
	}
	return effect, nil
}

func (s *routingSession) end(err error) {
	if s.ended {
		return
	}
	s.ended = true
	releaseInflightAdmission(s.ctx)
	finishRequestTrace(s.ctx, err)
	if s.release != nil {
		s.release()
	}
}
