package extproc

import (
	"context"
	"errors"
	"io"
	"net/http"
	"sort"
	"strconv"
	"strings"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/configsnapshot"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/looper"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing/graph"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/upstream"
)

// errNoLooperHops reports a router with nothing to send Looper hops with: no
// configuration snapshot that owns an upstream set.
var errNoLooperHops = errors.New("the router's configuration has no upstream set to send Looper calls with")

// looperHops returns the caller that serves this router's Looper hops in
// process: each hop is a routing session on this router, sent through the
// upstream set of the configuration snapshot it serves. A caller set on the
// router takes their place.
func (r *OpenAIRouter) looperHops() graph.Caller {
	if r.hopCaller != nil {
		return r.hopCaller
	}
	set := r.upstreamSet()
	if set == nil {
		if testLooperHops != nil {
			return testLooperHops(r)
		}
		return nil
	}
	opts := routing.DefaultOptions
	opts.ExecutesFallback = true
	return &graph.SessionCaller{
		Engine:       routing.NewEngine(r, opts),
		Upstream:     upstreamSender{set: set},
		MaxBodyBytes: r.Config.Looper.GetMaxResponseBytes(),
	}
}

// testLooperHops, set only by tests, sends the hops of a router that has no
// upstream set, such as one a test builds without the configuration
// lifecycle.
var testLooperHops func(*OpenAIRouter) graph.Caller

// upstreamSet is the upstream set of the snapshot this router serves.
func (r *OpenAIRouter) upstreamSet() *upstream.Set {
	r.routerLearningMu.Lock()
	generation := r.generation
	r.routerLearningMu.Unlock()
	if generation == nil || generation.snapshot == nil {
		return nil
	}
	set, _ := generation.snapshot.Part(configsnapshot.ComponentUpstream).(*upstream.Set)
	return set
}

// executeLooperGraph runs the decision's built-in Looper template with its
// hops in process.
func (r *OpenAIRouter) executeLooperGraph(
	ctx context.Context,
	program *graph.Program,
	hops graph.Caller,
	req *looper.Request,
	originalModel string,
	decision *config.Decision,
	reqCtx *RequestContext,
) (*looper.Response, *ext_proc.ProcessingResponse) {
	in := graph.Input{Values: map[string]any{}}
	looper.RequestValue.Put(in.Values, req)
	outcome, err := graph.Run(ctx, program, in, graph.Options{
		Caller: hops,
		Hop:    routing.Hop{Decision: decision.Name, Recipe: string(req.RecipeName)},
	})
	if err != nil {
		return nil, r.looperExecutionErrorResponse(looperStepError(err), originalModel, decision, reqCtx)
	}
	resp, ok := looper.ResponseValue.In(outcome.Values)
	if !ok {
		return nil, r.looperExecutionErrorResponse(looper.ErrNoRequest, originalModel, decision, reqCtx)
	}
	logging.ComponentEvent("extproc", "looper_execution_completed", map[string]interface{}{
		"request_id":     reqCtx.RequestID,
		"decision":       decision.Name,
		"algorithm":      resp.AlgorithmType,
		"models_used":    resp.ModelsUsed,
		"iterations":     resp.Iterations,
		"selected_model": resp.Model,
		"hops":           outcome.Hops,
	})
	return resp, nil
}

// looperStepError is the algorithm's own error, which the client response
// and the failure evidence quote, without the graph's step attribution.
func looperStepError(err error) error {
	var stepErr *graph.StepError
	if errors.As(err, &stepErr) && stepErr.Type == looper.StepType {
		return stepErr.Err
	}
	return err
}

// upstreamSender sends a hop's planned call through the upstream layer, as
// the standalone gateway sends a client's.
type upstreamSender struct {
	set *upstream.Set
}

func (s upstreamSender) Send(ctx context.Context, call *routing.Call) (*graph.Delivery, error) {
	result, err := s.set.Execute(ctx, call, "")
	if err != nil {
		if ctx.Err() != nil {
			return nil, ctx.Err()
		}
		return localDelivery(err), nil
	}
	if result.Immediate != nil {
		return &graph.Delivery{Immediate: result.Immediate}, nil
	}
	resp := result.Response
	return &graph.Delivery{
		Response: &routing.UpstreamResponse{Status: resp.StatusCode, Header: routingHeaderOf(resp.Header), Body: resp.Body},
		Local:    resp.Local != nil,
		Close:    func() { _ = resp.Body.Close() },
	}, nil
}

// localDelivery answers a failure the upstream layer reports as an error,
// rather than as Envoy's local reply, with that local reply.
func localDelivery(err error) *graph.Delivery {
	status := http.StatusServiceUnavailable
	var upstreamErr *upstream.Error
	if errors.As(err, &upstreamErr) {
		status = upstreamErr.StatusCode()
	}
	text := http.StatusText(status)
	header := routing.Header{{Name: "content-type", Value: "text/plain"}, {Name: "content-length", Value: strconv.Itoa(len(text))}}
	return &graph.Delivery{
		Response: &routing.UpstreamResponse{Status: status, Header: header, Body: io.NopCloser(strings.NewReader(text))},
		Local:    true,
	}
}

// routingHeaderOf converts a response header to the routing core's shape:
// lowercase names, sorted.
func routingHeaderOf(h http.Header) routing.Header {
	names := make([]string, 0, len(h))
	for name := range h {
		names = append(names, name)
	}
	sort.Strings(names)
	out := make(routing.Header, 0, len(names))
	for _, name := range names {
		for _, value := range h[name] {
			out = append(out, routing.HeaderField{Name: strings.ToLower(name), Value: value})
		}
	}
	return out
}
