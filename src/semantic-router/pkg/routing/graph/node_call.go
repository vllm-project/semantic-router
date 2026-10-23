package graph

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/fallback"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/routing"
)

// CallError reports a call step whose model call failed; Result holds what
// came back, if anything did.
type CallError struct {
	Result *Result
}

func (e *CallError) Error() string {
	if e.Result.Err != nil {
		return fmt.Sprintf("model %q call failed: %v", e.Result.Model, e.Result.Err)
	}
	return fmt.Sprintf("model %q answered %d", e.Result.Model, e.Result.Status)
}

func (e *CallError) Unwrap() error { return e.Result.Err }

// Call is the call step: it sends the state's request to one model and makes
// the answer the state's only result.
type Call struct {
	Model string
	// Fields are request fields this call sets, such as temperature or
	// max_tokens, as JSON values.
	Fields map[string]json.RawMessage
	// Decision, when set, names the decision whose plugin chain the hop runs
	// instead of the request's.
	Decision string
	// Reliability layers over the decision's timeouts and retries for this
	// hop.
	Reliability *routing.Reliability
	// Fallback overrides the decision's cross-model fallback for this hop.
	Fallback *fallback.FallbackOverride
}

// Run sends the call.
func (n *Call) Run(ctx context.Context, x *Exec, st *State) error {
	request := st.Request.Clone()
	if err := request.Set("model", n.Model); err != nil {
		return err
	}
	for name, value := range n.Fields {
		request.SetRaw(name, value)
	}
	body, err := request.Body()
	if err != nil {
		return err
	}
	header := x.Header()
	header.Set("content-length", strconv.Itoa(len(body)))
	hop := x.Hop()
	if n.Decision != "" {
		hop.Decision = n.Decision
	}
	hop.Iteration, hop.Fallback = 0, n.Fallback
	result := x.CallModel(ctx, &HopRequest{
		Hop:         hop,
		Model:       n.Model,
		Request:     routing.Request{Header: header, Body: body},
		Reliability: n.Reliability,
	})
	st.Results = []*Result{result}
	if !result.OK() {
		return &CallError{Result: result}
	}
	return nil
}

// CallModel sends one hop and returns its result, failed or not.
func (x *Exec) CallModel(ctx context.Context, req *HopRequest) *Result {
	started := time.Now()
	resp, err := x.Call(ctx, req)
	result := &Result{Step: stepOf(ctx), Model: req.Model, Latency: time.Since(started), Err: err}
	if resp != nil {
		result.Status, result.Header, result.Body = resp.Status, resp.Header, resp.Body
		result.Usage = completionUsage(resp.Body)
	}
	return result
}
