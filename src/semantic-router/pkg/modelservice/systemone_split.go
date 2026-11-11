package modelservice

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"sort"
	"strconv"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/presets"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemonecodec"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// sendCodec asks a fused task of a deployment through its schema's codec: a
// schema with batches gets the task's states in one call, one without them
// each state in a call of its own, at most maxConcurrency in flight (a
// sliding window: each answer starts the next), every call within the task's
// deadline. Each state is answered as soon as its call is: a call that fails,
// or that the deadline cuts short, leaves only its own states' questions
// unknown.
func (d *systemOneDeployment) sendCodec(ctx context.Context, task *bundleTask) {
	states := task.decision.states
	batch := systemoneprotocol.Batch{Model: d.model, Items: make([]systemoneprotocol.NamedRequest, len(states))}
	planErrs := make([]error, len(states))
	for index, state := range states {
		request, err := stateRequest(state)
		planErrs[index] = err
		batch.Items[index] = systemoneprotocol.NamedRequest{Name: strconv.Itoa(index + 1), Request: request}
	}
	responses := make([]Response, len(states))
	errs := make([]error, len(states))
	exchange := d.exchanger()
	exchange.Answered = func(index int, result systemonecodec.ItemResult) {
		responses[index], errs[index] = d.decided(result, states[index].request.Questions)
		task.deliverState(index, responses[index], errs[index])
	}
	exchange.Ask(ctx, batch, planErrs)
	complete := true
	for index := range states {
		complete = complete && errs[index] == nil && completeAll([]Response{responses[index]})
	}
	if cache := task.decision.cache; cache != nil {
		for _, call := range task.calls {
			cacheTotal.WithLabelValues(call.decision.deployment, "miss").Inc()
		}
		if complete {
			if len(responses) == 1 {
				cache.put(task.decision.key, responses[0])
			} else {
				cache.put(task.decision.key, responses)
			}
		}
	}
}

// decided is the response an item's result gives a state's questions, or the
// error that leaves them unknown.
func (d *systemOneDeployment) decided(result systemonecodec.ItemResult, questions []Question) (Response, error) {
	if result.Err != nil {
		return Response{}, d.decideError(result.Err)
	}
	response, err := decideResponse(result.Response, questions)
	if err != nil {
		d.reportFailure(http.StatusOK, ErrorReason(err), nil)
		return Response{}, err
	}
	return response, nil
}

// forwardCodec sends a client's request, in the internal form, to a
// deployment of another schema: read into the IR, asked through the
// deployment's codec and answered in the internal form, a batch reassembled
// under its entries' names.
func (d *systemOneDeployment) forwardCodec(ctx context.Context, body json.RawMessage) (SystemOneResult, error) {
	internal := systemonecodec.InternalCodec{}
	batch, err := internal.DecodeRequest(body)
	if err != nil {
		status, encoded := systemonecodec.NativeErrorBody(&systemoneprotocol.Problem{Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected, Detail: err.Error()}, "")
		return SystemOneResult{Status: status, Body: encoded}, nil
	}
	form := batch.Form
	batch.Model = d.model
	names := make([]string, len(batch.Items))
	for index := range batch.Items {
		batch.Items[index].Request.Model = ""
		names[index] = batch.Items[index].Name
	}
	results := d.exchanger().Ask(ctx, batch, nil)
	if systemonecodec.AllTransport(results) {
		return SystemOneResult{}, fmt.Errorf("%w: %w", ErrUnavailable, results[0].Err)
	}
	status, encoded, err := internal.EncodeResults(results, names, form, d.model)
	if err != nil {
		return SystemOneResult{}, err
	}
	return SystemOneResult{Status: status, Body: encoded}, nil
}

// exchanger asks the deployment through its codec and deviations, with the
// deployment's metrics and logs.
func (d *systemOneDeployment) exchanger() systemonecodec.Exchange {
	return systemonecodec.Exchange{
		Codec: d.codec, Deviations: d.deviations,
		Target:         systemonecodec.Target{Deployment: d.name, Model: d.model},
		Policy:         systemonecodec.Policy{StripUnsupported: true, Presets: CanonicalPreset},
		MaxConcurrency: d.maxConcurrency,
		Send:           d.exchangeCall,
		Observer: systemonecodec.Observer{
			Diagnostics: d.recordDiagnostics,
			Retried: func(question, questionError string) {
				systemOneRetries.WithLabelValues(d.name).Inc()
				logging.ComponentDebugEvent("model_runtime", "systemone_question_retried", map[string]interface{}{
					"deployment": d.name, "question": question, "question_error": questionError,
				})
			},
			Replied: func(reply systemonecodec.WireReply, err error, elapsed time.Duration) {
				outcome := err
				if outcome == nil {
					outcome = systemOneStatusError(reply.Status)
				}
				requestDuration.WithLabelValues(d.name, "decisions").Observe(elapsed.Seconds())
				requestsTotal.WithLabelValues(d.name, ErrorReason(outcome)).Inc()
				if err == nil && outcome != nil {
					d.reportFailure(reply.Status, ErrorReason(outcome), nil)
				}
			},
		},
	}
}

// modelPresets are the Router presets whose question a model preset is
// trained on.
var modelPresets = map[string]struct{ id, stage string }{
	"pii":  {presets.PII, presets.StageRequest},
	"halu": {presets.Hallucination, presets.StageResponse},
}

// CanonicalPreset is the Router's canonical question of a model preset (pii,
// halu) or a Router preset: its type, instructions and criteria, which a
// schema whose preset questions carry their content sends beside the
// preset's name.
func CanonicalPreset(name string) (systemoneprotocol.Question, bool) {
	id, stage := name, presets.StageRequest
	if preset, ok := modelPresets[name]; ok {
		id, stage = preset.id, preset.stage
	}
	canonical, ok := presets.Lookup(id, stage)
	if !ok {
		return systemoneprotocol.Question{}, false
	}
	value, err := systemoneprotocol.FromGo(encodeQuestion(PresetQuestion(name, canonical)))
	if err != nil {
		return systemoneprotocol.Question{}, false
	}
	question, err := systemoneprotocol.ParseQuestion(value)
	if err != nil {
		return systemoneprotocol.Question{}, false
	}
	if question, err = question.Standard(); err != nil {
		return systemoneprotocol.Question{}, false
	}
	return systemoneprotocol.Question{Type: question.Type, Instructions: question.Instructions, Criteria: question.Criteria}, true
}

// exchangeCall posts one call and returns the API's reply, an error status's
// included with its Retry-After; err is set when there is none.
func (d *systemOneDeployment) exchangeCall(ctx context.Context, call systemonecodec.WireCall, body []byte) (systemonecodec.WireReply, error) {
	headers := map[string]string{}
	for name, values := range call.Header {
		if len(values) > 0 && http.CanonicalHeaderKey(name) != "Content-Type" {
			headers[name] = values[0]
		}
	}
	operation := systemOneOperation
	operation.Method, operation.Path = http.MethodPost, call.Path
	result, err := d.client.DoRequest(ctx, operation, connector.Request{Body: body, Headers: headers})
	if ctxErr := ctx.Err(); ctxErr != nil {
		return systemonecodec.WireReply{}, ctxErr
	}
	if err == nil {
		return systemonecodec.WireReply{Status: result.StatusCode, Body: result.Body, Header: http.Header{}}, nil
	}
	var remote *connector.Error
	if errors.As(err, &remote) && remote.Kind == connector.KindStatus {
		data, _ := remote.ResponseBody()
		header := http.Header{}
		if remote.RetryAfter != "" {
			header.Set("Retry-After", remote.RetryAfter)
		}
		return systemonecodec.WireReply{Status: remote.StatusCode, Body: data, Header: header}, nil
	}
	_, _, err = d.exchangeFailure(err)
	return systemonecodec.WireReply{}, fmt.Errorf("%w: systemone deployment %q: %w", ErrFailed, d.name, err)
}

// exchangeFailure reports a transport or credential failure as exchange does.
func (d *systemOneDeployment) exchangeFailure(err error) (int, []byte, error) {
	var remote *connector.Error
	if errors.As(err, &remote) {
		if remote.Kind == connector.KindAuthorization {
			d.reportFailure(0, "credential", remote.Cause)
			return 0, nil, remote.Cause
		}
		cause := remote.Cause
		if cause == nil {
			cause = remote
		}
		d.reportFailure(remote.StatusCode, "transport", cause)
		return 0, nil, cause
	}
	return 0, nil, err
}

// decideError is an item's error as the package's errors.
func (d *systemOneDeployment) decideError(err error) error {
	var problem *systemoneprotocol.Problem
	if !errors.As(err, &problem) {
		return err
	}
	switch problem.Kind {
	case systemoneprotocol.ProblemOverloaded:
		return fmt.Errorf("%w: systemone deployment %q: %s", ErrOverloaded, d.name, problem.Error())
	case systemoneprotocol.ProblemUnavailable:
		return fmt.Errorf("%w: systemone deployment %q: %s", ErrUnavailable, d.name, problem.Error())
	case systemoneprotocol.ProblemRejected, systemoneprotocol.ProblemCredential:
		return fmt.Errorf("%w: systemone deployment %q: %s", ErrRejected, d.name, problem.Error())
	}
	return fmt.Errorf("%w: systemone deployment %q: %s", ErrFailed, d.name, problem.Error())
}

// recordDiagnostics logs what a codec or deviation changed, once per kind of
// change and deployment.
func (d *systemOneDeployment) recordDiagnostics(diagnostics systemonecodec.Diagnostics) {
	for _, diagnostic := range diagnostics {
		key := "diagnostic\x00" + diagnostic.Field + "\x00" + string(diagnostic.Action)
		if _, seen := d.failures.LoadOrStore(key, struct{}{}); seen {
			continue
		}
		logging.ComponentDebugEvent("model_runtime", "systemone_schema_conversion", map[string]interface{}{
			"deployment": d.name, "schema": string(d.schemaID()), "field": diagnostic.Field,
			"action": string(diagnostic.Action), "reason": diagnostic.Reason,
		})
	}
}

// stateRequest is a fused state's request in the schema-neutral form: its
// state as text or its named parts as an object (keys sorted, as the native
// request carries them), its images, and every question of its calls in
// name order, in their System One encoding.
func stateRequest(state *fusedState) (systemoneprotocol.Request, error) {
	own := state.calls[0].task.Decisions
	request := systemoneprotocol.Request{}
	value, err := systemoneprotocol.FromGo(own.State)
	if err != nil {
		return request, err
	}
	request.State = value
	if own.Images != nil && len(*own.Images) > 0 {
		images, err := systemoneprotocol.FromGo(*own.Images)
		if err != nil {
			return request, err
		}
		request.Extensions = append(request.Extensions, systemoneprotocol.Member{Key: "images", Value: images})
	}
	if own.Videos != nil && len(*own.Videos) > 0 {
		videos, err := systemoneprotocol.FromGo(*own.Videos)
		if err != nil {
			return request, err
		}
		request.Extensions = append(request.Extensions, systemoneprotocol.Member{Key: "videos", Value: videos})
	}
	questions := state.questions()
	names := make([]string, 0, len(questions))
	for name := range questions {
		names = append(names, name)
	}
	sort.Strings(names)
	for _, name := range names {
		encoded, err := systemoneprotocol.FromGo(questions[name])
		if err != nil {
			return request, err
		}
		question, err := systemoneprotocol.ParseQuestion(encoded)
		if err != nil {
			return request, fmt.Errorf("question %q: %w", name, err)
		}
		request.Questions = append(request.Questions, systemoneprotocol.NamedQuestion{Name: name, Question: question})
	}
	return request, nil
}

// decideResponse reads a schema-neutral response as the decisions answer to
// questions, through the native response the decoder reads.
func decideResponse(response systemoneprotocol.Response, questions []Question) (Response, error) {
	object := systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String(response.Model)},
		systemoneprotocol.Member{Key: "answers", Value: systemoneprotocol.Object(response.Answers...)},
		systemoneprotocol.Member{Key: "usage", Value: systemoneprotocol.Object(
			systemoneprotocol.Member{Key: "input_tokens", Value: systemoneprotocol.Int(response.Usage.InputTokens)},
			systemoneprotocol.Member{Key: "output_tokens", Value: systemoneprotocol.Int(response.Usage.OutputTokens)},
		)},
	)
	for _, member := range response.Extensions {
		object = object.With(member.Key, member.Value)
	}
	data, err := json.Marshal(object)
	if err != nil {
		return Response{}, err
	}
	decoded, err := decodeSystemOneBody(http.StatusOK, data, false)
	if err != nil {
		return Response{}, err
	}
	return decodeResponse(decoded, questions), nil
}

// deliverState hands every call about one of a decisions task's states its
// own answers, or the state's error.
func (t *bundleTask) deliverState(index int, response Response, err error) {
	for _, call := range t.decision.states[index].calls {
		call.err = err
		if err == nil {
			call.decision.response = response.part(call.decision.request.Questions)
		}
		close(call.done)
	}
}
