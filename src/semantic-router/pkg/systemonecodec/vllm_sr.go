package systemonecodec

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// VLLMSR40Codec speaks vllm-sr/4.0, the legacy model runtime contract 4.0.0
// exactly (config/schemas/systemone/vllm-sr-4.0.openapi.yaml): a batch in one
// call as states, the caller's deadline as options.deadline_ms, every
// extension field, choices, levels and labels, images and videos as data
// URLs, implicit roles from the state's keys, model cards in the listing's
// data, and an error as {error: {code, message, question, state,
// question_error}}. Its requests are the internal form without roles, which
// 4.0 cannot carry unless they are the ones its state keys imply.
type VLLMSR40Codec struct{}

func (VLLMSR40Codec) ID() SchemaID { return SchemaVLLMSR40 }

func (VLLMSR40Codec) Spec() SpecRef {
	spec, _ := VendoredSpec(SchemaVLLMSR40)
	return spec
}

func (VLLMSR40Codec) Capabilities() CapabilitySet {
	return Capabilities(CapabilityBatch, CapabilityObjectState, CapabilityDeadline, CapabilityPresets, CapabilitySetQuestions,
		CapabilitySpan, CapabilityFullInput, CapabilityImages, CapabilityVideos, CapabilityCard, CapabilityQuestionErrors)
}

func (VLLMSR40Codec) Stateless() bool { return true }

// ListingPath appends to the endpoint: POST /v1/systemone lists at
// GET /v1/systemone/models.
func (VLLMSR40Codec) ListingPath() string { return "/models" }

// Plan sends the batch in one call: a single request for one item, else a
// batch whose entries are the items' names.
func (VLLMSR40Codec) Plan(batch systemoneprotocol.Batch, target Target, _ Policy) (Plan, Diagnostics, error) {
	plan := Plan{Names: make([]string, len(batch.Items))}
	items := make([]int, len(batch.Items))
	var diagnostics Diagnostics
	sent := batch
	sent.Items = make([]systemoneprotocol.NamedRequest, len(batch.Items))
	for index, item := range batch.Items {
		plan.Names[index], items[index] = item.Name, index
		request := item.Request
		if request.Roles.Mode != systemoneprotocol.RolesImplicit {
			if !request.Roles.SameAsImplicit(request.State) {
				return Plan{}, diagnostics, fmt.Errorf("request %q: vllm-sr/4.0 gives an object state's parts the roles their keys name, so it cannot carry these roles", item.Name)
			}
			request.Roles = systemoneprotocol.Roles{}
			diagnostics = append(diagnostics, Diagnostic{Field: "roles", Action: DiagnosticStripped, Reason: "vllm-sr/4.0's implicit roles are the same"})
		}
		sent.Items[index] = systemoneprotocol.NamedRequest{Name: item.Name, Request: request}
	}
	if sent.Model == "" {
		sent.Model = target.Model
	}
	if len(sent.Items) == 1 {
		sent.Form = systemoneprotocol.FormAlone
		sent.Items[0].Request.Model = ""
	} else if sent.Form != systemoneprotocol.FormStates {
		sent.Form, sent.Order = systemoneprotocol.FormStates, nil
	}
	body, err := InternalCodec{}.EncodeRequestValue(sent)
	if err != nil {
		return Plan{}, diagnostics, err
	}
	header := http.Header{"Content-Type": {"application/json"}}
	plan.Calls = []WireCall{{Header: header, Body: body, Items: items, Batch: len(batch.Items) > 1}}
	return plan, diagnostics, nil
}

// DecodeReply reads a native reply: one response, a batch's entries, or the
// error that answers every item the call carried.
func (VLLMSR40Codec) DecodeReply(plan Plan, call WireCall, reply WireReply) ([]ItemResult, error) {
	results := make([]ItemResult, len(call.Items))
	if reply.Status < 200 || reply.Status >= 300 {
		problem := NativeProblem(reply.Status, reply.Body)
		for index := range results {
			results[index].Err = problem
		}
		return results, nil
	}
	value, err := systemoneprotocol.Parse(reply.Body)
	if err != nil || value.Kind() != systemoneprotocol.KindObject {
		return nil, &systemoneprotocol.Problem{Status: reply.Status, Kind: systemoneprotocol.ProblemFailed, Detail: "the response is not a JSON object"}
	}
	if !call.Batch {
		response, err := responseFromValue(value)
		for index := range results {
			results[index] = ItemResult{Response: response, Err: err}
		}
		return results, nil
	}
	states, _ := value.Get("states")
	model := textOf(mustGet(value, "model"))
	for index, item := range call.Items {
		entry, ok := states.Get(plan.Names[item])
		if !ok {
			results[index].Err = &systemoneprotocol.Problem{Status: reply.Status, Kind: systemoneprotocol.ProblemFailed, Detail: fmt.Sprintf("the batch response has no answers about %q", plan.Names[item])}
			continue
		}
		response, err := responseFromValue(entry)
		if response.Model == "" {
			response.Model = model
		}
		results[index] = ItemResult{Response: response, Err: err}
	}
	return results, nil
}

// NativeProblem reads a native error body,
// {error: {code, message, reason, limit, question, state, question_error}}.
func NativeProblem(status int, body []byte) *systemoneprotocol.Problem {
	problem := &systemoneprotocol.Problem{Status: status, Kind: problemKind(status)}
	var answer struct {
		Error struct {
			Code          string `json:"code"`
			Message       string `json:"message"`
			Reason        string `json:"reason"`
			Limit         *int64 `json:"limit"`
			Question      string `json:"question"`
			State         string `json:"state"`
			QuestionError string `json:"question_error"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &answer) != nil {
		problem.Detail = http.StatusText(status)
		return problem
	}
	fields := answer.Error
	problem.Code, problem.Detail, problem.Reason, problem.Limit = fields.Code, fields.Message, fields.Reason, fields.Limit
	if fields.Code == "question_failed" && fields.Question != "" {
		problem.Question, problem.Item, problem.QuestionError = fields.Question, fields.State, fields.QuestionError
	}
	return problem
}

// DecodeListing reads a runtime listing: its cards in data, by ID.
func (VLLMSR40Codec) DecodeListing(status int, body []byte) (systemoneprotocol.Listing, error) {
	if status != http.StatusOK {
		return systemoneprotocol.Listing{}, fmt.Errorf("the models listing answered %d", status)
	}
	var listing struct {
		Data []json.RawMessage `json:"data"`
	}
	if err := json.Unmarshal(body, &listing); err != nil || listing.Data == nil {
		return systemoneprotocol.Listing{}, fmt.Errorf("the models listing is not a model list")
	}
	decoded := systemoneprotocol.Listing{}
	for _, raw := range listing.Data {
		entry, err := systemoneprotocol.Parse(raw)
		if err != nil || entry.Kind() != systemoneprotocol.KindObject {
			return systemoneprotocol.Listing{}, fmt.Errorf("a models entry is not an object")
		}
		model := systemoneprotocol.ListedModel{Name: textOf(mustGet(entry, "id"))}
		model.Extensions = append(model.Extensions, systemoneprotocol.Member{Key: "card", Value: entry})
		decoded.Models = append(decoded.Models, model)
	}
	return decoded, nil
}

// NativeErrorBody is an error as the model runtime writes it, for a client of
// the native contract: its status and {error: {code, message, ...}}.
func NativeErrorBody(err error, item string) (int, []byte) {
	problem := AsProblem(err)
	status, code := nativeStatus(problem)
	body, _ := json.Marshal(map[string]any{"error": nativeErrorFields(problem, code, item)})
	return status, body
}

// AsProblem is err as a Problem: itself, or a failed call with its text.
func AsProblem(err error) *systemoneprotocol.Problem {
	var problem *systemoneprotocol.Problem
	if errors.As(err, &problem) {
		return problem
	}
	return &systemoneprotocol.Problem{Status: http.StatusBadGateway, Kind: systemoneprotocol.ProblemFailed, Detail: err.Error()}
}

func nativeErrorFields(problem *systemoneprotocol.Problem, code, item string) map[string]any {
	fields := map[string]any{"code": code, "message": problem.Detail}
	if problem.Reason != "" {
		fields["reason"] = problem.Reason
	}
	if problem.Limit != nil {
		fields["limit"] = *problem.Limit
	}
	if state := firstNonEmpty(problem.Item, item); state != "" && code == "question_failed" {
		fields["state"] = state
	}
	if code == "question_failed" {
		fields["question"], fields["question_error"] = problem.Question, firstNonEmpty(problem.QuestionError, "invalid_question")
	}
	return fields
}

func nativeStatus(problem *systemoneprotocol.Problem) (int, string) {
	if problem.Question != "" {
		return questionStatus(problem), "question_failed"
	}
	if problem.Code != "" && problem.Status >= 400 {
		return problem.Status, problem.Code
	}
	switch problem.Kind {
	case systemoneprotocol.ProblemRejected:
		switch problem.Status {
		case http.StatusRequestEntityTooLarge:
			return problem.Status, "request_too_large"
		case http.StatusUnsupportedMediaType:
			return problem.Status, "unsupported_media_type"
		case http.StatusNotFound:
			return problem.Status, "model_not_found"
		}
		return http.StatusBadRequest, "invalid_request"
	case systemoneprotocol.ProblemOverloaded:
		return http.StatusTooManyRequests, "overloaded"
	case systemoneprotocol.ProblemUnavailable:
		return http.StatusServiceUnavailable, "not_ready"
	case systemoneprotocol.ProblemDeadline:
		return http.StatusGatewayTimeout, "deadline_exceeded"
	}
	return http.StatusBadGateway, "internal_error"
}

// questionStatus is the status of a failed question, by its error, as the
// model runtime sets it.
func questionStatus(problem *systemoneprotocol.Problem) int {
	switch problem.QuestionError {
	case "invalid_input", "max_length_exceeded", "scan_budget_exceeded":
		return http.StatusBadRequest
	case "invalid_model_output":
		return http.StatusInternalServerError
	case "unavailable":
		return http.StatusServiceUnavailable
	case "deadline_exceeded":
		return http.StatusGatewayTimeout
	}
	if problem.Status >= 400 && problem.Status < 600 && problem.Status != http.StatusTooManyRequests {
		return problem.Status
	}
	return http.StatusUnprocessableEntity
}
