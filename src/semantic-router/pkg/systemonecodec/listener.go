package systemonecodec

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// The System One API's request paths on a listener.
const (
	PathSystemOne = "/v1/systemone"
	PathBatch     = "/v1/systemone/batch"
	PathDecisions = "/v1/decisions"
)

// PlanKind is how a listener sends a client's request on.
type PlanKind string

const (
	// PlanPassThrough forwards the client's bytes: a request already in the
	// internal form, which the listener decoded and checked first.
	PlanPassThrough PlanKind = "pass_through"
	// PlanTranslate forwards the IR written in the internal form.
	PlanTranslate PlanKind = "translate"
)

// LegacyForms are the vllm-sr/4.0 forms a listener counts, in order:
// implicit_roles is an object state without roles, which 4.0 reads by its
// keys and 5.0 as the user's content.
var LegacyForms = []string{"states", "images", "videos", "choices", "levels", "labels", "preset_only", "long_context", "implicit_roles"}

// ClientRequest is a client's request as a listener read and planned it.
type ClientRequest struct {
	Path  string
	Batch systemoneprotocol.Batch
	// Legacy are the vllm-sr/4.0 forms the request used, in LegacyForms
	// order.
	Legacy []string
	Plan   PlanKind
	// Internal is the request the listener sends on, in the internal form:
	// the client's bytes, or the IR's internal encoding.
	Internal    []byte
	Diagnostics Diagnostics
}

// ListenerCodec is the schema a listener speaks to its clients. It reads a
// client's request into the IR and plans it, and writes the answers the
// Router gives in the internal form (success, error and listing bodies) in
// the client's schema. /v1/decisions is native on every listener: it reads
// the internal form and answers in it, as on the model runtime.
type ListenerCodec interface {
	ID() SchemaID
	// Read decodes, checks and plans a request on one of the paths.
	Read(path string, body []byte, header http.Header) (ClientRequest, error)
	// AnswerSuccess writes an internal success body for the client; request
	// is nil when the listener never read one.
	AnswerSuccess(request *ClientRequest, body []byte) ([]byte, error)
	// AnswerError writes an internal error ({error: {...}}) for the client.
	AnswerError(request *ClientRequest, path string, status int, body []byte) (int, []byte)
	// AnswerListing writes the System One listing the Router gives
	// ({object, data, models}) for the client.
	AnswerListing(body []byte) ([]byte, error)
}

// Listener is the listener codec of a schema ID; empty is vllm-sr/5.0.
func Listener(id SchemaID) (ListenerCodec, error) {
	switch id {
	case "", SchemaVLLMSR50:
		return Listener50{}, nil
	case SchemaVLLMSR40:
		return Listener40{}, nil
	}
	return nil, fmt.Errorf("a listener speaks systemone in %s or %s, not %q", SchemaVLLMSR50, SchemaVLLMSR40, id)
}

// LegacyFormsOf are the vllm-sr/4.0 forms a request used, in LegacyForms
// order.
func LegacyFormsOf(batch systemoneprotocol.Batch) []string {
	used := map[string]bool{}
	if batch.Form == systemoneprotocol.FormStates {
		used["states"] = true
	}
	if batch.Options.LongContext {
		used["long_context"] = true
	}
	for _, item := range batch.Items {
		request := item.Request
		for _, member := range request.Extensions {
			if member.Key == "images" || member.Key == "videos" {
				used[member.Key] = true
			}
		}
		for _, named := range request.Questions {
			question := named.Question
			for _, member := range question.Extensions {
				switch member.Key {
				case "choices", "levels", "labels":
					used[member.Key] = true
				}
			}
			if _, preset := question.Extension("preset"); preset && (question.Type == "" || (question.Instructions.IsNull() && question.Criteria.IsNull())) {
				used["preset_only"] = true
			}
		}
		if request.State.Kind() == systemoneprotocol.KindObject && request.Roles.Mode == systemoneprotocol.RolesImplicit {
			used["implicit_roles"] = true
		}
	}
	var forms []string
	for _, form := range LegacyForms {
		if used[form] {
			forms = append(forms, form)
		}
	}
	return forms
}

// readNative reads a request in the internal form and plans it: its bytes
// passed through, or translated when the standard's long-context header asks
// for what its options do not say.
func readNative(path string, body []byte, header http.Header) (ClientRequest, error) {
	batch, err := InternalCodec{}.DecodeRequest(body)
	if err != nil {
		return ClientRequest{}, err
	}
	read := ClientRequest{Path: path, Batch: batch, Legacy: LegacyFormsOf(batch), Plan: PlanPassThrough, Internal: body}
	if strings.EqualFold(strings.TrimSpace(header.Get(HeaderLongContext)), "true") && !batch.Options.LongContext {
		read.Batch.Options.LongContext = true
		if read.Internal, err = (InternalCodec{}).EncodeRequest(read.Batch); err != nil {
			return ClientRequest{}, err
		}
		read.Plan = PlanTranslate
	}
	return read, nil
}

// translate writes a request the listener read in another schema in the
// internal form.
func translate(path string, batch systemoneprotocol.Batch) (ClientRequest, error) {
	internal := batch
	if internal.Form == systemoneprotocol.FormRequests {
		internal.Form = systemoneprotocol.FormStates
	}
	body, err := InternalCodec{}.EncodeRequest(internal)
	if err != nil {
		return ClientRequest{}, err
	}
	return ClientRequest{Path: path, Batch: batch, Legacy: LegacyFormsOf(batch), Plan: PlanTranslate, Internal: body}, nil
}

func (r *ClientRequest) diagnose() {
	for _, form := range r.Legacy {
		r.Diagnostics = append(r.Diagnostics, Diagnostic{Field: form, Action: DiagnosticLegacy, Reason: "a vllm-sr/4.0 form"})
	}
	action := DiagnosticTranslated
	if r.Plan == PlanPassThrough {
		action = DiagnosticPassedThrough
	}
	r.Diagnostics = append(r.Diagnostics, Diagnostic{Field: "request", Action: action})
}

// Listener40 is a vllm-sr/4.0 listener: a compatible superset of 4.0 and
// 5.0. It reads every form today's runtime contract 4.0.0 accepts and every
// vllm-sr/5.0 form, on the same paths and with no header. A request the
// internal form reads (every 4.0 form, and every 5.0 form on /v1/systemone)
// goes on as its client sent it, after it is decoded and checked; a batch
// on /v1/systemone/batch, and a request with the standard's long-context
// header, is translated through the IR. An object state without roles keeps
// 4.0's implicit roles, and an asset is read where its placeholder is.
// Answers keep the 4.0 shape; an error carries both 4.0's error and the
// standard's detail with the flat 5.0 fields, so either kind of client reads
// it.
type Listener40 struct{}

func (Listener40) ID() SchemaID { return SchemaVLLMSR40 }

func (Listener40) Read(path string, body []byte, header http.Header) (ClientRequest, error) {
	var read ClientRequest
	var err error
	switch path {
	case PathBatch:
		var batch systemoneprotocol.Batch
		batch, err = VLLMSR50Codec{}.DecodeBatch(body, header, ReadRules{Roles: systemoneprotocol.RolesImplicit})
		if err == nil {
			read, err = translate(path, batch)
		}
	default:
		read, err = readNative(path, body, header)
	}
	if err != nil {
		return ClientRequest{}, err
	}
	read.diagnose()
	return read, nil
}

func (Listener40) AnswerSuccess(request *ClientRequest, body []byte) ([]byte, error) {
	return answerBatch(request, body)
}

func (Listener40) AnswerError(_ *ClientRequest, _ string, status int, body []byte) (int, []byte) {
	native, fields, ok := nativeError(body)
	if !ok {
		return status, body
	}
	answer := systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "error", Value: native},
		systemoneprotocol.Member{Key: "detail", Value: systemoneprotocol.String(fields.message)},
	)
	answer = withFlatError(answer, fields)
	encoded, err := json.Marshal(answer)
	if err != nil {
		return status, body
	}
	return status, encoded
}

func (Listener40) AnswerListing(body []byte) ([]byte, error) { return body, nil }

// Listener50 is a vllm-sr/5.0 listener, the default: the standard plus the
// vllm-sr extensions. A request is decoded strictly, any member 5.0 does not
// define refused with its name, and translated through the IR: an object
// state without roles gives no part a role, and media placed in front of the
// state is read there. A batch is POST /v1/systemone/batch. An error is the
// standard detail (a list on a 422) with the flat code, question, state,
// reason and limit, and an overloaded server answers 529. The listing's
// models carry version, features and card. /v1/decisions stays native.
type Listener50 struct{}

func (Listener50) ID() SchemaID { return SchemaVLLMSR50 }

func (Listener50) Read(path string, body []byte, header http.Header) (ClientRequest, error) {
	var read ClientRequest
	var err error
	switch path {
	case PathDecisions:
		read, err = readNative(path, body, header)
	case PathBatch:
		var batch systemoneprotocol.Batch
		if batch, err = (VLLMSR50Codec{}).DecodeBatch(body, header, Rules50); err == nil {
			read, err = translate(path, batch)
		}
	default:
		var batch systemoneprotocol.Batch
		if batch, err = (VLLMSR50Codec{}).DecodeRequest(body, header, Rules50); err == nil {
			read, err = translate(path, batch)
		}
	}
	if err != nil {
		return ClientRequest{}, err
	}
	read.diagnose()
	return read, nil
}

func (Listener50) AnswerSuccess(request *ClientRequest, body []byte) ([]byte, error) {
	return answerBatch(request, body)
}

func (Listener50) AnswerError(request *ClientRequest, path string, status int, body []byte) (int, []byte) {
	if path == PathDecisions {
		return Listener40{}.AnswerError(request, path, status, body)
	}
	_, fields, ok := nativeError(body)
	if !ok {
		return status, body
	}
	if fields.code == "overloaded" {
		status = StatusOverloaded
	}
	detail := systemoneprotocol.String(fields.message)
	if status == http.StatusUnprocessableEntity {
		location := []systemoneprotocol.Value{systemoneprotocol.String("body")}
		if fields.state != "" && request != nil && request.Path == PathBatch {
			location = append(location, systemoneprotocol.String("requests"), systemoneprotocol.String(fields.state))
		}
		if fields.question != "" {
			location = append(location, systemoneprotocol.String("questions"), systemoneprotocol.String(fields.question))
		}
		kind := firstNonEmpty(fields.questionError, fields.code)
		detail = systemoneprotocol.Array(systemoneprotocol.Object(
			systemoneprotocol.Member{Key: "loc", Value: systemoneprotocol.Array(location...)},
			systemoneprotocol.Member{Key: "msg", Value: systemoneprotocol.String(fields.message)},
			systemoneprotocol.Member{Key: "type", Value: systemoneprotocol.String(kind)},
		))
	}
	answer := withFlatError(systemoneprotocol.Object(systemoneprotocol.Member{Key: "detail", Value: detail}), fields)
	if calls, ok := fields.native.Get("model_calls"); ok {
		answer = answer.With("model_calls", calls)
	}
	encoded, err := json.Marshal(answer)
	if err != nil {
		return status, body
	}
	return status, encoded
}

// StatusOverloaded is the standard's status of an overloaded server.
const StatusOverloaded = 529

// AnswerListing writes the standard listing: each System One model's name,
// description and release date, with the extension set's version, its
// features (the question types it answers) and its card.
func (Listener50) AnswerListing(body []byte) ([]byte, error) {
	value, err := systemoneprotocol.Parse(body)
	if err != nil || value.Kind() != systemoneprotocol.KindObject {
		return body, nil
	}
	data, _ := value.Get("data")
	models := make([]systemoneprotocol.Value, 0, len(data.Items()))
	for _, entry := range data.Items() {
		model := systemoneprotocol.Object(
			systemoneprotocol.Member{Key: "name", Value: mustGet(entry, "id")},
			systemoneprotocol.Member{Key: "description", Value: systemoneprotocol.String("System One decision model")},
			systemoneprotocol.Member{Key: "release_date", Value: systemoneprotocol.String("")},
			systemoneprotocol.Member{Key: "version", Value: systemoneprotocol.String(systemoneprotocol.ExtensionSetVersion)},
		)
		if types, ok := entry.Get("question_types"); ok {
			model = model.With("features", types)
		}
		models = append(models, model.With("card", entry))
	}
	return json.Marshal(systemoneprotocol.Object(systemoneprotocol.Member{Key: "models", Value: systemoneprotocol.Array(models...)}))
}

// answerBatch writes a batch endpoint's answer: the internal form's entries
// under states as the named responses under responses. Any other answer is
// the client's as it is.
func answerBatch(request *ClientRequest, body []byte) ([]byte, error) {
	if request == nil || request.Batch.Form != systemoneprotocol.FormRequests {
		return body, nil
	}
	names := make([]string, len(request.Batch.Items))
	for index, item := range request.Batch.Items {
		names[index] = item.Name
	}
	results, model, err := InternalCodec{}.DecodeResponse(systemoneprotocol.FormStates, names, body)
	if err != nil {
		return nil, err
	}
	value, _ := systemoneprotocol.Parse(body)
	responses := systemoneprotocol.Object()
	for index, result := range results {
		if result.Err != nil {
			return nil, result.Err
		}
		responses = responses.With(names[index], responseValue(result.Response, model))
	}
	answer := systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String(model)},
		systemoneprotocol.Member{Key: "responses", Value: responses},
	)
	for _, member := range value.Members() {
		switch member.Key {
		case "model", "states":
		default:
			answer = answer.With(member.Key, member.Value)
		}
	}
	return json.Marshal(answer)
}

// errorFields are an internal error's fields.
type errorFields struct {
	native                                                systemoneprotocol.Value
	code, message, question, state, reason, questionError string
	limit                                                 systemoneprotocol.Value
}

// nativeError reads an internal error body, {error: {...}}.
func nativeError(body []byte) (systemoneprotocol.Value, errorFields, bool) {
	value, err := systemoneprotocol.Parse(body)
	if err != nil || value.Kind() != systemoneprotocol.KindObject {
		return systemoneprotocol.Value{}, errorFields{}, false
	}
	native, ok := value.Get("error")
	if !ok || native.Kind() != systemoneprotocol.KindObject {
		return systemoneprotocol.Value{}, errorFields{}, false
	}
	fields := errorFields{
		native: native, code: textOf(mustGet(native, "code")), message: textOf(mustGet(native, "message")),
		question: textOf(mustGet(native, "question")), state: textOf(mustGet(native, "state")),
		reason: textOf(mustGet(native, "reason")), questionError: textOf(mustGet(native, "question_error")),
		limit: mustGet(native, "limit"),
	}
	if fields.message == "" {
		fields.message = strings.ReplaceAll(fields.code, "_", " ")
	}
	return native, fields, true
}

// withFlatError adds the flat 5.0 error fields to an error answer.
func withFlatError(answer systemoneprotocol.Value, fields errorFields) systemoneprotocol.Value {
	if fields.code != "" {
		answer = answer.With("code", systemoneprotocol.String(fields.code))
	}
	if fields.question != "" {
		answer = answer.With("question", systemoneprotocol.String(fields.question))
	}
	if fields.state != "" {
		answer = answer.With("state", systemoneprotocol.String(fields.state))
	}
	if fields.reason != "" {
		answer = answer.With("reason", systemoneprotocol.String(fields.reason))
	}
	if fields.limit.Kind() == systemoneprotocol.KindNumber {
		answer = answer.With("limit", fields.limit)
	}
	return answer
}
