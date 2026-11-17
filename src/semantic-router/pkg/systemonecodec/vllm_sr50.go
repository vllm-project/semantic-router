package systemonecodec

import (
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// HeaderLongContext is the standard's request header that lets a request read
// a long context; it replaces vllm-sr/4.0's options.long_context.
const HeaderLongContext = "TypeSafe-Danger-Zone-Long-Context"

// BatchPath is the vllm-sr/5.0 batch endpoint after the request endpoint's
// path: POST /v1/systemone/batch.
const BatchPath = "/batch"

// VLLMSR50Codec speaks vllm-sr/5.0: the standard plus the vllm-sr extension
// set 5.0 (config/schemas/systemone/vllm-sr-5.0.extensions.yaml), which
// vllm-sr serve speaks. A request is a standard request with the extension
// fields flat in their places: explicit roles, the options, and each
// question's standard type, instructions and criteria beside its hints
// (preset, head, over, ...). A batch goes in one call to the batch endpoint
// as named standard requests. Images and videos go as assets whose
// placeholders lead the state, which a vllm-sr server reads in front of it,
// as 4.0 read its images. An error is the standard detail with flat code,
// question, state, reason and limit, and the listing's models carry their
// cards.
type VLLMSR50Codec struct{}

func (VLLMSR50Codec) ID() SchemaID { return SchemaVLLMSR50 }

func (VLLMSR50Codec) Spec() SpecRef {
	spec, _ := VendoredSpec(SchemaVLLMSR50)
	return spec
}

func (VLLMSR50Codec) Capabilities() CapabilitySet {
	return Capabilities(CapabilityBatch, CapabilityObjectState, CapabilityDeadline, CapabilityPresets, CapabilitySetQuestions,
		CapabilitySpan, CapabilityFullInput, CapabilityImages, CapabilityVideos, CapabilityCard, CapabilityQuestionErrors)
}

func (VLLMSR50Codec) Stateless() bool { return true }

// ListingPath replaces the endpoint's last segment: POST /v1/systemone lists
// at GET /v1/models.
func (VLLMSR50Codec) ListingPath() string { return "models" }

// Plan sends one request to the endpoint and several, as named standard
// requests, to its batch endpoint.
func (c VLLMSR50Codec) Plan(batch systemoneprotocol.Batch, target Target, policy Policy) (Plan, Diagnostics, error) {
	plan := Plan{Names: make([]string, len(batch.Items))}
	header := http.Header{"Content-Type": {"application/json"}}
	if batch.Options.LongContext {
		header.Set(HeaderLongContext, "true")
	}
	var diagnostics Diagnostics
	if len(batch.Options.Extra) > 0 {
		return Plan{}, diagnostics, fmt.Errorf("options.%s is not an option of vllm-sr/5.0", batch.Options.Extra[0].Key)
	}
	body := systemoneprotocol.Object(systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String(firstNonEmpty(batch.Model, target.Model))})
	items := make([]int, len(batch.Items))
	requests := systemoneprotocol.Object()
	for index, item := range batch.Items {
		plan.Names[index], items[index] = item.Name, index
		members, err := c.requestMembers(item.Request, policy, &diagnostics)
		if err != nil {
			var problem *systemoneprotocol.Problem
			if errors.As(err, &problem) {
				problem.Item = item.Name
				return Plan{}, diagnostics, problem
			}
			if len(batch.Items) == 1 {
				return Plan{}, diagnostics, err
			}
			return Plan{}, diagnostics, fmt.Errorf("request %q: %w", item.Name, err)
		}
		if len(batch.Items) == 1 {
			for _, member := range members {
				body = body.With(member.Key, member.Value)
			}
			continue
		}
		requests = requests.With(item.Name, systemoneprotocol.Object(members...))
	}
	call := WireCall{Header: header, Items: items}
	if len(batch.Items) > 1 {
		body = body.With("requests", requests)
		call.Path, call.Batch = BatchPath, true
	}
	if options := encodeOptions(batch.Options, false); options.Kind() == systemoneprotocol.KindObject && len(options.Members()) > 0 {
		body = body.With("options", options)
	}
	call.Body = body
	plan.Calls = []WireCall{call}
	return plan, diagnostics, nil
}

// requestMembers are one request's members in vllm-sr/5.0 without its
// model: the state (with any images and videos placed in front of it), the
// questions, the assets and the roles.
func (VLLMSR50Codec) requestMembers(request systemoneprotocol.Request, policy Policy, diagnostics *Diagnostics) ([]systemoneprotocol.Member, error) {
	images, videos, err := mediaOf(request.Extensions)
	if err != nil {
		return nil, err
	}
	state, assets, err := placeMedia(request.State, images, videos, request.Assets)
	if err != nil {
		return nil, err
	}
	if len(images)+len(videos) > 0 {
		*diagnostics = append(*diagnostics, Diagnostic{Field: "images", Action: DiagnosticConverted, Reason: "images and videos go as assets that lead the state"})
	}
	for _, member := range request.Extensions {
		if member.Key != "images" && member.Key != "videos" {
			return nil, fmt.Errorf("vllm-sr/5.0 has no request member %q", member.Key)
		}
	}
	questions := systemoneprotocol.Object()
	for _, named := range request.Questions {
		question, err := fiveQuestion(named.Question, policy)
		var problem *systemoneprotocol.Problem
		if errors.As(err, &problem) {
			problem.Question = named.Name
			return nil, problem
		}
		if err != nil {
			return nil, fmt.Errorf("question %q: %w", named.Name, err)
		}
		questions = questions.With(named.Name, question)
	}
	members := []systemoneprotocol.Member{{Key: "state", Value: state}, {Key: "questions", Value: questions}}
	if len(assets) > 0 {
		members = append(members, systemoneprotocol.Member{Key: "assets", Value: systemoneprotocol.Object(assets...)})
	}
	switch request.Roles.Mode {
	case systemoneprotocol.RolesExplicit:
		members = append(members, systemoneprotocol.Member{Key: "roles", Value: systemoneprotocol.Object(request.Roles.Map...)})
	case systemoneprotocol.RolesImplicit:
		if roles := systemoneprotocol.ImplicitRoles(request.State); len(roles) > 0 {
			members = append(members, systemoneprotocol.Member{Key: "roles", Value: systemoneprotocol.Object(roles...)})
			*diagnostics = append(*diagnostics, Diagnostic{Field: "roles", Action: DiagnosticConverted, Reason: "the state keys' implicit roles go as explicit roles"})
		}
	}
	return members, nil
}

// fiveQuestion is a question in vllm-sr/5.0: its standard form, a preset
// question with the instructions and criteria of the Router's canonical
// question for it beside its preset hint, and only fields a schema defines.
func fiveQuestion(question systemoneprotocol.Question, policy Policy) (systemoneprotocol.Value, error) {
	question, err := question.Standard()
	if err != nil {
		return systemoneprotocol.Value{}, err
	}
	if preset, ok := question.Extension("preset"); ok && question.Instructions.IsNull() && question.Criteria.IsNull() {
		name, _ := preset.Text()
		expanded, known := systemoneprotocol.Question{}, false
		if policy.Presets != nil {
			expanded, known = policy.Presets(name)
		}
		if !known {
			return systemoneprotocol.Value{}, &systemoneprotocol.Problem{
				Status: http.StatusUnprocessableEntity, Kind: systemoneprotocol.ProblemRejected,
				QuestionError: "invalid_question", Reason: "preset_unknown",
				Detail: fmt.Sprintf("a preset question goes to vllm-sr/5.0 with its instructions and criteria, and the Router has none for preset %q", name),
			}
		}
		if question.Type != "" && question.Type != expanded.Type {
			return systemoneprotocol.Value{}, fmt.Errorf("preset %q is a %s question, not %s", name, expanded.Type, question.Type)
		}
		question.Type, question.Instructions, question.Criteria = expanded.Type, expanded.Instructions, expanded.Criteria
	}
	if question.Type == "" {
		return systemoneprotocol.Value{}, fmt.Errorf("a question needs a type")
	}
	for _, member := range question.Extensions {
		if _, ok := systemoneprotocol.ExtensionField(systemoneprotocol.LevelQuestion, member.Key); !ok {
			return systemoneprotocol.Value{}, fmt.Errorf("question field %q is defined by no schema", member.Key)
		}
	}
	return question.QuestionValue(), nil
}

// DecodeReply reads a vllm-sr/5.0 reply: the response of a request alone, a
// batch's responses by name, or the error that answers every item the call
// carried.
func (VLLMSR50Codec) DecodeReply(plan Plan, call WireCall, reply WireReply) ([]ItemResult, error) {
	results := make([]ItemResult, len(call.Items))
	if reply.Status < 200 || reply.Status >= 300 {
		problem := FiveProblem(reply, time.Now())
		if !call.Batch && len(call.Items) == 1 && call.Items[0] < len(plan.Names) && problem.Item == "" {
			problem.Item = plan.Names[call.Items[0]]
		}
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
	responses, _ := value.Get("responses")
	model := textOf(mustGet(value, "model"))
	for index, item := range call.Items {
		entry, ok := responses.Get(plan.Names[item])
		if !ok {
			results[index].Err = &systemoneprotocol.Problem{Status: reply.Status, Kind: systemoneprotocol.ProblemFailed, Detail: fmt.Sprintf("the batch response has no response for %q", plan.Names[item])}
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

// FiveProblem reads a vllm-sr/5.0 error reply: the standard's detail (with
// its Retry-After), and the flat code, question, state, reason and limit. A
// failed question's error is the one its reason belongs to.
func FiveProblem(reply WireReply, now time.Time) *systemoneprotocol.Problem {
	problem := standardProblem(reply, now)
	value, err := systemoneprotocol.Parse(reply.Body)
	if err != nil || value.Kind() != systemoneprotocol.KindObject {
		return problem
	}
	if code := textOf(mustGet(value, "code")); code != "" {
		problem.Code = code
	}
	if reason := textOf(mustGet(value, "reason")); reason != "" {
		problem.Reason = reason
	}
	if question := textOf(mustGet(value, "question")); question != "" {
		problem.Question = question
	}
	if state := textOf(mustGet(value, "state")); state != "" {
		problem.Item = state
	}
	if limit, ok := mustGet(value, "limit").Float64(); ok {
		whole := int64(limit)
		problem.Limit = &whole
	}
	if problem.Question != "" && (problem.QuestionError == "" || problem.Reason != "") {
		problem.QuestionError = QuestionError(problem.Reason, reply.Status)
	}
	return problem
}

// questionReasons are the model runtime's question errors by their fixed
// reasons (vllm_srun/errors.py QUESTION_REASONS): each reason belongs to one.
var questionReasons = map[string]string{ //nolint:gosec // reason words, not credentials
	"input_malformed":    "invalid_input",
	"question_tokens":    "max_length_exceeded",
	"require_full_input": "max_length_exceeded",
	"scan_budget":        "scan_budget_exceeded",
	"model_output":       "invalid_model_output",
	"deadline":           "deadline_exceeded",
	"model_unavailable":  "unavailable",
}

// QuestionError is the error of a failed question with a reason, or by its
// status without a known one.
func QuestionError(reason string, status int) string {
	if code, ok := questionReasons[reason]; ok {
		return code
	}
	switch status {
	case http.StatusInternalServerError:
		return "invalid_model_output"
	case http.StatusServiceUnavailable:
		return "unavailable"
	case http.StatusGatewayTimeout:
		return "deadline_exceeded"
	}
	return "invalid_question"
}

// QuestionReason is the reason a failed question's error gives when the
// server names none.
func QuestionReason(questionError string) string {
	for reason, code := range questionReasons {
		if code == questionError && reason != "require_full_input" {
			return reason
		}
	}
	return "question_malformed"
}

// DecodeListing reads {models: [{name, description, release_date, version,
// features, card}]}.
func (VLLMSR50Codec) DecodeListing(status int, body []byte) (systemoneprotocol.Listing, error) {
	return SystemOne02Codec{}.DecodeListing(status, body)
}

// ListedCard is a listed model's card: the card member of a vllm-sr/5.0
// models entry, or a vllm-sr/4.0 data entry.
func ListedCard(model systemoneprotocol.ListedModel) (systemoneprotocol.Value, bool) {
	for _, member := range model.Extensions {
		if member.Key == "card" && member.Value.Kind() == systemoneprotocol.KindObject {
			return member.Value, true
		}
	}
	return systemoneprotocol.Value{}, false
}

// ListedCardJSON is ListedCard as JSON, for a caller that decodes it into its
// own card type.
func ListedCardJSON(model systemoneprotocol.ListedModel) (json.RawMessage, bool) {
	card, ok := ListedCard(model)
	if !ok {
		return nil, false
	}
	data, err := json.Marshal(card)
	return data, err == nil
}
