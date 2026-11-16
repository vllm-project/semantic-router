package systemonecodec

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// headerImages is the standard's request header for a request with assets.
const headerImages = "TypeSafe-Danger-Zone-Images"

// SystemOne02Codec speaks systemone/0.2, the published standard: one request
// per call, a text, object or array state, the standard question types
// (choice, noul, score, box) with only type, instructions and criteria,
// images as assets, and no options. A batch is split into one call per
// request, and an error is a detail.
type SystemOne02Codec struct{}

func (SystemOne02Codec) ID() SchemaID { return SchemaSystemOne02 }

func (SystemOne02Codec) Spec() SpecRef {
	spec, _ := VendoredSpec(SchemaSystemOne02)
	return spec
}

func (SystemOne02Codec) Capabilities() CapabilitySet {
	return Capabilities(CapabilityObjectState, CapabilityImages)
}

func (SystemOne02Codec) Stateless() bool { return true }

// ListingPath replaces the endpoint's last segment: POST /v1/systemone lists
// at GET /v1/models.
func (SystemOne02Codec) ListingPath() string { return "models" }

// Plan sends each request of the batch in a call of its own.
func (c SystemOne02Codec) Plan(batch systemoneprotocol.Batch, target Target, policy Policy) (Plan, Diagnostics, error) {
	plan := Plan{Names: make([]string, len(batch.Items))}
	var diagnostics Diagnostics
	if len(batch.Items) > 1 {
		diagnostics = append(diagnostics, Diagnostic{Field: "batch", Action: DiagnosticSplit, Reason: "systemone/0.2 takes one request per call"})
	}
	if !batch.Options.DeadlineMs.IsNull() || batch.Options.ReturnMeta != nil || !batch.Options.MaxTokens.IsNull() {
		diagnostics = append(diagnostics, Diagnostic{Field: "options", Action: DiagnosticStripped, Reason: "systemone/0.2 has no options; the client's timeout bounds the call"})
	}
	for index, item := range batch.Items {
		plan.Names[index] = item.Name
		model := item.Request.Model
		if model == "" {
			model = batch.Model
		}
		if model == "" {
			model = target.Model
		}
		body, header, itemDiagnostics, err := c.encodeRequest(item.Request, model, policy)
		if err != nil {
			return Plan{}, diagnostics, fmt.Errorf("request %q: %w", item.Name, err)
		}
		diagnostics = append(diagnostics, itemDiagnostics...)
		if batch.Options.LongContext {
			header.Set(HeaderLongContext, "true")
		}
		plan.Calls = append(plan.Calls, WireCall{Header: header, Body: body, Items: []int{index}})
	}
	return plan, diagnostics, nil
}

func (c SystemOne02Codec) encodeRequest(request systemoneprotocol.Request, model string, policy Policy) (systemoneprotocol.Value, http.Header, Diagnostics, error) {
	var diagnostics Diagnostics
	header := http.Header{"Content-Type": {"application/json"}}
	if request.Roles.Mode == systemoneprotocol.RolesExplicit {
		if !policy.StripUnsupported {
			return systemoneprotocol.Value{}, nil, nil, fmt.Errorf("systemone/0.2 has no roles")
		}
		diagnostics = append(diagnostics, Diagnostic{Field: "roles", Action: DiagnosticStripped, Reason: "the state keys name the parts"})
	}
	images, videos, err := mediaOf(request.Extensions)
	if err != nil {
		return systemoneprotocol.Value{}, nil, nil, err
	}
	if len(videos) > 0 {
		return systemoneprotocol.Value{}, nil, nil, fmt.Errorf("systemone/0.2 carries no video")
	}
	for _, member := range request.Extensions {
		if member.Key == "images" || member.Key == "videos" {
			continue
		}
		if extensionErr := c.extension(systemoneprotocol.LevelRequest, member.Key, policy, &diagnostics); extensionErr != nil {
			return systemoneprotocol.Value{}, nil, nil, extensionErr
		}
	}
	state, assets, err := placeMedia(request.State, images, nil, request.Assets)
	if err != nil {
		return systemoneprotocol.Value{}, nil, nil, err
	}
	if len(images) > 0 {
		diagnostics = append(diagnostics, Diagnostic{Field: "images", Action: DiagnosticConverted, Reason: "images go as assets that lead the state"})
	}
	questions := systemoneprotocol.Object()
	for _, named := range request.Questions {
		question, err := c.encodeQuestion(named.Question, policy, &diagnostics)
		if err != nil {
			return systemoneprotocol.Value{}, nil, nil, fmt.Errorf("question %q: %w", named.Name, err)
		}
		questions = questions.With(named.Name, question)
	}
	body := systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String(model)},
		systemoneprotocol.Member{Key: "state", Value: state},
		systemoneprotocol.Member{Key: "questions", Value: questions},
	)
	if len(assets) > 0 {
		for _, asset := range assets {
			if kind, _ := asset.Value.Get("type"); textOf(kind) == "video" {
				return systemoneprotocol.Value{}, nil, nil, fmt.Errorf("systemone/0.2 carries no video")
			}
		}
		body = body.With("assets", systemoneprotocol.Object(assets...))
		header.Set(headerImages, "true")
	}
	return body, header, diagnostics, nil
}

func (c SystemOne02Codec) encodeQuestion(question systemoneprotocol.Question, policy Policy, diagnostics *Diagnostics) (systemoneprotocol.Value, error) {
	question, err := question.Standard()
	if err != nil {
		return systemoneprotocol.Value{}, err
	}
	if preset, ok := question.Extension("preset"); ok && question.Instructions.IsNull() && question.Criteria.IsNull() && policy.Presets != nil {
		name, _ := preset.Text()
		if expanded, known := policy.Presets(name); known && (question.Type == "" || question.Type == expanded.Type) {
			question.Type, question.Instructions, question.Criteria = expanded.Type, expanded.Instructions, expanded.Criteria
		}
	}
	switch question.Type {
	case systemoneprotocol.QuestionChoice, systemoneprotocol.QuestionNoul, systemoneprotocol.QuestionScore, systemoneprotocol.QuestionBox:
	default:
		if field, ok := systemoneprotocol.ExtensionField(systemoneprotocol.LevelQuestionType, question.Type); ok && field.StandardTarget == systemoneprotocol.TargetRefuse {
			return systemoneprotocol.Value{}, fmt.Errorf("systemone/0.2 has no %s questions", question.Type)
		}
		return systemoneprotocol.Value{}, fmt.Errorf("question type %q is defined by no schema", question.Type)
	}
	if question.Instructions.IsNull() && question.Criteria.IsNull() {
		if _, preset := question.Extension("preset"); preset {
			return systemoneprotocol.Value{}, fmt.Errorf("a preset question goes to systemone/0.2 only with its instructions and criteria")
		}
	}
	object := systemoneprotocol.Object(systemoneprotocol.Member{Key: "type", Value: systemoneprotocol.String(question.Type)})
	if !question.Instructions.IsNull() {
		object = object.With("instructions", question.Instructions)
	}
	if !question.Criteria.IsNull() {
		object = object.With("criteria", question.Criteria)
	}
	for _, member := range question.Extensions {
		if err := c.extension(systemoneprotocol.LevelQuestion, member.Key, policy, diagnostics); err != nil {
			return systemoneprotocol.Value{}, err
		}
	}
	return object, nil
}

// extension applies the extension spec's standard_target to a field the
// standard does not define: strip it, or refuse it.
func (SystemOne02Codec) extension(level systemoneprotocol.Level, name string, policy Policy, diagnostics *Diagnostics) error {
	field, ok := systemoneprotocol.ExtensionField(level, name)
	if !ok {
		return fmt.Errorf("%s field %q is defined by no schema", level, name)
	}
	if field.StandardTarget != systemoneprotocol.TargetStrip || !policy.StripUnsupported {
		return fmt.Errorf("systemone/0.2 cannot carry %s (%s)", name, field.Summary)
	}
	*diagnostics = append(*diagnostics, Diagnostic{Field: string(level) + "." + name, Action: DiagnosticStripped, Reason: "systemone/0.2 has no " + name})
	return nil
}

// DecodeReply reads a standard reply: the response of the call's one
// request, or the error that answers it.
func (SystemOne02Codec) DecodeReply(plan Plan, call WireCall, reply WireReply) ([]ItemResult, error) {
	results := make([]ItemResult, len(call.Items))
	if reply.Status >= 200 && reply.Status < 300 {
		response, err := decodeStandardResponse(reply.Body)
		for index := range results {
			results[index] = ItemResult{Response: response, Err: err}
		}
		return results, nil
	}
	problem := standardProblem(reply, time.Now())
	if len(call.Items) == 1 && call.Items[0] < len(plan.Names) {
		problem.Item = plan.Names[call.Items[0]]
	}
	for index := range results {
		results[index].Err = problem
	}
	return results, nil
}

// decodeStandardResponse reads {model, answers, usage} and keeps every other
// member as an extension.
func decodeStandardResponse(body []byte) (systemoneprotocol.Response, error) {
	value, err := systemoneprotocol.Parse(body)
	if err != nil || value.Kind() != systemoneprotocol.KindObject {
		return systemoneprotocol.Response{}, &systemoneprotocol.Problem{Status: http.StatusOK, Kind: systemoneprotocol.ProblemFailed, Detail: "the response is not a JSON object"}
	}
	return responseFromValue(value)
}

func responseFromValue(value systemoneprotocol.Value) (systemoneprotocol.Response, error) {
	answers, ok := value.Get("answers")
	if !ok || answers.Kind() != systemoneprotocol.KindObject {
		return systemoneprotocol.Response{}, &systemoneprotocol.Problem{Status: http.StatusOK, Kind: systemoneprotocol.ProblemFailed, Detail: "the response has no answers"}
	}
	response := systemoneprotocol.Response{Answers: append([]systemoneprotocol.Member{}, answers.Members()...)}
	for _, member := range value.Members() {
		switch member.Key {
		case "model":
			response.Model = textOf(member.Value)
		case "answers":
		case "usage":
			response.Usage = usageOf(member.Value)
		default:
			response.Extensions = append(response.Extensions, member)
		}
	}
	return response, nil
}

func usageOf(value systemoneprotocol.Value) systemoneprotocol.Usage {
	usage := systemoneprotocol.Usage{}
	if input, ok := value.Get("input_tokens"); ok {
		if number, ok := input.Float64(); ok {
			usage.InputTokens = int64(number)
		}
	}
	if output, ok := value.Get("output_tokens"); ok {
		if number, ok := output.Float64(); ok {
			usage.OutputTokens = int64(number)
		}
	}
	return usage
}

// standardProblem reads a standard error reply: 400 {detail}, 422 {detail:
// [{loc, msg, type}]} (whose loc may name a question), 429 and 529 with
// Retry-After, and every other status by its class.
func standardProblem(reply WireReply, now time.Time) *systemoneprotocol.Problem {
	problem := &systemoneprotocol.Problem{Status: reply.Status, Kind: problemKind(reply.Status)}
	if wait, ok := RetryAfter(reply.Header, now); ok {
		problem.RetryAfter = wait
	}
	value, err := systemoneprotocol.Parse(reply.Body)
	if err != nil {
		problem.Detail = http.StatusText(reply.Status)
		return problem
	}
	detail, _ := value.Get("detail")
	switch detail.Kind() {
	case systemoneprotocol.KindString:
		problem.Detail = textOf(detail)
	case systemoneprotocol.KindObject:
		problem.Detail = systemoneprotocol.Canonical(detail)
	case systemoneprotocol.KindArray:
		var messages []string
		for _, issue := range detail.Items() {
			message := textOf(mustGet(issue, "msg"))
			location := locationOf(mustGet(issue, "loc"))
			if problem.Question == "" && len(location) >= 3 && location[0] == "body" && location[1] == "questions" {
				problem.Question, problem.QuestionError = location[2], "invalid_question"
			}
			messages = append(messages, strings.Join(location, ".")+": "+message)
		}
		problem.Detail = strings.Join(messages, "; ")
	}
	if code, ok := value.Get("code"); ok {
		problem.Code = textOf(code)
	}
	return problem
}

func problemKind(status int) systemoneprotocol.ProblemKind {
	switch {
	case status == http.StatusUnauthorized || status == http.StatusForbidden:
		return systemoneprotocol.ProblemCredential
	case status == http.StatusTooManyRequests || status == 529:
		return systemoneprotocol.ProblemOverloaded
	case status == http.StatusServiceUnavailable:
		return systemoneprotocol.ProblemUnavailable
	case status == http.StatusRequestTimeout || status == http.StatusGatewayTimeout:
		return systemoneprotocol.ProblemDeadline
	case status >= 400 && status < 500:
		return systemoneprotocol.ProblemRejected
	}
	return systemoneprotocol.ProblemFailed
}

func locationOf(value systemoneprotocol.Value) []string {
	var location []string
	for _, item := range value.Items() {
		if text, ok := item.Text(); ok {
			location = append(location, text)
		} else if literal, ok := item.Literal(); ok {
			location = append(location, literal)
		}
	}
	return location
}

func mustGet(value systemoneprotocol.Value, key string) systemoneprotocol.Value {
	member, _ := value.Get(key)
	return member
}

func textOf(value systemoneprotocol.Value) string {
	text, _ := value.Text()
	return text
}

// DecodeListing reads {models: [{name, description, release_date}]}.
func (SystemOne02Codec) DecodeListing(status int, body []byte) (systemoneprotocol.Listing, error) {
	if status != http.StatusOK {
		return systemoneprotocol.Listing{}, fmt.Errorf("the models listing answered %d", status)
	}
	var listing struct {
		Models []json.RawMessage `json:"models"`
	}
	if err := json.Unmarshal(body, &listing); err != nil || listing.Models == nil {
		return systemoneprotocol.Listing{}, fmt.Errorf("the models listing has no models")
	}
	decoded := systemoneprotocol.Listing{}
	for _, raw := range listing.Models {
		entry, err := systemoneprotocol.Parse(raw)
		if err != nil || entry.Kind() != systemoneprotocol.KindObject {
			return systemoneprotocol.Listing{}, fmt.Errorf("a models entry is not an object")
		}
		model := systemoneprotocol.ListedModel{}
		for _, member := range entry.Members() {
			switch member.Key {
			case "name":
				model.Name = textOf(member.Value)
			case "description":
				model.Description = textOf(member.Value)
			case "release_date":
				model.ReleaseDate = textOf(member.Value)
			default:
				model.Extensions = append(model.Extensions, member)
			}
		}
		decoded.Models = append(decoded.Models, model)
	}
	return decoded, nil
}
