package systemonecodec

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// The members the internal form takes at each level.
var (
	internalAloneMembers    = memberSet("model", "state", "questions", "images", "videos", "assets", "options", "roles")
	internalBatchMembers    = memberSet("model", "states", "options", "assets")
	internalItemMembers     = memberSet("state", "questions", "images", "videos", "roles")
	internalQuestionMembers = memberSet("type", "instructions", "criteria", "choices", "levels", "labels", "at", "rate",
		"over", "require_full_input", "overflow", "head", "threshold", "preset", "decision_adapter")
	internalOptionMembers = memberSet("deadline_ms", "return_meta", "max_tokens", "long_context")
	assetMembers          = memberSet("type", "format", "data_base64", "part")
)

func memberSet(names ...string) map[string]bool {
	set := make(map[string]bool, len(names))
	for _, name := range names {
		set[name] = true
	}
	return set
}

// InternalCodec reads and writes the Router's internal request form, the
// wire encoding of the IR on the hops inside a deployment: the model
// runtime's /v1/bundle and /v1/decisions read it, and a vllm-sr/4.0
// listener's requests travel in it. It is vllm-sr/4.0's request form plus
// roles, where an absent roles is 4.0's implicit roles, null is none (the
// whole state is the user's content) and an object names them.
//
// It loses nothing: a request DecodeRequest reads, EncodeRequest writes back
// as Value writes the same JSON, with its members' order, its 4.0 spellings
// and its number literals.
type InternalCodec struct{}

// DecodeRequest reads an internal request, alone or a batch under states. It
// checks every member's name at every level and the shapes the IR needs,
// and leaves what a request must hold to its target, so a client of the
// model runtime gets the runtime's own answer.
func (c InternalCodec) DecodeRequest(body []byte) (systemoneprotocol.Batch, error) {
	value, err := systemoneprotocol.Parse(body)
	if err != nil {
		return systemoneprotocol.Batch{}, fmt.Errorf("the request is not valid JSON")
	}
	return c.DecodeRequestValue(value)
}

// DecodeRequestValue is DecodeRequest of a parsed request.
func (InternalCodec) DecodeRequestValue(value systemoneprotocol.Value) (systemoneprotocol.Batch, error) {
	if value.Kind() != systemoneprotocol.KindObject {
		return systemoneprotocol.Batch{}, fmt.Errorf("the request must be a JSON object")
	}
	batch := systemoneprotocol.Batch{}
	if model, ok := value.Get("model"); ok {
		text, isText := model.Text()
		if !isText {
			return systemoneprotocol.Batch{}, fmt.Errorf("model must be a string")
		}
		batch.Model = text
	}
	if options, ok := value.Get("options"); ok && options.IsNull() {
		batch.Options.Null = true
	} else if ok {
		decoded, err := decodeOptions(options, internalOptionMembers, "", true)
		if err != nil {
			return systemoneprotocol.Batch{}, err
		}
		batch.Options = decoded
	}
	states, isBatch := value.Get("states")
	if !isBatch {
		if err := checkMembers(value, internalAloneMembers, ""); err != nil {
			return systemoneprotocol.Batch{}, err
		}
		request, err := decodeInternalRequest(value, internalAloneMembers, "")
		if err != nil {
			return systemoneprotocol.Batch{}, err
		}
		request.Model = batch.Model
		batch.Items = []systemoneprotocol.NamedRequest{{Request: request}}
		return batch, nil
	}
	for _, member := range value.Members() {
		if !internalBatchMembers[member.Key] {
			return systemoneprotocol.Batch{}, fmt.Errorf("%s does not go with states: a batch takes model, states, options and assets, and each entry its own state and questions", member.Key)
		}
	}
	batch.Form, batch.Order = systemoneprotocol.FormStates, value.Keys()
	if assets, ok := value.Get("assets"); ok {
		if err := checkAssets(assets, "assets"); err != nil {
			return systemoneprotocol.Batch{}, err
		}
		batch.Shared = append(batch.Shared, systemoneprotocol.Member{Key: "assets", Value: assets})
	}
	if states.Kind() != systemoneprotocol.KindObject || len(states.Members()) == 0 {
		return systemoneprotocol.Batch{}, fmt.Errorf("states must be a nonempty object of named entries")
	}
	for _, entry := range states.Members() {
		if entry.Value.Kind() != systemoneprotocol.KindObject {
			return systemoneprotocol.Batch{}, fmt.Errorf("states.%s must be an object", entry.Key)
		}
		if err := checkMembers(entry.Value, internalItemMembers, "states."+entry.Key+"."); err != nil {
			return systemoneprotocol.Batch{}, err
		}
		request, err := decodeInternalRequest(entry.Value, internalItemMembers, "states."+entry.Key+".")
		if err != nil {
			return systemoneprotocol.Batch{}, err
		}
		batch.Items = append(batch.Items, systemoneprotocol.NamedRequest{Name: entry.Key, Request: request})
	}
	return batch, nil
}

// checkMembers refuses a member allowed does not name.
func checkMembers(value systemoneprotocol.Value, allowed map[string]bool, at string) error {
	for _, member := range value.Members() {
		if !allowed[member.Key] {
			return fmt.Errorf("%s%s is not a member of the request", at, member.Key)
		}
	}
	return nil
}

func decodeInternalRequest(value systemoneprotocol.Value, allowed map[string]bool, at string) (systemoneprotocol.Request, error) {
	request := systemoneprotocol.Request{Order: value.Keys()}
	for _, member := range value.Members() {
		switch member.Key {
		case "model", "options":
		case "state":
			request.State = member.Value
		case "questions":
			questions, err := decodeQuestions(member.Value, internalQuestionMembers, at)
			if err != nil {
				return request, err
			}
			request.Questions = questions
		case "assets":
			if err := checkAssets(member.Value, at+"assets"); err != nil {
				return request, err
			}
			request.Assets = append(request.Assets, member.Value.Members()...)
		case "roles":
			roles, err := decodeRoles(member.Value, at)
			if err != nil {
				return request, err
			}
			request.Roles = roles
		case "images", "videos":
			for index, item := range member.Value.Items() {
				if _, ok := item.Text(); !ok {
					return request, fmt.Errorf("%s%s[%d] must be a data URL string", at, member.Key, index)
				}
			}
			if member.Value.Kind() != systemoneprotocol.KindArray {
				return request, fmt.Errorf("%s%s must be an array of data URLs", at, member.Key)
			}
			request.Extensions = append(request.Extensions, member)
		default:
			if !allowed[member.Key] {
				return request, fmt.Errorf("%s%s is not a member of the request", at, member.Key)
			}
			request.Extensions = append(request.Extensions, member)
		}
	}
	return request, nil
}

func decodeQuestions(value systemoneprotocol.Value, allowed map[string]bool, at string) ([]systemoneprotocol.NamedQuestion, error) {
	if value.Kind() != systemoneprotocol.KindObject || (allowed == nil && len(value.Members()) == 0) {
		return nil, fmt.Errorf("%squestions must be a nonempty object of named questions", at)
	}
	questions := make([]systemoneprotocol.NamedQuestion, 0, len(value.Members()))
	for _, named := range value.Members() {
		question, err := systemoneprotocol.ParseQuestion(named.Value)
		if err != nil {
			return nil, fmt.Errorf("%squestions.%s: %w", at, named.Key, err)
		}
		for _, key := range question.Order {
			if allowed != nil && !allowed[key] {
				return nil, fmt.Errorf("%squestions.%s: %s is not a member of a question", at, named.Key, key)
			}
		}
		questions = append(questions, systemoneprotocol.NamedQuestion{Name: named.Key, Question: question})
	}
	return questions, nil
}

func decodeRoles(value systemoneprotocol.Value, at string) (systemoneprotocol.Roles, error) {
	switch value.Kind() {
	case systemoneprotocol.KindNull:
		return systemoneprotocol.Roles{Mode: systemoneprotocol.RolesNone}, nil
	case systemoneprotocol.KindObject:
	default:
		return systemoneprotocol.Roles{}, fmt.Errorf("%sroles must be an object of state keys to roles", at)
	}
	for _, member := range value.Members() {
		role, _ := member.Value.Text()
		switch systemoneprotocol.Role(role) {
		case systemoneprotocol.RoleUser, systemoneprotocol.RoleContext, systemoneprotocol.RoleAnswer:
		default:
			return systemoneprotocol.Roles{}, fmt.Errorf("%sroles.%s must be user, context or answer", at, member.Key)
		}
	}
	return systemoneprotocol.Roles{Mode: systemoneprotocol.RolesExplicit, Map: append([]systemoneprotocol.Member{}, value.Members()...)}, nil
}

func checkAssets(value systemoneprotocol.Value, at string) error {
	if value.Kind() != systemoneprotocol.KindObject {
		return fmt.Errorf("%s must be an object of placeholders to assets", at)
	}
	for _, asset := range value.Members() {
		if asset.Value.Kind() != systemoneprotocol.KindObject {
			return fmt.Errorf("%s.%s must be an object", at, asset.Key)
		}
		for _, member := range asset.Value.Members() {
			if !assetMembers[member.Key] {
				return fmt.Errorf("%s.%s: %s is not a member of an asset", at, asset.Key, member.Key)
			}
		}
	}
	return nil
}

// decodeOptions reads options, keeping their order and number literals. A
// lenient reader (the internal form) keeps an option it cannot read in Extra
// for the target to judge; a strict one refuses it, naming schema's
// replacement of a removed one.
func decodeOptions(value systemoneprotocol.Value, allowed map[string]bool, schema string, lenient bool) (systemoneprotocol.Options, error) {
	options := systemoneprotocol.Options{}
	if value.Kind() != systemoneprotocol.KindObject {
		return options, fmt.Errorf("options must be an object")
	}
	options.Order = value.Keys()
	for _, member := range value.Members() {
		if !allowed[member.Key] {
			if lenient {
				options.Extra = append(options.Extra, member)
				continue
			}
			if member.Key == "long_context" && schema != "" {
				return options, fmt.Errorf("options.long_context is not an option in %s; send the request header %s: true", schema, HeaderLongContext)
			}
			return options, fmt.Errorf("options.%s is not an option", member.Key)
		}
		if err := readOption(&options, member); err != nil {
			if lenient {
				options.Extra = append(options.Extra, member)
				continue
			}
			return options, err
		}
	}
	return options, nil
}

func readOption(options *systemoneprotocol.Options, member systemoneprotocol.Member) error {
	switch member.Key {
	case "deadline_ms":
		number, ok := member.Value.Float64()
		if !ok || number <= 0 {
			return fmt.Errorf("options.deadline_ms must be a positive number")
		}
		options.DeadlineMs = member.Value
	case "return_meta":
		enabled, ok := member.Value.Boolean()
		if !ok {
			return fmt.Errorf("options.return_meta must be a boolean")
		}
		options.ReturnMeta = &enabled
	case "max_tokens":
		number, ok := member.Value.Float64()
		if !ok || number < 1 || number != float64(int64(number)) {
			return fmt.Errorf("options.max_tokens must be a positive integer")
		}
		options.MaxTokens = member.Value
	case "long_context":
		enabled, ok := member.Value.Boolean()
		if !ok {
			return fmt.Errorf("options.long_context must be a boolean")
		}
		options.LongContext = enabled
	}
	return nil
}

// EncodeRequest writes a batch in the internal form: a batch of one in
// FormAlone as the request alone, every other batch under states, whose
// shared assets are the items' assets.
func (c InternalCodec) EncodeRequest(batch systemoneprotocol.Batch) ([]byte, error) {
	value, err := c.EncodeRequestValue(batch)
	if err != nil {
		return nil, err
	}
	return json.Marshal(value)
}

// EncodeRequestValue is EncodeRequest as a Value.
func (InternalCodec) EncodeRequestValue(batch systemoneprotocol.Batch) (systemoneprotocol.Value, error) {
	if len(batch.Items) == 0 {
		return systemoneprotocol.Value{}, fmt.Errorf("a request has no state")
	}
	if len(batch.Items) == 1 && batch.Form == systemoneprotocol.FormAlone {
		item := batch.Items[0].Request
		members := internalRequestMembers(item)
		if model := firstNonEmpty(item.Model, batch.Model); model != "" || hasKey(item.Order, "model") {
			members["model"] = systemoneprotocol.String(model)
		}
		if !batch.Options.IsZero() || hasKey(item.Order, "options") {
			members["options"] = encodeOptions(batch.Options, true)
		}
		return orderMembers(item.Order, members, internalAloneRank), nil
	}
	members := map[string]systemoneprotocol.Value{}
	if batch.Model != "" || hasKey(batch.Order, "model") {
		members["model"] = systemoneprotocol.String(batch.Model)
	}
	states := systemoneprotocol.Object()
	shared := systemoneprotocol.Object()
	for _, member := range batch.Shared {
		if member.Key == "assets" {
			shared = member.Value
		} else {
			members[member.Key] = member.Value
		}
	}
	for _, item := range batch.Items {
		request := item.Request
		if len(request.Assets) > 0 {
			merged, err := mergeAssets(shared, request.Assets, item.Name)
			if err != nil {
				return systemoneprotocol.Value{}, err
			}
			shared, request.Assets = merged, nil
		}
		entry := internalRequestMembers(request)
		states = states.With(item.Name, orderMembers(request.Order, entry, internalItemRank))
	}
	members["states"] = states
	if len(shared.Members()) > 0 {
		members["assets"] = shared
	}
	if !batch.Options.IsZero() || hasKey(batch.Order, "options") {
		members["options"] = encodeOptions(batch.Options, true)
	}
	return orderMembers(batch.Order, members, internalBatchRank), nil
}

// mergeAssets adds a batch item's assets to the assets a batch form shares;
// a placeholder may stand for one asset only.
func mergeAssets(shared systemoneprotocol.Value, assets []systemoneprotocol.Member, item string) (systemoneprotocol.Value, error) {
	if shared.Kind() != systemoneprotocol.KindObject {
		shared = systemoneprotocol.Object()
	}
	for _, asset := range assets {
		if earlier, ok := shared.Get(asset.Key); ok && !earlier.Equal(asset.Value) {
			return systemoneprotocol.Value{}, fmt.Errorf("request %q uses placeholder %q for another asset than an earlier request of the batch; give each asset its own placeholder", item, asset.Key)
		}
		shared = shared.With(asset.Key, asset.Value)
	}
	return shared, nil
}

// internalRequestMembers are a request's own members in the internal form:
// the state and questions it has (the target says what a request needs).
func internalRequestMembers(request systemoneprotocol.Request) map[string]systemoneprotocol.Value {
	members := map[string]systemoneprotocol.Value{}
	if !request.State.IsNull() || hasKey(request.Order, "state") {
		members["state"] = request.State
	}
	if len(request.Questions) > 0 || hasKey(request.Order, "questions") {
		questions := systemoneprotocol.Object()
		for _, named := range request.Questions {
			questions = questions.With(named.Name, named.Question.QuestionValue())
		}
		members["questions"] = questions
	}
	if len(request.Assets) > 0 || hasKey(request.Order, "assets") {
		members["assets"] = systemoneprotocol.Object(request.Assets...)
	}
	switch request.Roles.Mode {
	case systemoneprotocol.RolesNone:
		members["roles"] = systemoneprotocol.Null()
	case systemoneprotocol.RolesExplicit:
		members["roles"] = systemoneprotocol.Object(request.Roles.Map...)
	}
	for _, member := range request.Extensions {
		members[member.Key] = member.Value
	}
	return members
}

var (
	internalAloneRank = rankOf("model", "state", "questions", "images", "videos", "assets", "roles", "options")
	internalItemRank  = rankOf("state", "questions", "images", "videos", "roles")
	internalBatchRank = rankOf("model", "states", "assets", "options")
	optionsRank       = rankOf("deadline_ms", "return_meta", "max_tokens", "long_context")
)

func rankOf(keys ...string) func(string) int {
	return func(key string) int {
		for index, name := range keys {
			if name == key {
				return index
			}
		}
		return len(keys)
	}
}

// orderMembers writes members in order, then the rest by rank (and name).
func orderMembers(order []string, members map[string]systemoneprotocol.Value, rank func(string) int) systemoneprotocol.Value {
	object := systemoneprotocol.Object()
	written := make(map[string]bool, len(members))
	for _, key := range order {
		if value, ok := members[key]; ok && !written[key] {
			object, written[key] = object.With(key, value), true
		}
	}
	rest := make([]string, 0, len(members))
	for key := range members {
		if !written[key] {
			rest = append(rest, key)
		}
	}
	sort.SliceStable(rest, func(i, j int) bool {
		if rank(rest[i]) != rank(rest[j]) {
			return rank(rest[i]) < rank(rest[j])
		}
		return rest[i] < rest[j]
	})
	for _, key := range rest {
		object = object.With(key, members[key])
	}
	return object
}

// encodeOptions writes options in their order; internal says whether
// long_context goes as an option (the internal form and vllm-sr/4.0) or not
// at all (it is a header elsewhere).
func encodeOptions(options systemoneprotocol.Options, internal bool) systemoneprotocol.Value {
	if options.Null && options.IsZero() {
		return systemoneprotocol.Null()
	}
	members := map[string]systemoneprotocol.Value{}
	if !options.DeadlineMs.IsNull() {
		members["deadline_ms"] = options.DeadlineMs
	}
	if options.ReturnMeta != nil {
		members["return_meta"] = systemoneprotocol.Bool(*options.ReturnMeta)
	}
	if !options.MaxTokens.IsNull() {
		members["max_tokens"] = options.MaxTokens
	}
	if options.LongContext && internal {
		members["long_context"] = systemoneprotocol.Bool(true)
	} else if internal && hasKey(options.Order, "long_context") {
		members["long_context"] = systemoneprotocol.Bool(false)
	}
	for _, member := range options.Extra {
		members[member.Key] = member.Value
	}
	return orderMembers(options.Order, members, optionsRank)
}

func hasKey(order []string, key string) bool {
	for _, name := range order {
		if name == key {
			return true
		}
	}
	return false
}

// unanswerable is the runtime's refusal of a request none of whose questions
// is valid, every answer of every item failed as invalid_question: 400
// invalid_request naming the first question, and its entry in a batch. It is
// nil when something in the request is answered or failed otherwise.
func unanswerable(results []ItemResult, names []string) *systemoneprotocol.Problem {
	var refusal *systemoneprotocol.Problem
	var reasons []string
	for index, result := range results {
		for _, answer := range result.Response.Answers {
			if textOf(mustGet(answer.Value, "error")) != "invalid_question" {
				return nil
			}
			reason, location := textOf(mustGet(answer.Value, "reason")), answer.Key
			if names != nil {
				location = "states." + names[index] + ".questions." + answer.Key
			}
			reasons = append(reasons, location+": "+firstNonEmpty(reason, "invalid_question"))
			if refusal == nil {
				refusal = &systemoneprotocol.Problem{
					Status: http.StatusBadRequest, Kind: systemoneprotocol.ProblemRejected, Code: "invalid_request",
					Question: answer.Key, QuestionError: "invalid_question", Reason: reason,
				}
				if names != nil {
					refusal.Item = names[index]
				}
			}
		}
	}
	if refusal != nil {
		refusal.Detail = "no question is valid (" + strings.Join(reasons, "; ") + ")"
	}
	return refusal
}

// HasRoles reports whether an internal request carries roles, null among
// them: a vllm-sr/4.0 server reads none, so such a request goes to one
// through its codec, which strips roles its keys imply and refuses others.
func HasRoles(body []byte) bool {
	batch, err := InternalCodec{}.DecodeRequest(body)
	if err != nil {
		return false
	}
	for _, item := range batch.Items {
		if item.Request.Roles.Mode != systemoneprotocol.RolesImplicit {
			return true
		}
	}
	return false
}

// DecodeResponse reads an internal answer to a request of form: a response
// alone, or a batch's entries under states, each under its name.
func (InternalCodec) DecodeResponse(form systemoneprotocol.BatchForm, names []string, body []byte) ([]ItemResult, string, error) {
	value, err := systemoneprotocol.Parse(body)
	if err != nil || value.Kind() != systemoneprotocol.KindObject {
		return nil, "", fmt.Errorf("the response is not a JSON object")
	}
	model := textOf(mustGet(value, "model"))
	if form == systemoneprotocol.FormAlone {
		response, err := responseFromValue(value)
		return []ItemResult{{Response: response, Err: err}}, model, nil
	}
	states, _ := value.Get("states")
	results := make([]ItemResult, len(names))
	for index, name := range names {
		entry, ok := states.Get(name)
		if !ok {
			results[index].Err = &systemoneprotocol.Problem{Status: http.StatusBadGateway, Kind: systemoneprotocol.ProblemFailed, Detail: fmt.Sprintf("the batch response has no answers about %q", name)}
			continue
		}
		response, err := responseFromValue(entry)
		if response.Model == "" {
			response.Model = model
		}
		results[index] = ItemResult{Response: response, Err: err}
	}
	return results, model, nil
}

// EncodeResults writes items' results as the internal form answers a request
// of form: a response alone, else the entries under states with the usage of
// them all. An item's error answers the request as the runtime would: the
// first failed item's, with its name as the error's state.
func (InternalCodec) EncodeResults(results []ItemResult, names []string, form systemoneprotocol.BatchForm, model string) (int, []byte, error) {
	if form == systemoneprotocol.FormAlone {
		if len(results) != 1 {
			return 0, nil, fmt.Errorf("a single request has %d results", len(results))
		}
		if results[0].Err != nil {
			status, body := NativeErrorBody(results[0].Err, "")
			return status, body, nil
		}
		if refusal := unanswerable(results, nil); refusal != nil {
			status, body := NativeErrorBody(refusal, "")
			return status, body, nil
		}
		body, err := json.Marshal(responseValue(results[0].Response, model))
		return http.StatusOK, body, err
	}
	if len(results) != len(names) {
		return 0, nil, fmt.Errorf("a batch of %d requests has %d results", len(names), len(results))
	}
	states := systemoneprotocol.Object()
	usage := systemoneprotocol.Usage{}
	for index, result := range results {
		if result.Err != nil {
			status, body := NativeErrorBody(result.Err, names[index])
			return status, body, nil
		}
		states = states.With(names[index], responseValue(result.Response, model))
		usage.InputTokens += result.Response.Usage.InputTokens
		usage.OutputTokens += result.Response.Usage.OutputTokens
	}
	if refusal := unanswerable(results, names); refusal != nil {
		status, body := NativeErrorBody(refusal, "")
		return status, body, nil
	}
	body, err := json.Marshal(systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String(model)},
		systemoneprotocol.Member{Key: "states", Value: states},
		systemoneprotocol.Member{Key: "usage", Value: usageValue(usage)},
	))
	return http.StatusOK, body, err
}

// responseValue is a response as a JSON object: model, answers, the members
// beside them, then usage.
func responseValue(response systemoneprotocol.Response, model string) systemoneprotocol.Value {
	if response.Model != "" {
		model = response.Model
	}
	object := systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "model", Value: systemoneprotocol.String(model)},
		systemoneprotocol.Member{Key: "answers", Value: systemoneprotocol.Object(response.Answers...)},
	)
	for _, member := range response.Extensions {
		object = object.With(member.Key, member.Value)
	}
	return object.With("usage", usageValue(response.Usage))
}

func usageValue(usage systemoneprotocol.Usage) systemoneprotocol.Value {
	return systemoneprotocol.Object(
		systemoneprotocol.Member{Key: "input_tokens", Value: systemoneprotocol.Int(usage.InputTokens)},
		systemoneprotocol.Member{Key: "output_tokens", Value: systemoneprotocol.Int(usage.OutputTokens)},
	)
}

func firstNonEmpty(values ...string) string {
	for _, value := range values {
		if value != "" {
			return value
		}
	}
	return ""
}
