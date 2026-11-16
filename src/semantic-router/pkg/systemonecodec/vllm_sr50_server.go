package systemonecodec

import (
	"fmt"
	"net/http"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/systemoneprotocol"
)

// ReadRules are the two rules a server reading vllm-sr/5.0 forms applies
// where 4.0 and 5.0 overlap.
type ReadRules struct {
	// Roles is what an object state without roles gets: RolesNone on a
	// vllm-sr/5.0 server, RolesImplicit on a vllm-sr/4.0 one.
	Roles systemoneprotocol.RolesMode
	// FrontMedia reads media placed in front of the state (UnplaceMedia) as
	// 4.0's images and videos, as a vllm-sr/5.0 server does; a vllm-sr/4.0
	// server reads every asset where its placeholder is.
	FrontMedia bool
}

// Rules50 are a vllm-sr/5.0 server's rules.
var Rules50 = ReadRules{Roles: systemoneprotocol.RolesNone, FrontMedia: true}

// DecodeRequest reads a vllm-sr/5.0 request alone, POST /v1/systemone's
// body: the standard's members and the extension fields only, any other
// member, a 4.0 form among them, refused with its name. The standard's
// long-context header sets the long context.
func (c VLLMSR50Codec) DecodeRequest(body []byte, header http.Header, rules ReadRules) (systemoneprotocol.Batch, error) {
	value, err := parseObject(body)
	if err != nil {
		return systemoneprotocol.Batch{}, err
	}
	batch := systemoneprotocol.Batch{Form: systemoneprotocol.FormAlone}
	if err = checkFiveMembers(value, systemoneprotocol.LevelRequest, ""); err != nil {
		return batch, err
	}
	if batch.Model, err = fiveModel(value, ""); err != nil {
		return batch, err
	}
	if batch.Options, err = fiveOptions(value, header); err != nil {
		return batch, err
	}
	request, err := c.decodeFiveRequest(value, rules, "")
	if err != nil {
		return batch, err
	}
	request.Model = batch.Model
	batch.Items = []systemoneprotocol.NamedRequest{{Request: request}}
	return batch, nil
}

// DecodeBatch reads POST /v1/systemone/batch's body: model, the named
// standard requests under requests (each may repeat the batch's model), and
// the options.
func (c VLLMSR50Codec) DecodeBatch(body []byte, header http.Header, rules ReadRules) (systemoneprotocol.Batch, error) {
	value, err := parseObject(body)
	if err != nil {
		return systemoneprotocol.Batch{}, err
	}
	batch := systemoneprotocol.Batch{Form: systemoneprotocol.FormRequests}
	for _, member := range value.Members() {
		switch member.Key {
		case "model", "requests", "options":
		default:
			return batch, fmt.Errorf("%q is not a member of a batch; a batch takes model, requests and options", member.Key)
		}
	}
	if batch.Model, err = fiveModel(value, ""); err != nil {
		return batch, err
	}
	if batch.Model == "" {
		return batch, fmt.Errorf("model is required")
	}
	if batch.Options, err = fiveOptions(value, header); err != nil {
		return batch, err
	}
	requests, _ := value.Get("requests")
	if requests.Kind() != systemoneprotocol.KindObject || len(requests.Members()) == 0 {
		return batch, fmt.Errorf("requests must be a nonempty object of named requests")
	}
	for _, entry := range requests.Members() {
		at := "requests." + entry.Key + "."
		if entry.Value.Kind() != systemoneprotocol.KindObject {
			return batch, fmt.Errorf("requests.%s must be an object", entry.Key)
		}
		for _, member := range entry.Value.Members() {
			if member.Key == "options" {
				return batch, fmt.Errorf("%soptions: a batch's options are the batch's own", at)
			}
		}
		if err := checkFiveMembers(entry.Value, systemoneprotocol.LevelRequest, at); err != nil {
			return batch, err
		}
		model, err := fiveModel(entry.Value, at)
		if err != nil {
			return batch, err
		}
		if model != "" && model != batch.Model {
			return batch, fmt.Errorf("%smodel %q is not the batch's model %q", at, model, batch.Model)
		}
		request, err := c.decodeFiveRequest(entry.Value, rules, at)
		if err != nil {
			return batch, err
		}
		batch.Items = append(batch.Items, systemoneprotocol.NamedRequest{Name: entry.Key, Request: request})
	}
	return batch, nil
}

func parseObject(body []byte) (systemoneprotocol.Value, error) {
	value, err := systemoneprotocol.Parse(body)
	if err != nil {
		return systemoneprotocol.Value{}, fmt.Errorf("the request is not valid JSON")
	}
	if value.Kind() != systemoneprotocol.KindObject {
		return systemoneprotocol.Value{}, fmt.Errorf("the request must be a JSON object")
	}
	return value, nil
}

func fiveModel(value systemoneprotocol.Value, at string) (string, error) {
	model, ok := value.Get("model")
	if !ok {
		return "", nil
	}
	text, isText := model.Text()
	if !isText {
		return "", fmt.Errorf("%smodel must be a string", at)
	}
	return text, nil
}

var fiveOptionMembers = memberSet("deadline_ms", "return_meta", "max_tokens")

func fiveOptions(value systemoneprotocol.Value, header http.Header) (systemoneprotocol.Options, error) {
	options := systemoneprotocol.Options{}
	if raw, ok := value.Get("options"); ok {
		decoded, err := decodeOptions(raw, fiveOptionMembers, string(SchemaVLLMSR50), false)
		if err != nil {
			return options, err
		}
		options = decoded
	}
	if longContext := strings.TrimSpace(header.Get(HeaderLongContext)); longContext != "" {
		if !strings.EqualFold(longContext, "true") {
			return options, fmt.Errorf("%s takes true", HeaderLongContext)
		}
		options.LongContext = true
	}
	return options, nil
}

// checkFiveMembers refuses a member vllm-sr/5.0 does not define at a level,
// naming a removed 4.0 form's replacement.
func checkFiveMembers(value systemoneprotocol.Value, level systemoneprotocol.Level, at string) error {
	for _, member := range value.Members() {
		if systemoneprotocol.IsField(level, member.Key) {
			continue
		}
		if removed, ok := systemoneprotocol.Removed(string(level), member.Key); ok {
			return fmt.Errorf("%s%s is not part of vllm-sr/5.0; use %s (%s)", at, member.Key, removed.ReplacedBy, strings.TrimSuffix(removed.Note, "."))
		}
		return fmt.Errorf("%s%s is not a field of a vllm-sr/5.0 %s", at, member.Key, level)
	}
	return nil
}

func (VLLMSR50Codec) decodeFiveRequest(value systemoneprotocol.Value, rules ReadRules, at string) (systemoneprotocol.Request, error) {
	request := systemoneprotocol.Request{}
	state, ok := value.Get("state")
	if !ok {
		return request, fmt.Errorf("%sstate is required", at)
	}
	request.State = state
	raw, _ := value.Get("questions")
	questions, err := decodeQuestions(raw, nil, at)
	if err != nil {
		return request, err
	}
	for _, named := range questions {
		if err := checkFiveQuestion(named, at); err != nil {
			return request, err
		}
	}
	request.Questions = questions
	if assets, ok := value.Get("assets"); ok {
		if err := checkFiveAssets(assets, at+"assets"); err != nil {
			return request, err
		}
		request.Assets = append(request.Assets, assets.Members()...)
	}
	if rules.FrontMedia {
		unplaced, images, videos, rest, placed, err := UnplaceMedia(request.State, request.Assets)
		if err != nil {
			return request, fmt.Errorf("%s%w", at, err)
		}
		if placed {
			request.State, request.Assets = unplaced, rest
			if len(images) > 0 {
				request.Extensions = append(request.Extensions, systemoneprotocol.Member{Key: "images", Value: stringArray(images)})
			}
			if len(videos) > 0 {
				request.Extensions = append(request.Extensions, systemoneprotocol.Member{Key: "videos", Value: stringArray(videos)})
			}
		}
	}
	for _, asset := range request.Assets {
		if textOf(mustGet(asset.Value, "type")) == "video" {
			return request, fmt.Errorf("%sassets.%s: a video asset goes in front of the state: [image placeholders..., video placeholders..., state]", at, asset.Key)
		}
	}
	roles, given := value.Get("roles")
	switch {
	case given:
		decoded, err := decodeRoles(roles, at)
		if err != nil || roles.Kind() != systemoneprotocol.KindObject {
			if err == nil {
				err = fmt.Errorf("%sroles must be an object of state keys to roles", at)
			}
			return request, err
		}
		if request.State.Kind() != systemoneprotocol.KindObject {
			return request, fmt.Errorf("%sroles name the keys of an object state", at)
		}
		for _, member := range decoded.Map {
			if !request.State.Has(member.Key) {
				return request, fmt.Errorf("%sroles.%s names no key of the state", at, member.Key)
			}
		}
		request.Roles = decoded
	case request.State.Kind() == systemoneprotocol.KindObject:
		request.Roles = systemoneprotocol.Roles{Mode: rules.Roles}
	}
	return request, nil
}

func checkFiveQuestion(named systemoneprotocol.NamedQuestion, at string) error {
	where := at + "questions." + named.Name
	question := named.Question
	if question.Type == "" {
		return fmt.Errorf("%s: type is required", where)
	}
	if !systemoneprotocol.IsField(systemoneprotocol.LevelQuestionType, question.Type) {
		return fmt.Errorf("%s: type %q is not a question type", where, question.Type)
	}
	for _, key := range question.Order {
		if systemoneprotocol.IsField(systemoneprotocol.LevelQuestion, key) {
			continue
		}
		if removed, ok := systemoneprotocol.Removed("question", key); ok {
			return fmt.Errorf("%s: %s is not part of vllm-sr/5.0; use %s (%s)", where, key, removed.ReplacedBy, strings.TrimSuffix(removed.Note, "."))
		}
		return fmt.Errorf("%s: %s is not a field of a vllm-sr/5.0 question", where, key)
	}
	return nil
}

func checkFiveAssets(value systemoneprotocol.Value, at string) error {
	if err := checkAssets(value, at); err != nil {
		return err
	}
	for _, asset := range value.Members() {
		kind := textOf(mustGet(asset.Value, "type"))
		if kind != "" && !systemoneprotocol.IsField(systemoneprotocol.LevelAssetType, kind) {
			return fmt.Errorf("%s.%s: type %q is not an asset type", at, asset.Key, kind)
		}
	}
	return nil
}

func stringArray(texts []string) systemoneprotocol.Value {
	items := make([]systemoneprotocol.Value, len(texts))
	for index, text := range texts {
		items[index] = systemoneprotocol.String(text)
	}
	return systemoneprotocol.Array(items...)
}
