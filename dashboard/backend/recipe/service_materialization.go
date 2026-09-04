package recipe

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sort"
	"strings"

	"gopkg.in/yaml.v3"
)

func materializeChatRequest(probe ProbeDetail) (ChatRequest, error) {
	request := ChatRequest{
		Model:               probe.Model,
		Messages:            []map[string]any{},
		Tools:               cloneObjects(probe.Tools),
		ToolChoice:          cloneJSONValue(probe.ToolChoice),
		Stream:              true,
		MaxCompletionTokens: defaultCompletionTokens,
	}
	messageLimit, err := requestEnvelopeValueByteLimit(
		request, request.Messages, maxRequestBytes,
	)
	if err != nil {
		return ChatRequest{}, err
	}
	messages, err := materializeMessagesWithLimit(probe, messageLimit)
	if err != nil {
		return ChatRequest{}, err
	}
	if len(messages) == 0 {
		text, textErr := materializeText(probe)
		if textErr != nil {
			return ChatRequest{}, textErr
		}
		messages = []map[string]any{{"role": "user", "content": text}}
	}
	request.Messages = messages
	data, err := json.Marshal(request)
	if err != nil {
		return ChatRequest{}, fmt.Errorf("%w: encode run plan: %w", ErrInvalid, err)
	}
	if len(data) > maxRequestBytes {
		return ChatRequest{}, fmt.Errorf("%w: materialized request exceeds %d byte limit", ErrBadRequest, maxRequestBytes)
	}
	return request, nil
}

func materializeEvalRequest(probe ProbeDetail) (EvalRequest, error) {
	return materializeEvalRequestWithLimit(probe, maxRequestBytes)
}

func materializeEvalRequestWithLimit(probe ProbeDetail, limit int) (EvalRequest, error) {
	request := EvalRequest{
		Model:      probe.Model,
		Tools:      cloneObjects(probe.Tools),
		ToolChoice: cloneJSONValue(probe.ToolChoice),
	}
	if len(probe.Messages) > 0 {
		messageLimit, err := requestValueByteLimit(
			"messages", probe.Model, request.Tools, request.ToolChoice, []any{}, limit,
		)
		if err != nil {
			return EvalRequest{}, err
		}
		messages, err := materializeMessagesWithLimit(probe, messageLimit)
		if err != nil {
			return EvalRequest{}, err
		}
		request.Messages = messages
	} else {
		text, err := materializeText(probe)
		if err != nil {
			return EvalRequest{}, err
		}
		request.Text = text
	}
	data, err := json.Marshal(request)
	if err != nil {
		return EvalRequest{}, fmt.Errorf("%w: encode eval request: %w", ErrInvalid, err)
	}
	if len(data) > limit {
		return EvalRequest{}, fmt.Errorf("%w: materialized request exceeds %d byte limit", ErrBadRequest, limit)
	}
	return request, nil
}

func requestValueByteLimit(
	field string,
	model string,
	tools []map[string]any,
	toolChoice any,
	emptyValue any,
	limit int,
) (int, error) {
	envelope := map[string]any{field: emptyValue}
	if model != "" {
		envelope["model"] = model
	}
	if len(tools) > 0 {
		envelope["tools"] = tools
	}
	if toolChoice != nil {
		envelope["tool_choice"] = toolChoice
	}
	return requestEnvelopeValueByteLimit(envelope, emptyValue, limit)
}

func requestEnvelopeValueByteLimit(
	envelope any,
	emptyValue any,
	limit int,
) (int, error) {
	if limit < 1 {
		return 0, fmt.Errorf("%w: materialized request byte limit must be positive", ErrInvalid)
	}
	encodedEnvelope, err := json.Marshal(envelope)
	if err != nil {
		return 0, fmt.Errorf("%w: encode request envelope: %w", ErrInvalid, err)
	}
	encodedValue, err := json.Marshal(emptyValue)
	if err != nil {
		return 0, fmt.Errorf("%w: encode empty request value: %w", ErrInvalid, err)
	}
	valueLimit := limit - (len(encodedEnvelope) - len(encodedValue))
	if valueLimit < 1 {
		return 0, fmt.Errorf("%w: materialized request exceeds %d byte limit", ErrBadRequest, limit)
	}
	return valueLimit, nil
}

func materializeMessages(probe ProbeDetail) ([]map[string]any, error) {
	return materializeMessagesWithLimit(probe, maxRequestBytes)
}

func materializeMessagesWithLimit(probe ProbeDetail, limit int) ([]map[string]any, error) {
	if err := rejectUntrustedProbeImageParts(probe.Messages); err != nil {
		return nil, err
	}
	if err := validateMaterializedMessageSize(probe, limit); err != nil {
		return nil, err
	}
	messages := cloneObjects(probe.Messages)
	plan, err := planGeneratedMessage(probe, messages)
	if err != nil {
		return nil, err
	}
	if plan != nil {
		generatedItem := map[string]any{
			"type": "text",
			"text": strings.Repeat(plan.character, plan.bytes),
		}
		materialized := make([]any, 0, len(plan.content)+1)
		materialized = append(materialized, plan.content[:plan.contentIndex]...)
		materialized = append(materialized, generatedItem)
		materialized = append(materialized, plan.content[plan.contentIndex:]...)
		messages[probe.GeneratedText.MessageIndex]["content"] = materialized
	}
	if err := materializeImageFixtures(messages, probe.imageFixtures); err != nil {
		return nil, err
	}
	if err := validateMaterializedProbeImages(messages, probe.imageFixtures); err != nil {
		return nil, err
	}
	return messages, nil
}

func rejectUntrustedProbeImageParts(messages []map[string]any) error {
	for messageIndex, message := range messages {
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for contentIndex, raw := range content {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			itemType, _ := item["type"].(string)
			switch strings.ToLower(strings.TrimSpace(itemType)) {
			case "image_url", "input_image":
				return fmt.Errorf(
					"%w: messages[%d].content[%d] must use a declared image_fixture",
					ErrInvalid,
					messageIndex,
					contentIndex,
				)
			case "image_fixture":
				if itemType != "image_fixture" {
					return fmt.Errorf(
						`%w: messages[%d].content[%d].type must be exactly "image_fixture"`,
						ErrInvalid,
						messageIndex,
						contentIndex,
					)
				}
			}
		}
	}
	return nil
}

func validateMaterializedProbeImages(
	messages []map[string]any,
	fixtures map[string]probeImageFixture,
) error {
	allowed := make(map[string]struct{}, len(fixtures))
	for _, fixture := range fixtures {
		allowed["data:"+fixture.MediaType+";base64,"+fixture.DataBase64] = struct{}{}
	}
	for messageIndex, message := range messages {
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for contentIndex, raw := range content {
			item, ok := raw.(map[string]any)
			if !ok {
				continue
			}
			itemType, _ := item["type"].(string)
			if strings.ToLower(strings.TrimSpace(itemType)) != "image_url" {
				continue
			}
			imageURL, ok := item["image_url"].(map[string]any)
			urlValue, urlOK := imageURL["url"].(string)
			if !ok || !urlOK {
				return fmt.Errorf(
					"%w: messages[%d].content[%d] has an invalid materialized image",
					ErrInvalid,
					messageIndex,
					contentIndex,
				)
			}
			if _, trusted := allowed[urlValue]; !trusted {
				return fmt.Errorf(
					"%w: messages[%d].content[%d] contains an unverified materialized image",
					ErrInvalid,
					messageIndex,
					contentIndex,
				)
			}
		}
	}
	return nil
}

type generatedMessagePlan struct {
	content      []any
	contentIndex int
	character    string
	bytes        int
}

func planGeneratedMessage(probe ProbeDetail, messages []map[string]any) (*generatedMessagePlan, error) {
	generated := probe.GeneratedText
	if generated == nil {
		return nil, nil
	}
	label := "generated_text"
	if generated.MessageIndex < 0 || generated.MessageIndex >= len(messages) {
		return nil, fmt.Errorf("%w: %s.message_index is out of range", ErrInvalid, label)
	}
	content, ok := messages[generated.MessageIndex]["content"].([]any)
	if !ok {
		return nil, fmt.Errorf("%w: %s requires the selected message content to be a list", ErrInvalid, label)
	}
	if generated.ContentIndex < 0 || generated.ContentIndex > len(content) {
		return nil, fmt.Errorf("%w: %s.content_index is out of range", ErrInvalid, label)
	}
	character := generated.Character
	if character == "" {
		character = "x"
	}
	if len(character) != 1 || character[0] < 0x20 || character[0] > 0x7e {
		return nil, fmt.Errorf("%w: %s.character must be one printable ASCII character", ErrInvalid, label)
	}
	if generated.TargetTextBytes < 1 || generated.TargetTextBytes > maxGeneratedTextBytes {
		return nil, fmt.Errorf(
			"%w: %s.target_text_bytes must be between 1 and %d",
			ErrInvalid,
			label,
			maxGeneratedTextBytes,
		)
	}
	existingTextBytes := messageContentTextBytes(content)
	generatedBytes := generated.TargetTextBytes - existingTextBytes
	if generatedBytes < 1 {
		return nil, fmt.Errorf(
			"%w: %s.target_text_bytes must exceed the selected message's %d explicit text bytes",
			ErrInvalid,
			label,
			existingTextBytes,
		)
	}
	return &generatedMessagePlan{
		content:      content,
		contentIndex: generated.ContentIndex,
		character:    character,
		bytes:        generatedBytes,
	}, nil
}

func validateMaterializedMessageSize(probe ProbeDetail, limit int) error {
	if limit < 1 {
		return fmt.Errorf("%w: materialized message byte limit must be positive", ErrInvalid)
	}
	messages := cloneObjects(probe.Messages)
	plan, err := planGeneratedMessage(probe, messages)
	if err != nil {
		return err
	}
	generatedExtra := 0
	if plan != nil {
		materialized := make([]any, 0, len(plan.content)+1)
		materialized = append(materialized, plan.content[:plan.contentIndex]...)
		materialized = append(materialized, map[string]any{"type": "text", "text": ""})
		materialized = append(materialized, plan.content[plan.contentIndex:]...)
		messages[probe.GeneratedText.MessageIndex]["content"] = materialized
		encodedCharacter, _ := json.Marshal(plan.character)
		generatedExtra = plan.bytes * (len(encodedCharacter) - 2)
	}
	encoded, err := json.Marshal(messages)
	if err != nil {
		return fmt.Errorf("%w: encode compact messages: %w", ErrInvalid, err)
	}
	total, err := addMaterializedMessageBytes(len(encoded), generatedExtra, limit)
	if err != nil {
		return err
	}
	return addImageFixtureMessageBytes(probe, messages, total, limit)
}

func addImageFixtureMessageBytes(
	probe ProbeDetail,
	messages []map[string]any,
	total int,
	limit int,
) error {
	for messageIndex, message := range messages {
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for contentIndex, raw := range content {
			item, ok := raw.(map[string]any)
			if !ok || item["type"] != "image_fixture" {
				continue
			}
			fixture, err := resolveImageFixture(probe.imageFixtures, item, messageIndex, contentIndex)
			if err != nil {
				return err
			}
			compact, _ := json.Marshal(item)
			empty, _ := json.Marshal(map[string]any{"type": "image_url", "image_url": map[string]any{"url": ""}})
			extra := len(empty) - len(compact) + len("data:"+fixture.MediaType+";base64,") + len(fixture.DataBase64)
			if extra < 0 {
				extra = 0
			}
			total, err = addMaterializedMessageBytes(total, extra, limit)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func addMaterializedMessageBytes(total, extra, limit int) (int, error) {
	if total > limit || extra > limit-total {
		return 0, fmt.Errorf("%w: materialized messages exceed %d byte limit", ErrBadRequest, limit)
	}
	return total + extra, nil
}

func materializeImageFixtures(
	messages []map[string]any,
	fixtures map[string]probeImageFixture,
) error {
	for messageIndex, message := range messages {
		content, ok := message["content"].([]any)
		if !ok {
			continue
		}
		for contentIndex, raw := range content {
			item, ok := raw.(map[string]any)
			if !ok || item["type"] != "image_fixture" {
				continue
			}
			fixture, err := resolveImageFixture(fixtures, item, messageIndex, contentIndex)
			if err != nil {
				return err
			}
			content[contentIndex] = map[string]any{
				"type": "image_url",
				"image_url": map[string]any{
					"url": "data:" + fixture.MediaType + ";base64," + fixture.DataBase64,
				},
			}
		}
	}
	return nil
}

func resolveImageFixture(
	fixtures map[string]probeImageFixture,
	item map[string]any,
	messageIndex int,
	contentIndex int,
) (probeImageFixture, error) {
	fixtureName, ok := item["fixture"].(string)
	if !ok || strings.TrimSpace(fixtureName) == "" {
		return probeImageFixture{}, fmt.Errorf(
			"%w: messages[%d].content[%d].fixture must be a non-empty string",
			ErrInvalid,
			messageIndex,
			contentIndex,
		)
	}
	fixtureName = strings.TrimSpace(fixtureName)
	fixture, exists := fixtures[fixtureName]
	if !exists {
		return probeImageFixture{}, fmt.Errorf(
			"%w: messages[%d].content[%d] references unknown image fixture %q",
			ErrInvalid,
			messageIndex,
			contentIndex,
			fixtureName,
		)
	}
	return fixture, nil
}

func messageContentTextBytes(content []any) int {
	total := 0
	for _, raw := range content {
		item, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		rawKind, hasKind := item["type"]
		kind, kindIsString := rawKind.(string)
		if hasKind && !kindIsString {
			continue
		}
		kind = strings.ToLower(strings.TrimSpace(kind))
		if kind != "" && kind != "text" && kind != "input_text" && kind != "output_text" {
			continue
		}
		text, ok := item["text"].(string)
		if ok {
			total += len(text)
		}
	}
	return total
}

func materializeText(probe ProbeDetail) (string, error) {
	if probe.Query == "" {
		return "", fmt.Errorf("%w: text probe has no query", ErrInvalid)
	}
	repeated, err := repeatLinesBounded(probe.Query, probe.Repeat, maxRequestBytes)
	if err != nil {
		return "", err
	}
	if probe.Padding == nil {
		return repeated, nil
	}
	extraSeparators := 1
	if probe.Padding.Placement == "around" && probe.Padding.Repeat > 1 {
		extraSeparators = 2
	}
	padding, err := repeatLinesBounded(probe.Padding.Text, probe.Padding.Repeat, maxRequestBytes-len(repeated)-extraSeparators)
	if err != nil {
		return "", err
	}
	text, err := placeProbePadding(probe, repeated, padding)
	if err != nil {
		return "", err
	}
	if len(text) > maxRequestBytes {
		return "", fmt.Errorf("%w: materialized query exceeds %d byte limit", ErrBadRequest, maxRequestBytes)
	}
	return text, nil
}

func placeProbePadding(probe ProbeDetail, repeated, padding string) (string, error) {
	switch probe.Padding.Placement {
	case "before":
		return joinNonEmpty(padding, repeated), nil
	case "after":
		return joinNonEmpty(repeated, padding), nil
	case "around":
		return placeAroundProbePadding(probe, repeated)
	default:
		return "", fmt.Errorf("%w: unsupported padding placement %q", ErrInvalid, probe.Padding.Placement)
	}
}

func placeAroundProbePadding(probe ProbeDetail, repeated string) (string, error) {
	midpoint := probe.Padding.Repeat / 2
	before := ""
	if midpoint > 0 {
		var err error
		before, err = repeatLinesBounded(probe.Padding.Text, midpoint, maxRequestBytes)
		if err != nil {
			return "", err
		}
	}
	after, err := repeatLinesBounded(probe.Padding.Text, probe.Padding.Repeat-midpoint, maxRequestBytes)
	if err != nil {
		return "", err
	}
	return joinNonEmpty(before, repeated, after), nil
}

func repeatLinesBounded(value string, repeat, limit int) (string, error) {
	if repeat < 1 || limit < 0 {
		return "", fmt.Errorf("%w: materialized query exceeds %d byte limit", ErrBadRequest, maxRequestBytes)
	}
	lineBytes := len(value)
	separators := repeat - 1
	if separators > limit || lineBytes > (limit-separators)/repeat {
		return "", fmt.Errorf("%w: materialized query exceeds %d byte limit", ErrBadRequest, maxRequestBytes)
	}
	total := lineBytes*repeat + separators
	if total > limit {
		return "", fmt.Errorf("%w: materialized query exceeds %d byte limit", ErrBadRequest, maxRequestBytes)
	}
	return strings.TrimSuffix(strings.Repeat(value+"\n", repeat), "\n"), nil
}

func joinNonEmpty(parts ...string) string {
	nonEmpty := parts[:0]
	for _, part := range parts {
		if part != "" {
			nonEmpty = append(nonEmpty, part)
		}
	}
	return strings.Join(nonEmpty, "\n")
}

func cloneObjects(items []map[string]any) []map[string]any {
	if len(items) == 0 {
		return nil
	}
	cloned := make([]map[string]any, len(items))
	for i, item := range items {
		cloned[i] = make(map[string]any, len(item))
		for key, value := range item {
			cloned[i][key] = cloneJSONValue(value)
		}
	}
	return cloned
}

func contains(values []string, value string) bool {
	for _, candidate := range values {
		if candidate == value {
			return true
		}
	}
	return false
}

func stableUnique(values []string) []string {
	seen := map[string]struct{}{}
	result := make([]string, 0, len(values))
	for _, value := range values {
		if _, exists := seen[value]; exists {
			continue
		}
		seen[value] = struct{}{}
		result = append(result, value)
	}
	return result
}

func probeKey(decisionID, variantID string) string {
	return decisionID + "\x00" + variantID
}

func decodeStrictYAML(data []byte, target any) error {
	decoder := yaml.NewDecoder(bytes.NewReader(data))
	decoder.KnownFields(true)
	if err := decoder.Decode(target); err != nil {
		return err
	}
	var extra any
	if err := decoder.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			return errors.New("must contain exactly one YAML document")
		}
		return err
	}
	return nil
}

func sortedKeys(values map[string]struct{}) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}
