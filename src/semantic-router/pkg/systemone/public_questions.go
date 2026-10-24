package systemone

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// questionStatus is the status of a System One request answered as a whole
// for one failed question, by that question's error code.
var questionStatus = map[string]int{
	"invalid_question":     http.StatusUnprocessableEntity,
	"invalid_input":        http.StatusBadRequest,
	"max_length_exceeded":  http.StatusBadRequest,
	"scan_budget_exceeded": http.StatusBadRequest,
	"invalid_model_output": http.StatusInternalServerError,
	"unavailable":          http.StatusServiceUnavailable,
	"deadline_exceeded":    http.StatusGatewayTimeout,
}

// questionReasons word the runtime's fixed per-question reasons; the runtime's
// own messages never reach a client.
var questionReasons = map[string]string{
	"question_malformed":          "the question is malformed",
	"type_unsupported":            "the question type is not one System One defines",
	"box_unsupported":             "this model does not answer box questions",
	"track_unsupported":           "this model does not answer track questions",
	"question_images_unsupported": "this model reads images only in the state, not in a question's instructions or options",
	"box_needs_image":             "a box question needs an image",
	"box_target_ambiguous":        "the box question's instructions must reference the one image to search",
	"video_needed":                "this question needs one video in the state",
	"video_target_ambiguous":      "this question needs exactly one video in the state",
	"box_video_too_long":          "the video is longer than this model's Box-track limit (limits.box_video_max_seconds)",
	"at_out_of_range":             "at is past the end of the video",
	"preset_unknown":              "the model defines no such preset",
	"input_malformed":             "the input is malformed",
	"question_tokens":             "the question is longer than the model reads; long context (options.long_context) reads more",
	"require_full_input":          "the question cannot read its whole input",
	"scan_budget":                 "the input is longer than its scan budget",
	"model_output":                "the model produced an invalid answer",
	"deadline":                    "the question was not answered before the deadline",
	"model_unavailable":           "the model is unavailable",
}

// requestReasons word the runtime's reasons for refusing a body or its assets.
var requestReasons = map[string]string{
	"encoding_unsupported": "send the body uncompressed, or with one Content-Encoding: gzip or zstd",
	"encoding_malformed":   "the request body does not decompress",
	"multipart_malformed":  "the multipart body is malformed",
	"asset_malformed":      "an asset is malformed; assets take type, format and data_base64 or a multipart part",
	"asset_unreferenced":   "every asset must be referenced in the state, instructions or criteria",
	"asset_reference":      "asset references are 1 to 1,024 bytes, at most 65,536 bytes together",
	"asset_uses":           "the request's image uses exceed the request size; put a shared image in the state",
	"request_tokens":       "the request has more tokens than the model reads",
}

// questionFailure is a failed question that answers a whole System One request.
type questionFailure struct {
	id, code, reason string
	failed           int
}

type answerOutcome struct {
	Error  string `json:"error"`
	Reason string `json:"reason"`
}

// failedQuestion is the first failed question of a runtime's 200 answer: of
// the request's own state, then of each states entry.
func failedQuestion(result map[string]json.RawMessage) (questionFailure, bool) {
	var found questionFailure
	scan := func(raw json.RawMessage) {
		var answers map[string]answerOutcome
		if json.Unmarshal(raw, &answers) != nil {
			return
		}
		for _, id := range orderedKeys(raw) {
			outcome := answers[id]
			if outcome.Error == "" {
				continue
			}
			found.failed++
			if found.failed == 1 {
				found.id, found.code, found.reason = id, outcome.Error, outcome.Reason
			}
		}
	}
	scan(result["answers"])
	var states map[string]struct {
		Answers json.RawMessage `json:"answers"`
	}
	if raw, present := result["states"]; present && json.Unmarshal(raw, &states) == nil {
		for _, name := range orderedKeys(raw) {
			scan(states[name].Answers)
		}
	}
	return found, found.failed > 0
}

// refusedQuestion is the question a runtime names when none of a request's
// questions is valid; on /v1/systemone it answers the request.
func refusedQuestion(status int, body []byte, systemOne bool) (questionFailure, bool) {
	var answer struct {
		Error struct {
			Code     string `json:"code"`
			Reason   string `json:"reason"`
			Question string `json:"question"`
		} `json:"error"`
	}
	if !systemOne || status != http.StatusBadRequest || json.Unmarshal(body, &answer) != nil || answer.Error.Question == "" {
		return questionFailure{}, false
	}
	return questionFailure{id: answer.Error.Question, code: "invalid_question", reason: answer.Error.Reason, failed: 1}, true
}

func writeQuestionFailure(w http.ResponseWriter, failure questionFailure) {
	status, known := questionStatus[failure.code]
	if !known {
		status = http.StatusBadGateway
	}
	words, worded := questionReasons[failure.reason]
	if !worded {
		words, failure.reason = strings.ReplaceAll(failure.code, "_", " "), ""
	}
	id := failure.id
	if len(id) > 128 {
		id = id[:128]
	}
	message := fmt.Sprintf("question %q cannot be answered: %s", id, words)
	if failure.failed > 1 {
		message += fmt.Sprintf(" (%d questions failed; POST /v1/decisions answers each one)", failure.failed)
	}
	writeErrorBody(w, status, map[string]any{
		"code": "question_failed", "type": "systemone_error", "question_error": failure.code,
		"reason": failure.reason, "question": failure.id, "message": message,
	})
}

// orderedKeys are the keys of a JSON object in document order.
func orderedKeys(raw json.RawMessage) []string {
	decoder := json.NewDecoder(strings.NewReader(string(raw)))
	if token, err := decoder.Token(); err != nil || token != json.Delim('{') {
		return nil
	}
	var keys []string
	for decoder.More() {
		token, err := decoder.Token()
		if err != nil {
			return keys
		}
		key, _ := token.(string)
		keys = append(keys, key)
		var skip json.RawMessage
		if decoder.Decode(&skip) != nil {
			return keys
		}
	}
	return keys
}
