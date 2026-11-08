package modelservice

import (
	"encoding/json"
	"fmt"
	"maps"
	"net/http"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// questionFailure is a System One API's failure of a whole call on one of its
// questions: the question's ID and why it failed.
type questionFailure struct {
	question string
	code     string
}

// failedQuestion reads a failed call's answer: the question a System One API
// names when it fails a call on it, as
// {"error": {"code": "question_failed", "question": "<id>", "question_error": ...}}.
func failedQuestion(status int, body []byte) (questionFailure, bool) {
	if status < http.StatusBadRequest || status >= http.StatusInternalServerError || status == http.StatusTooManyRequests {
		return questionFailure{}, false
	}
	var answer struct {
		Error struct {
			Code          string          `json:"code"`
			Question      string          `json:"question"`
			QuestionError json.RawMessage `json:"question_error"`
		} `json:"error"`
	}
	if len(body) > SystemOneResponseLimit || json.Unmarshal(body, &answer) != nil {
		return questionFailure{}, false
	}
	if answer.Error.Code != "question_failed" || answer.Error.Question == "" {
		return questionFailure{}, false
	}
	return questionFailure{question: answer.Error.Question, code: questionErrorCode(answer.Error.QuestionError)}, true
}

// questionErrorCode is the item error a failed question carries: the API's
// code, given as a string or as an object's code, else question_failed.
func questionErrorCode(raw json.RawMessage) string {
	var code string
	if json.Unmarshal(raw, &code) == nil && code != "" {
		return code
	}
	var object struct {
		Code string `json:"code"`
	}
	if json.Unmarshal(raw, &object) == nil && object.Code != "" {
		return object.Code
	}
	return "question_failed"
}

// answer hands the failed question its error, about every state that asked
// it.
func (f questionFailure) answer(task *bundleTask, responses []Response) {
	for index, state := range task.decision.states {
		for _, question := range state.request.Questions {
			if question.ID == f.question {
				responses[index].Answers[question.ID] = Answer{Type: question.Type, Error: f.code}
			}
		}
	}
}

// remotePlan is how a fused task went in one System One call: the call's
// body, the task's state it asked as the request's own, and the states it
// left out because it asked nothing else about them.
type remotePlan struct {
	body    api.DecisionRequest
	own     int
	skipped map[int]bool
}

func sentAs(task *bundleTask) remotePlan {
	return remotePlan{body: *task.task.Decisions}
}

// without is the plan that asks every question of p but id. ok is false when
// p does not ask id or asks nothing else. When the request's own state was
// asked only id, the first of its further states takes its place.
func (p remotePlan) without(task *bundleTask, id string) (remotePlan, bool) {
	rest := remotePlan{body: p.body, own: p.own, skipped: maps.Clone(p.skipped)}
	if rest.skipped == nil {
		rest.skipped = make(map[int]bool)
	}
	var asked bool
	rest.body.Questions, asked = dropQuestion(p.body.Questions, id)
	states := make(map[string]api.DecisionState)
	if p.body.States != nil {
		for name, state := range *p.body.States {
			questions, dropped := dropQuestion(state.Questions, id)
			asked = asked || dropped
			if len(questions) == 0 {
				rest.skipped[task.stateIndex(name)] = true
				continue
			}
			state.Questions = questions
			states[name] = state
		}
	}
	if !asked {
		return p, false
	}
	if len(rest.body.Questions) == 0 {
		names := slices.Sorted(maps.Keys(states))
		if len(names) == 0 {
			return p, false
		}
		rest.skipped[p.own] = true
		promoted := states[names[0]]
		rest.body.State, rest.body.Questions = promoted.State, promoted.Questions
		rest.body.Images, rest.body.Videos = promoted.Images, promoted.Videos
		rest.own = task.stateIndex(names[0])
		delete(states, names[0])
	}
	rest.body.States = nil
	if len(states) > 0 {
		rest.body.States = &states
	}
	return rest, true
}

// decode reads the answers of a call made with p about each of the task's
// states; a state p left out has none.
func (p remotePlan) decode(task *bundleTask, status int, data []byte) ([]Response, error) {
	body, err := decodeSystemOneBody(status, data)
	if err != nil {
		return nil, err
	}
	responses := make([]Response, len(task.decision.states))
	for index, state := range task.decision.states {
		switch {
		case p.skipped[index]:
			responses[index] = Response{Model: body.Model, Answers: make(map[string]Answer)}
		case index == p.own:
			responses[index] = decodeResponse(body, state.request.Questions)
		default:
			var entry api.DecisionStateResponse
			found := false
			if body.States != nil {
				entry, found = (*body.States)[state.name]
			}
			if !found {
				return nil, fmt.Errorf("%w: the System One response has no answers about state %q", ErrFailed, state.name)
			}
			responses[index] = decodeResponse(api.DecisionResponse{
				Model: entry.Model, Answers: entry.Answers, Sets: entry.Sets, Spans: entry.Spans,
				Thresholds: entry.Thresholds, SpanHeads: entry.SpanHeads, Usage: entry.Usage,
			}, state.request.Questions)
		}
	}
	return responses, nil
}

// stateIndex is the index of the task's state a request names name; the
// request's own state is "".
func (t *bundleTask) stateIndex(name string) int {
	for index, state := range t.decision.states {
		if state.name == name {
			return index
		}
	}
	return -1
}

func dropQuestion(questions map[string]api.Question, id string) (map[string]api.Question, bool) {
	if _, ok := questions[id]; !ok {
		return questions, false
	}
	rest := maps.Clone(questions)
	delete(rest, id)
	return rest, true
}
