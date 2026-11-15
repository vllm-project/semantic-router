package modelservice

import (
	"encoding/json"
	"maps"
	"net/http"
	"slices"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice/api"
)

// questionFailure is a System One API's failure of a whole call on one of its
// questions: the question's ID, the batch entry it failed about ("" when the
// API names none) and why it failed.
type questionFailure struct {
	question string
	state    string
	code     string
}

// failedQuestion reads a failed call's answer: the question a System One API
// names when it fails a call on it and, in a batch, its entry, as
// {"error": {"code": "question_failed", "question": "<id>", "state": "<entry>", "question_error": ...}}.
func failedQuestion(status int, body []byte) (questionFailure, bool) {
	if status < http.StatusBadRequest || status >= http.StatusInternalServerError || status == http.StatusTooManyRequests {
		return questionFailure{}, false
	}
	var answer struct {
		Error struct {
			Code          string          `json:"code"`
			Question      string          `json:"question"`
			State         string          `json:"state"`
			QuestionError json.RawMessage `json:"question_error"`
		} `json:"error"`
	}
	if len(body) > SystemOneResponseLimit || json.Unmarshal(body, &answer) != nil {
		return questionFailure{}, false
	}
	if answer.Error.Code != "question_failed" || answer.Error.Question == "" {
		return questionFailure{}, false
	}
	return questionFailure{question: answer.Error.Question, state: answer.Error.State, code: questionErrorCode(answer.Error.QuestionError)}, true
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

// about is whether f is about state: the entry the API named, or every state
// when it names none.
func (f questionFailure) about(state *fusedState) bool {
	return f.state == "" || f.state == state.name
}

// fails is whether f fails question id about state: its question or, when
// the API refused the input of the entry it names (invalid_input, as the
// model runtime refuses an entry's image), every question of that entry.
func (f questionFailure) fails(state *fusedState, id string) bool {
	return f.about(state) && (id == f.question || f.state != "" && f.code == "invalid_input")
}

// answer hands f's error to every question f fails, about every state that
// asked it.
func (f questionFailure) answer(task *bundleTask, responses []Response) {
	for index, state := range task.decision.states {
		for _, question := range state.request.Questions {
			if f.fails(state, question.ID) {
				responses[index].Answers[question.ID] = Answer{Type: question.Type, Error: f.code}
			}
		}
	}
}

// remotePlan is how a fused task went in one System One call: the call's
// body and the task's states it asked about.
type remotePlan struct {
	body  api.DecisionRequest
	asked []*fusedState
}

func sentAs(task *bundleTask) remotePlan {
	return remotePlan{body: *task.task.Decisions, asked: task.decision.states}
}

// questions is what p asks about one of its states.
func (p remotePlan) questions(state *fusedState) map[string]api.Question {
	if p.body.States == nil {
		return p.body.Questions
	}
	return (*p.body.States)[state.name].Questions
}

// without is the plan that asks every question of p that failed does not
// fail, about the states it asks anything else about: a standard request
// about one of them, a batch about several. ok is false when failed fails
// none of p's questions, or all of them.
func (p remotePlan) without(failed questionFailure) (remotePlan, bool) {
	rest := make(map[*fusedState]map[string]api.Question, len(p.asked))
	var asked []*fusedState
	dropped := false
	for _, state := range p.asked {
		questions := maps.Clone(p.questions(state))
		maps.DeleteFunc(questions, func(id string, _ api.Question) bool { return failed.fails(state, id) })
		dropped = dropped || len(questions) < len(p.questions(state))
		if len(questions) > 0 {
			rest[state] = questions
			asked = append(asked, state)
		}
	}
	if !dropped || len(asked) == 0 {
		return p, false
	}
	body := decisionsAbout(p.body, asked, func(state *fusedState) map[string]api.Question { return rest[state] })
	return remotePlan{body: body, asked: asked}, true
}

// decode reads the answers of a call made with p about each of the task's
// states; a state p left out has none.
func (p remotePlan) decode(task *bundleTask, status int, data []byte) ([]Response, error) {
	body, err := decodeSystemOneBody(status, data, p.body.States != nil)
	if err != nil {
		return nil, err
	}
	answered, err := decodeStates(body, p.asked)
	if err != nil {
		return nil, err
	}
	responses := make([]Response, len(task.decision.states))
	for index, state := range task.decision.states {
		responses[index] = Response{Model: body.Model, Answers: make(map[string]Answer)}
		if at := slices.Index(p.asked, state); at >= 0 {
			responses[index] = answered[at]
		}
	}
	return responses, nil
}
