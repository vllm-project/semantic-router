package modelservice

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"
)

// TaskDefinition describes a semantic judgment, independently of its model.
// Definitions are internal templates; custom decision questions compile into
// the same plans without introducing another authored configuration tree.
type TaskDefinition struct {
	ID          string                  `json:"id"`
	Title       string                  `json:"title"`
	Description string                  `json:"description"`
	Stage       string                  `json:"stage"`
	Input       string                  `json:"input"`
	Output      string                  `json:"output"`
	FullInput   bool                    `json:"full_input"`
	Question    Question                `json:"-"`
	Consumers   []TaskConsumerReference `json:"consumers,omitempty"`
}

// TaskConsumerReference describes a registry-owned consumer requirement.
// Optional tasks enrich a consumer (for example exact positions), but their
// absence must not disable its verdict-only implementation.
type TaskConsumerReference struct {
	Kind     string `json:"kind"`
	Type     string `json:"type"`
	Binding  string `json:"binding,omitempty"`
	Optional bool   `json:"optional,omitempty"`
}

// TaskCapability describes structural support, never measured model quality.
type TaskCapability struct {
	TaskID         string `json:"task_id"`
	Supported      bool   `json:"supported"`
	Implementation string `json:"implementation,omitempty"`
	Reason         string `json:"reason,omitempty"`
	Quality        string `json:"quality"`
}

// TaskPlan is immutable preparation input. It retains the semantic question
// and the exact native questions used to implement it.
type TaskPlan struct {
	Definition     TaskDefinition
	Question       Question
	Implementation string
	Questions      []Question
}

// TaskResult keeps unknown separate from a negative answer. Coverage describes
// the input read by this task, not other content in the same user request.
type TaskResult struct {
	TaskID         string
	Answer         Answer
	Status         string
	Coverage       string
	Implementation string
}

// PrepareTaskQuestion applies the task's input contract and checks requirements
// independent of model availability. Compilation still checks model support.
func PrepareTaskQuestion(definition TaskDefinition, question Question) (Question, error) {
	question.RequireFullInput = question.RequireFullInput || definition.FullInput
	if question.RequireFullInput && question.Truncate {
		return question, fmt.Errorf("%w: task %s cannot require complete input and truncate it", ErrRejected, definition.ID)
	}
	if question.ID == "" {
		return question, fmt.Errorf("%w: task %s requires a question ID", ErrRejected, definition.ID)
	}
	return question, nil
}

// CompileTask selects an implemented output adapter from the actual card.
// Family and training ancestry intentionally do not participate in admission.
func CompileTask(definition TaskDefinition, question Question, card ModelCard) (TaskPlan, error) {
	question, err := PrepareTaskQuestion(definition, question)
	plan := TaskPlan{Definition: definition, Question: question, Implementation: "native"}
	if err != nil {
		return plan, err
	}
	if question.Preset != "" {
		if !card.HasPreset(question.Preset) {
			return plan, fmt.Errorf("%w: model %s does not define preset %s", ErrRejected, card.ID, question.Preset)
		}
		plan.Questions = []Question{question}
		return plan, nil
	}
	if card.Answers(question.Type) {
		plan.Questions = []Question{question}
		return plan, nil
	}
	if question.Type != "set" || !card.Answers("noul") {
		return plan, fmt.Errorf("%w: model %s does not support %s required by %s", ErrRejected, card.ID, question.Type, definition.ID)
	}
	if len(question.Labels) == 0 {
		return plan, fmt.Errorf("%w: set task %s requires labels", ErrRejected, definition.ID)
	}
	plan.Implementation = "composed_noul"
	for index, label := range question.Labels {
		plan.Questions = append(plan.Questions, Question{
			ID: fmt.Sprintf("%s:label:%d", question.ID, index), Type: "noul", Truncate: question.Truncate,
			RequireFullInput: question.RequireFullInput,
			Instructions:     question.Instructions + "\nDoes the input satisfy this category: " + label.Key + " — " + label.Description + "?",
		})
	}
	return plan, nil
}

func CapabilityForTask(definition TaskDefinition, card ModelCard) TaskCapability {
	plan, err := CompileTask(definition, definition.Question, card)
	capability := TaskCapability{TaskID: definition.ID, Supported: err == nil, Quality: "unevaluated"}
	if err != nil {
		capability.Reason = err.Error()
	} else {
		capability.Implementation = plan.Implementation
	}
	return capability
}

// ExecuteTaskPlans shares one native request when plans read the same input.
// Separate calls still participate in the existing multi-state Bundle fusion.
func ExecuteTaskPlans(ctx context.Context, decider Decider, deployment string, input Request, plans []TaskPlan) (Response, map[string]TaskResult, error) {
	started := time.Now()
	input.Questions = nil
	seen := make(map[string]bool)
	for _, plan := range plans {
		for _, question := range plan.Questions {
			if seen[question.ID] {
				return Response{}, nil, fmt.Errorf("%w: duplicate compiled question %q", ErrRejected, question.ID)
			}
			seen[question.ID] = true
			input.Questions = append(input.Questions, question)
		}
	}
	response, err := decider.Decide(ctx, deployment, input)
	results := make(map[string]TaskResult, len(plans))
	if err != nil {
		for _, plan := range plans {
			status := "error"
			if errors.Is(err, ErrUnavailable) {
				status = "unknown"
			}
			results[plan.Question.ID] = TaskResult{TaskID: plan.Definition.ID, Status: status, Coverage: "unknown", Implementation: plan.Implementation}
			recordTaskResult(deployment, plan, results[plan.Question.ID], time.Since(started))
		}
		return response, results, err
	}
	answers := make(map[string]Answer, len(plans))
	for _, plan := range plans {
		answer := reduceTaskAnswer(plan, response)
		result := TaskResult{TaskID: plan.Definition.ID, Answer: answer, Status: "ok", Coverage: "unknown", Implementation: plan.Implementation}
		if answer.Error != "" {
			result.Status, result.Coverage = "unknown", "unknown"
			if !taskAnswerUnknown(answer.Error) {
				result.Status = "error"
			}
		} else if plan.Question.Truncate {
			result.Coverage = "partial"
		} else if answer.InputCoverage == "complete" {
			result.Coverage = "complete"
		}
		answers[plan.Question.ID], results[plan.Question.ID] = answer, result
		recordTaskResult(deployment, plan, result, time.Since(started))
	}
	response.Answers = answers
	return response, results, nil
}

func reduceTaskAnswer(plan TaskPlan, response Response) Answer {
	if plan.Implementation == "native" {
		answer, ok := response.Answers[plan.Question.ID]
		if !ok {
			return Answer{Type: plan.Question.Type, Error: "missing_answer"}
		}
		if answer.Error == "" && plan.Question.Type != "" && answer.Type != plan.Question.Type {
			answer.Error = "unexpected_answer_type"
		}
		if answer.Error == "" {
			answer.Error = validateTaskAnswer(plan.Question, answer)
		}
		return requireInputCoverage(plan.Question, answer)
	}
	answer := Answer{Type: "set", Probabilities: make(map[string]float64), Selected: []string{}, Threshold: .5}
	if plan.Question.Threshold != nil {
		answer.Threshold = *plan.Question.Threshold
	}
	for index, question := range plan.Questions {
		item, ok := response.Answers[question.ID]
		item = requireInputCoverage(question, item)
		if ok && item.Error != "" {
			return Answer{Type: "set", Error: item.Error}
		}
		if !ok || item.Type != "noul" || math.IsNaN(item.Noul) || math.IsInf(item.Noul, 0) || item.Noul < 0 || item.Noul > 1 {
			return Answer{Type: "set", Error: "incomplete_label_judgments"}
		}
		label := plan.Question.Labels[index].Key
		answer.Probabilities[label] = item.Noul
		if item.Noul > answer.Threshold {
			answer.Selected = append(answer.Selected, label)
		}
	}
	if plan.Question.RequireFullInput {
		answer.InputCoverage = "complete"
	}
	return answer
}

// A legacy or attached runtime may ignore a newly added request field. Its
// successful scalar is not evidence that the full input was inspected.
func requireInputCoverage(question Question, answer Answer) Answer {
	if question.RequireFullInput && answer.Error == "" && answer.InputCoverage != "complete" {
		return Answer{Type: answer.Type, Error: "input_coverage_unknown"}
	}
	return answer
}

// ExecuteQuestions compiles authored and built-in questions through one path.
// Native API passthrough does not call this function: it must preserve the
// model's advertised native capabilities without implicit composition.
func ExecuteQuestions(ctx context.Context, decider Decider, deployment string, card ModelCard, request Request) (Response, error) {
	plans := make([]TaskPlan, 0, len(request.Questions))
	for _, question := range request.Questions {
		taskID, stage := question.TaskID, question.Stage
		if taskID == "" {
			taskID = "decision"
		}
		if stage == "" {
			stage = "request"
		}
		plan, err := CompileTask(TaskDefinition{ID: taskID, Stage: stage, Output: question.Type, FullInput: !question.Truncate}, question, card)
		if err != nil {
			return Response{}, err
		}
		plans = append(plans, plan)
	}
	response, _, err := ExecuteTaskPlans(ctx, decider, deployment, request, plans)
	return response, err
}

// Validate values before a successful result can become routing or privacy
// evidence. The native transport usually rejects these too; alternate Decider
// implementations must obey the same result contract.
func validateTaskAnswer(question Question, answer Answer) string {
	probability := func(p float64) bool { return !math.IsNaN(p) && !math.IsInf(p, 0) && p >= 0 && p <= 1 }
	switch answer.Type {
	case "noul":
		if !probability(answer.Noul) {
			return "invalid_probability"
		}
	case "score":
		if math.IsNaN(answer.Score) || math.IsInf(answer.Score, 0) || answer.Score < 0 || (len(question.Levels) > 0 && answer.Score > float64(len(question.Levels)-1)) {
			return "invalid_score"
		}
	case "choice":
		// Every native Choice adapter returns the complete candidate
		// distribution. Match Decision 1's numerical normalization tolerance.
		if len(answer.Probabilities) == 0 || (len(question.Choices) > 0 && len(answer.Probabilities) != len(question.Choices)) {
			return "invalid_choice_distribution"
		}
		total := 0.0
		for _, p := range answer.Probabilities {
			if !probability(p) {
				return "invalid_choice_distribution"
			}
			total += p
		}
		if math.Abs(total-1) > 2e-5 {
			return "invalid_choice_distribution"
		}
		if _, exists := answer.Probabilities[answer.Choice]; !exists {
			return "unknown_choice"
		}
		if len(question.Choices) > 0 {
			found := false
			for _, choice := range question.Choices {
				if _, exists := answer.Probabilities[choice.Key]; !exists {
					return "invalid_choice_distribution"
				}
				found = found || answer.Choice == choice.Key
			}
			if !found {
				return "unknown_choice"
			}
		}
	case "set":
		labels := make(map[string]bool, len(question.Labels))
		// Presets own their label vocabulary. In that case the returned
		// distribution supplies the labels, just as it does for preset Choice.
		if question.Preset != "" && len(question.Labels) == 0 {
			for label, p := range answer.Probabilities {
				if !probability(p) {
					return "incomplete_label_judgments"
				}
				labels[label] = true
			}
		}
		for _, label := range question.Labels {
			labels[label.Key] = true
			p, exists := answer.Probabilities[label.Key]
			if !exists || !probability(p) {
				return "incomplete_label_judgments"
			}
		}
		selected := make(map[string]bool, len(answer.Selected))
		for _, label := range answer.Selected {
			if !labels[label] || selected[label] {
				return "invalid_selected_labels"
			}
			selected[label] = true
		}
	}
	return ""
}

func taskAnswerUnknown(code string) bool {
	switch code {
	case "unknown", "missing_answer", "missing_answer_value", "incomplete_label_judgments", "input_coverage_unknown", "unavailable", "not_ready", "max_length_exceeded", "input_too_long", "input_limit", "scan_budget_exceeded", "deadline_exceeded":
		return true
	}
	return false
}
