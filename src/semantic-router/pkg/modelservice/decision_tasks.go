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
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
	Stage       string `json:"stage"`
	Input       string `json:"input"`
	Output      string `json:"output"`
	FullInput   bool   `json:"full_input"`
	// Preset names the preset the task asks (presets), whose trained
	// bindings choose the question's wire form per model family.
	Preset    string                  `json:"preset,omitempty"`
	Question  Question                `json:"-"`
	Consumers []TaskConsumerReference `json:"consumers,omitempty"`
}

// TaskConsumerReference describes a registry-owned consumer requirement.
type TaskConsumerReference struct {
	Kind    string `json:"kind"`
	Type    string `json:"type"`
	Binding string `json:"binding,omitempty"`
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
// and the exact native questions used to implement it, which may go out
// under other IDs (a family's trained IDs) and with other options (a
// family's trained options, each answering the canonical option Canonical
// names); answers come back under the semantic question's ID and options.
type TaskPlan struct {
	Definition     TaskDefinition
	Question       Question
	Implementation string
	Questions      []Question
	Canonical      map[string]string
	// base marks a preset's question the card's family answers with its
	// base model, every adapter off (presets.BaseOnly).
	base bool
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
// Family and training ancestry intentionally do not participate in admission;
// they only choose the wire form of a preset the family was trained on
// (presetPlan), and mark the questions no adapter was trained on for a family
// with an opt-in decision adapter (withDecisionAdapter).
func CompileTask(definition TaskDefinition, question Question, card ModelCard) (TaskPlan, error) {
	plan, err := compileTask(definition, question, card)
	if err != nil {
		return plan, err
	}
	return withDecisionAdapter(plan, card), nil
}

func compileTask(definition TaskDefinition, question Question, card ModelCard) (TaskPlan, error) {
	question, err := PrepareTaskQuestion(definition, question)
	plan := TaskPlan{Definition: definition, Question: question, Implementation: implementationNative}
	if err != nil {
		return plan, err
	}
	if question.RequireFullInput && !card.ReportsInputCoverage() {
		return plan, fmt.Errorf("%w: model %s answers over %s, which has no input_coverage to prove that %s read its whole input; read it truncated (coverage: truncate), or bind it to a deployment that reports coverage", ErrRejected, card.ID, card.Schema, definition.ID)
	}
	if question.Preset != "" {
		if !card.HasPreset(question.Preset) {
			return plan, fmt.Errorf("%w: model %s does not define preset %s", ErrRejected, card.ID, question.Preset)
		}
		plan.Questions = []Question{question}
		return plan, nil
	}
	if form, ok := presetPlan(definition, question, card); ok {
		plan.Questions, plan.Implementation, plan.Canonical, plan.base = form.questions, form.implementation, form.canonical, form.base
		if form.semantic.ID != "" {
			plan.Question = form.semantic
		}
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
	plan.Implementation = implementationComposedNoul
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
// Plans that compile to the same question under one ID (two consumers of a
// family's trained question) share its one answer.
func ExecuteTaskPlans(ctx context.Context, decider Decider, deployment string, input Request, plans []TaskPlan) (Response, map[string]TaskResult, error) {
	started := time.Now()
	input.Questions = nil
	planned := make(map[string]bool, len(plans))
	asked := make(map[string]Question)
	for _, plan := range plans {
		if planned[plan.Question.ID] {
			return Response{}, nil, fmt.Errorf("%w: duplicate compiled question %q", ErrRejected, plan.Question.ID)
		}
		planned[plan.Question.ID] = true
		for _, question := range plan.Questions {
			if previous, exists := asked[question.ID]; exists {
				if !sameWireQuestion(previous, question) {
					return Response{}, nil, fmt.Errorf("%w: duplicate compiled question %q", ErrRejected, question.ID)
				}
				continue
			}
			asked[question.ID] = question
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
	if plan.Implementation == implementationNative || plan.Implementation == implementationTrained {
		answer, ok := response.Answers[plan.Questions[0].ID]
		if !ok {
			return Answer{Type: plan.Question.Type, Error: "missing_answer"}
		}
		answer = canonicalAnswer(answer, plan.Canonical)
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

// canonicalAnswer reads an answer to a family's trained options as the
// canonical options they answer.
func canonicalAnswer(answer Answer, canonical map[string]string) Answer {
	if len(canonical) == 0 || answer.Error != "" {
		return answer
	}
	if key, ok := canonical[answer.Choice]; ok {
		answer.Choice = key
	}
	if answer.Probabilities != nil {
		probabilities := make(map[string]float64, len(answer.Probabilities))
		for key, probability := range answer.Probabilities {
			if mapped, ok := canonical[key]; ok {
				key = mapped
			}
			probabilities[key] = probability
		}
		answer.Probabilities = probabilities
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
			taskID = "systemone"
		}
		if stage == "" {
			stage = "request"
		}
		definition := TaskDefinition{ID: taskID, Stage: stage, Output: question.Type, FullInput: !question.Truncate, Preset: builtinTaskPresets()[taskID]}
		plan, err := CompileTask(definition, question, card)
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
