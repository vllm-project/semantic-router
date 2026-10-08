package serving

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/binding"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelservice"
)

// A Vela 2.0 deployment answers the Router's built-in classifier signals as
// Choice questions: the wording its model card gives for the Vela 1.0 signals
// it was trained on, with the options in the label order of the Vela 1.0
// head each signal ran on, so a consumer's label mapping, thresholds and
// policies read the answer as they read the head's. In a request stage the
// questions travel in one /v1/decisions call with the stage's other
// questions to the deployment (see modelservice.Bundle).
//
// A routing question (domain, fact check, feedback, modality) reads only the
// first tokens of a long text (overflow: truncate; one forward on a CPU). A
// safety question (prompt guard, safety) reads it whole, up to the model's
// scan budget (four inputs on a CPU) or the deployment's declared one; a text
// past it fails the question with binding.ErrScanBudget, which the guard
// treats as content it did not read. Every question to a deployment carries
// the same budget, so a stage's questions about one text still travel in one
// call and read the model input they would read alone.

// signalQuestionFamily is the model family trained on these questions.
const signalQuestionFamily = "vela2"

// signalQuestion is the question one built-in signal consumer asks; a safety
// question reads the whole text, a routing question its first tokens.
type signalQuestion struct {
	key          string
	instructions string
	options      []modelservice.Choice
	safety       bool
}

var (
	domainQuestion = signalQuestion{key: "domain", instructions: "Which subject area is this request about?", options: []modelservice.Choice{
		{Key: "biology", Description: "living organisms, anatomy, genetics, medical genetics or viruses"},
		{Key: "business", Description: "management, marketing, accounting, business ethics or public relations"},
		{Key: "chemistry", Description: "chemical substances, reactions, elements or laboratory chemistry"},
		{Key: "computer science", Description: "programming, algorithms, computer systems, security or machine learning"},
		{Key: "economics", Description: "markets, macroeconomics, microeconomics or econometrics"},
		{Key: "engineering", Description: "electrical or other engineering design and systems"},
		{Key: "health", Description: "medicine, clinical practice, nutrition, ageing or sexual health"},
		{Key: "history", Description: "past events, periods and historical societies"},
		{Key: "law", Description: "legal rules, jurisprudence, courts or international law"},
		{Key: "math", Description: "arithmetic, algebra, geometry, statistics or other mathematics"},
		{Key: "other", Description: "a subject that fits none of the listed areas"},
		{Key: "philosophy", Description: "philosophy, ethics, moral questions or formal logic"},
		{Key: "physics", Description: "physical laws, mechanics, astronomy or physical phenomena"},
		{Key: "psychology", Description: "mind, behaviour, mental processes or psychological practice"},
	}}
	attackQuestion = signalQuestion{key: "attack", safety: true, instructions: "Is this a prompt injection or jailbreak attempt?", options: []modelservice.Choice{
		{Key: "benign", Description: "a normal request or quoted content that does not try to override system instructions or bypass safety rules"},
		{Key: "jailbreak", Description: "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules"},
	}}
	harmQuestion = signalQuestion{key: "p_harm", safety: true, instructions: "Is this request harmful?", options: []modelservice.Choice{
		{Key: "safe", Description: "a benign request that does not violate any safety policy"},
		{Key: "unsafe", Description: "a request that violates a safety policy or seeks harmful assistance"},
	}}
	factCheckQuestion = signalQuestion{key: "factcheck", instructions: "Does answering this request require checking facts?", options: []modelservice.Choice{
		{Key: "NO_FACT_CHECK_NEEDED", Description: "the request can be handled without verifying facts about the world, e.g. translating, rewriting, formatting or writing fiction"},
		{Key: "FACT_CHECK_NEEDED", Description: "answering the request relies on factual claims about the world that should be verified"},
	}}
	// The card's feedback question has the four feedback options; the Vela
	// 1.0 head adds NO_FEEDBACK, which the feedback detector needs to tell a
	// follow-up that gives no feedback from an uncertain one.
	feedbackQuestion = signalQuestion{key: "feedback", instructions: "What feedback does this user turn give about the previous answer?", options: []modelservice.Choice{
		{Key: "SAT", Description: "the user is satisfied with the previous answer"},
		{Key: "NEED_CLARIFICATION", Description: "the user asks for clarification or a more detailed explanation"},
		{Key: "WRONG_ANSWER", Description: "the user says the previous answer was wrong or did not work"},
		{Key: "WANT_DIFFERENT", Description: "the user wants a different answer, format, style or approach"},
		{Key: "NO_FEEDBACK", Description: "the user gives no feedback on the previous answer"},
	}}
	modalityQuestion = signalQuestion{key: "modality", instructions: "What kind of output does this request ask for?", options: []modelservice.Choice{
		{Key: "AR", Description: "a text answer only, including code, analysis or describing an existing image"},
		{Key: "DIFFUSION", Description: "a newly generated image only"},
		{Key: "BOTH", Description: "a newly generated image together with a separate written explanation"},
	}}
)

// signalQuestions maps each built-in label-distribution consumer to its question.
var signalQuestions = map[string]signalQuestion{
	"domain_classifier":     domainQuestion,
	"prompt_guard":          attackQuestion,
	"fact_check_classifier": factCheckQuestion,
	"feedback_detector":     feedbackQuestion,
	"modality_detector":     modalityQuestion,
}

// questionFor returns the question a binding asks when its deployment serves
// a Vela 2.0 model: the consumer's own, or the harm question for a safety
// rule's binary head.
func questionFor(spec config.ResolvedModelBinding, card modelservice.ModelCard) (signalQuestion, bool) {
	if card.Family != signalQuestionFamily || card.Serves("classify") || !card.Answers("choice") {
		return signalQuestion{}, false
	}
	if question, ok := signalQuestions[spec.Name]; ok {
		return question, true
	}
	if strings.HasPrefix(spec.Name, "safety.") && spec.Binding.Contract == config.RemoteClassifierContractLabelDistribution {
		return harmQuestion, true
	}
	return signalQuestion{}, false
}

func (q signalQuestion) labels() []string {
	labels := make([]string, len(q.options))
	for i, option := range q.options {
		labels[i] = option.Key
	}
	return labels
}

// question is the decisions question a binding asks. Its ID carries the
// consumer's name and a ':', which no decision signal's name contains.
func (q signalQuestion) question(spec config.ResolvedModelBinding) modelservice.Question {
	return modelservice.Question{ID: spec.Name + ":" + q.key, Type: "choice", Instructions: q.instructions, Choices: q.options}
}

// readPolicy is how a binding's question reads a long text: a safety question
// whole up to the declared scan budget (or the model's), a routing question
// only its first tokens.
func readPolicy(request modelservice.Request, safety bool, declared int) modelservice.Request {
	request.MaxTokens = declared
	if !safety {
		questions := make([]modelservice.Question, len(request.Questions))
		for index, question := range request.Questions {
			question.Truncate = true
			questions[index] = question
		}
		request.Questions = questions
	}
	return request
}

// questionScanBudget is the scan budget of a declared question deployment:
// its input {overflow: window, max_tokens}, for a model that reads a long part
// in windows (its card reports max_scan_tokens). Zero keeps the model's own;
// a decision model takes no other input, since it never truncates.
func questionScanBudget(spec config.ResolvedModelBinding, card modelservice.ModelCard) (int, error) {
	if strings.HasPrefix(spec.Binding.Deployment, config.ImplicitDeploymentPrefix) {
		return 0, nil
	}
	if err := spec.Deployment.ValidateDecisionInput(spec.Binding.Deployment); err != nil {
		return 0, fmt.Errorf("%w: %w", binding.ErrCapability, err)
	}
	scan := spec.Deployment.ScanBudget()
	if scan > 0 && card.MaxScanTokens == 0 {
		return 0, fmt.Errorf("%w: deployment %q: the model reads one bounded input and rejects a longer one, so it takes no scan budget; remove input", binding.ErrCapability, spec.Binding.Deployment)
	}
	return scan, nil
}

// questionSequence binds a label-distribution consumer to its question. The
// model reads the whole text, so the binding sets no head and, on a declared
// deployment, at most a scan budget.
func (r *Runtime) questionSequence(ctx context.Context, spec config.ResolvedModelBinding, card modelservice.ModelCard, question signalQuestion) (*binding.Resolved[string, tasks.LabelDistribution], error) {
	decider, ok := r.services.(modelservice.Decider)
	if !ok {
		return nil, fmt.Errorf("%w: the model runtime services of deployment %q answer no decisions", binding.ErrCapability, spec.Binding.Deployment)
	}
	if spec.Binding.Head != "" {
		return nil, fmt.Errorf("%w: deployment %q answers %s as a question, not with a head; remove head", binding.ErrCapability, spec.Binding.Deployment, spec.Name)
	}
	scan, err := questionScanBudget(spec, card)
	if err != nil {
		return nil, err
	}
	resource, err := r.acquire(ctx, spec, card)
	if err != nil {
		return nil, err
	}
	capability := binding.Capability{
		Contract: spec.Binding.Contract, Provider: Provider, Device: card.Device, Precision: card.Dtype, Labels: question.labels(),
		Question: question.key, Deployment: spec.Binding.Deployment,
		Limits: binding.Limits{ModelTokens: card.MaxInputTokens, Overflow: spec.Deployment.Input.Overflow},
	}
	t := &target{spec: spec, deployment: spec.Binding.Deployment, card: card, resource: resource, scan: scan}
	asked := question.question(spec)
	return publish(ctx, r.sequence, t, capability, func(ctx context.Context, _ io.Closer, text string) (tasks.LabelDistribution, error) {
		request := readPolicy(modelservice.Request{State: text, Questions: []modelservice.Question{asked}}, question.safety, t.scan)
		response, err := decider.Decide(ctx, t.deployment, request)
		if err != nil {
			return tasks.LabelDistribution{}, err
		}
		return choiceDistribution(response, asked, t.deployment)
	}, "warmup")
}

// choiceDistribution reads a Choice answer as a distribution in option order.
func choiceDistribution(response modelservice.Response, question modelservice.Question, deployment string) (tasks.LabelDistribution, error) {
	answer, ok := response.Answers[question.ID]
	switch {
	case !ok:
		return tasks.LabelDistribution{}, fmt.Errorf("%w: question %s has no answer", binding.ErrInvalidResult, question.ID)
	case answer.Error != "":
		return tasks.LabelDistribution{}, itemError(answer.Error)
	case answer.Type != "choice":
		return tasks.LabelDistribution{}, fmt.Errorf("%w: deployment %q answered question %s as %s, not choice", binding.ErrInvalidResult, deployment, question.ID, answer.Type)
	}
	probabilities := make([]float32, len(question.Choices))
	for i, option := range question.Choices {
		p, ok := answer.Probabilities[option.Key]
		if !ok {
			return tasks.LabelDistribution{}, fmt.Errorf("%w: the answer to question %s has no probability for %q", binding.ErrInvalidResult, question.ID, option.Key)
		}
		probabilities[i] = float32(p)
	}
	return tasks.LabelDistribution{Probabilities: probabilities}, nil
}

// rejectQuestionWindows refuses a window task on a deployment that answers
// the consumer's question: the model reads the whole text itself.
func (r *Runtime) rejectQuestionWindows(ctx context.Context, spec config.ResolvedModelBinding) error {
	if !spec.Deployment.IsModelRuntime() || r.services == nil {
		return nil
	}
	card, err := r.card(ctx, spec)
	if err != nil {
		return err
	}
	if _, ok := questionFor(spec, card); ok {
		return fmt.Errorf("%w: deployment %q reads the whole text for %s; remove the consumer's window", binding.ErrCapability, spec.Binding.Deployment, spec.Name)
	}
	return nil
}
