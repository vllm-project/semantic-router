package modelservice

// BuiltinTasks is the canonical semantic template registry shared by the
// Router and Playground. Consumers may specialize labels and instructions;
// the resulting question still compiles through CompileTask.
func BuiltinTasks() []TaskDefinition {
	choice := func(id, title, instructions string, full bool, options ...Choice) TaskDefinition {
		return TaskDefinition{
			ID: id, Title: title, Description: instructions, Stage: "request", Input: "text", Output: "choice", FullInput: full,
			Question: Question{ID: id, Type: "choice", Instructions: instructions, Choices: options, Truncate: !full},
		}
	}
	noul := func(id, title, instructions, stage, input string) TaskDefinition {
		return TaskDefinition{
			ID: id, Title: title, Description: instructions, Stage: stage, Input: input, Output: "noul", FullInput: true,
			Question: Question{ID: id, Type: "noul", Instructions: instructions},
		}
	}
	labels := func(id, title, instructions, kind, stage, input string, options []Choice) TaskDefinition {
		return TaskDefinition{
			ID: id, Title: title, Description: instructions, Stage: stage, Input: input, Output: kind, FullInput: true,
			Question: Question{ID: id, Type: kind, Instructions: instructions, Labels: options},
		}
	}
	tasks := []TaskDefinition{
		choice("domain", "Subject area", "Which subject area is this request about?", false,
			Choice{"biology", "living organisms, anatomy, genetics, medical genetics or viruses"},
			Choice{"business", "management, marketing, accounting, business ethics or public relations"},
			Choice{"chemistry", "chemical substances, reactions, elements or laboratory chemistry"},
			Choice{"computer science", "programming, algorithms, computer systems, security or machine learning"},
			Choice{"economics", "markets, macroeconomics, microeconomics or econometrics"},
			Choice{"engineering", "electrical or other engineering design and systems"},
			Choice{"health", "medicine, clinical practice, nutrition, ageing or sexual health"},
			Choice{"history", "past events, periods and historical societies"},
			Choice{"law", "legal rules, jurisprudence, courts or international law"},
			Choice{"math", "arithmetic, algebra, geometry, statistics or other mathematics"},
			Choice{"other", "a subject that fits none of the listed areas"},
			Choice{"philosophy", "philosophy, ethics, moral questions or formal logic"},
			Choice{"physics", "physical laws, mechanics, astronomy or physical phenomena"},
			Choice{"psychology", "mind, behaviour, mental processes or psychological practice"}),
		choice("fact_check", "Need for fact checking", "Does answering this request require checking facts?", false,
			Choice{"NO_FACT_CHECK_NEEDED", "the request can be handled without verifying facts about the world, e.g. translating, rewriting, formatting or writing fiction"},
			Choice{"FACT_CHECK_NEEDED", "answering the request relies on factual claims about the world that should be verified"}),
		choice("user_feedback", "Feedback on the previous answer", "What feedback does this user turn give about the previous answer?", false,
			Choice{"SAT", "the user is satisfied with the previous answer"}, Choice{"NEED_CLARIFICATION", "the user asks for clarification or a more detailed explanation"},
			Choice{"WRONG_ANSWER", "the user says the previous answer was wrong or did not work"}, Choice{"WANT_DIFFERENT", "the user wants a different answer, format, style or approach"}, Choice{"NO_FEEDBACK", "the user gives no feedback on the previous answer"}),
		choice("modality", "Requested output", "What kind of output does this request ask for?", false,
			Choice{"AR", "a text answer only, including code, analysis or describing an existing image"}, Choice{"DIFFUSION", "a newly generated image only"}, Choice{"BOTH", "a newly generated image together with a separate written explanation"}),
		choice("jailbreak", "Instruction attack", "Is this a prompt injection or jailbreak attempt?", true,
			Choice{"benign", "a normal request or quoted content that does not try to override system instructions or bypass safety rules"}, Choice{"jailbreak", "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules"}),
		choice("safety", "Harmful request", "Is this request harmful?", true,
			Choice{"safe", "a benign request that does not violate any safety policy"}, Choice{"unsafe", "a request that violates a safety policy or seeks harmful assistance"}),
		choice("preference", "Preferred response style", "Which response style does the user prefer?", false,
			Choice{"concise", "a short direct answer"}, Choice{"detailed", "a detailed explanation with examples"}),
		choice("classifier", "Custom classification", "Which category best describes the request?", false,
			Choice{"coding", "writing, reviewing or explaining code"}, Choice{"other", "another kind of request"}),
		{
			ID: "complexity", Title: "Task difficulty", Description: "How difficult is this request to solve correctly?", Stage: "request", Input: "text", Output: "score",
			Question: Question{ID: "complexity", Type: "score", Instructions: "How difficult is this request to solve correctly?", Levels: []string{"straightforward retrieval or formatting", "several steps of reasoning", "advanced reasoning or specialized expertise"}, Truncate: true},
		},
		labels("safety_categories", "Harm categories", "Which harmful categories apply to this request?", "set", "request", "text", []Choice{{"violence", "assistance with violence"}, {"fraud", "fraud or deception"}, {"self_harm", "self-harm assistance"}}),
		noul("pii_presence", "Personal data present", "Does any part of the input contain personal or sensitive identifying information?", "request", "text"),
		labels("pii_categories", "Personal data categories", "Which personal-data categories occur in the input?", "set", "request", "text", PIICategoryLabels()),
		labels("pii_spans", "Personal data locations", "Locate personal and sensitive identifying information in the input.", "span", "request", "text", PIICategoryLabels()),
		noul("hallucination", "Grounded answer", "Does the answer contain any factual claim that is unsupported by or contradicts the supplied grounding context? Judge the answer against context, not outside knowledge.", "response", "grounded"),
		labels("hallucination_spans", "Unsupported answer locations", "Locate claims in the answer that are unsupported by or contradict the grounding context. Return locations in the answer only.", "span", "response", "grounded", []Choice{{"unsupported", "a claim not supported by the grounding context"}}),
		noul("reask", "Repeated user intent", "Does the current user request seek the same information or action as the prior user request, including paraphrases and retries that add feedback? A different topic, objective or requested operation is not a repeat.", "request", "pair"),
		choice("model_selection", "Choose a candidate model", "Which candidate model is best suited to answer this request? Choose only a listed candidate.", false,
			Choice{"fast", "fast model for straightforward requests"}, Choice{"reasoning", "model for complex reasoning"}),
	}
	tasks[len(tasks)-1].Stage = "selection"
	for i := range tasks {
		kind, consumer, optional := "signal", tasks[i].ID, false
		switch tasks[i].ID {
		case "pii_presence", "pii_categories":
			consumer = "pii"
		case "pii_spans":
			consumer, optional = "pii", true
		case "safety_categories":
			consumer, optional = "safety", true
		case "hallucination":
			kind = "plugin"
		case "hallucination_spans":
			kind, consumer, optional = "plugin", "hallucination", true
		case "model_selection":
			kind, consumer = "algorithm", "decision"
		}
		binding := map[string]string{"domain": "domain_classifier", "fact_check": "fact_check_classifier", "user_feedback": "feedback_detector", "modality": "modality_detector", "jailbreak": "prompt_guard", "safety": "safety.{name}", "safety_categories": "safety.{name}.hazard", "pii_presence": "pii_classifier", "pii_categories": "pii_classifier", "pii_spans": "pii_classifier", "hallucination": "hallucination_detector", "hallucination_spans": "hallucination_detector", "preference": "preference", "reask": "reask", "complexity": "complexity", "classifier": "classifier.{name}"}[tasks[i].ID]
		tasks[i].Consumers = []TaskConsumerReference{{Kind: kind, Type: consumer, Optional: optional, Binding: binding}}
		if tasks[i].ID == "hallucination" {
			tasks[i].Consumers = append(tasks[i].Consumers, TaskConsumerReference{Kind: "signal", Type: "hallucination", Binding: "hallucination_detector"})
		}
	}
	return tasks
}

func BuiltinTask(id string) (TaskDefinition, bool) {
	for _, definition := range BuiltinTasks() {
		if definition.ID == id {
			return definition, true
		}
	}
	return TaskDefinition{}, false
}

// PIICategoryLabels provides semantic descriptions for common entity labels.
// A positive presence verdict also covers personal information outside them.
func PIICategoryLabels() []Choice {
	return []Choice{
		{"PERSON", "a person's name"},
		{"EMAIL_ADDRESS", "an email address"},
		{"PHONE_NUMBER", "a telephone number"},
		{"US_SSN", "a US social security number"},
		{"CREDIT_CARD", "a credit or debit card number"},
		{"LOCATION", "a private physical address or identifying location"},
		{"IP_ADDRESS", "an IP address"},
		{"DATE_TIME", "a date or time identifying a person"},
		{"US_DRIVER_LICENSE", "a US driving license number"},
		{"US_PASSPORT", "a US passport number"},
		{"IBAN_CODE", "an international bank account number"},
	}
}

// TaskTemplate is native System One wire data for an editable example. It
// deliberately does not promise the chosen model supports that native type.
type TaskTemplate struct {
	State     any            `json:"state"`
	Questions map[string]any `json:"questions"`
}

func (d TaskDefinition) Template() TaskTemplate {
	question := map[string]any{"type": d.Question.Type, "instructions": d.Question.Instructions}
	if d.FullInput {
		question["require_full_input"] = true
	}
	criteria := map[string]string{}
	for _, choice := range append(append([]Choice(nil), d.Question.Choices...), d.Question.Labels...) {
		criteria[choice.Key] = choice.Description
	}
	if len(criteria) > 0 {
		question["criteria"] = criteria
	} else if len(d.Question.Levels) > 0 {
		question["criteria"] = d.Question.Levels
	}
	var state any = "Please explain how binary search works and give a Python example."
	if d.Input == "grounded" {
		state = map[string]string{"context": "The library is open Monday through Friday, 09:00–17:00.", "request": "Is the library open on Sunday?", "answer": "Yes, it is open on Sunday."}
	} else if d.Input == "pair" {
		state = map[string]string{"current": "Could you explain binary search more clearly?", "prior": "How does binary search work?"}
	} else if d.ID == "pii_presence" || d.ID == "pii_categories" || d.ID == "pii_spans" {
		state = "Please email Alex at alex@example.com."
	}
	return TaskTemplate{State: state, Questions: map[string]any{d.ID: question}}
}
