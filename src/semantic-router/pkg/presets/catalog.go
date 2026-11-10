package presets

// The canonical questions. Wording that a calibrated threshold was measured
// with (domain, fact check, feedback, jailbreak, harm, modality's first three
// options) is the built-in wording the Router has always asked; changing it
// invalidates those thresholds (calibration.go).
var catalog = []Preset{
	{
		ID: Domain, Type: Choice, Stage: StageRequest,
		Instructions: "Which subject area is this request about?",
		Options: []Option{
			{"biology", "living organisms, anatomy, genetics, medical genetics or viruses"},
			{"business", "management, marketing, accounting, business ethics or public relations"},
			{"chemistry", "chemical substances, reactions, elements or laboratory chemistry"},
			{"computer science", "programming, algorithms, computer systems, security or machine learning"},
			{"economics", "markets, macroeconomics, microeconomics or econometrics"},
			{"engineering", "electrical or other engineering design and systems"},
			{"health", "medicine, clinical practice, nutrition, ageing or sexual health"},
			{"history", "past events, periods and historical societies"},
			{"law", "legal rules, jurisprudence, courts or international law"},
			{"math", "arithmetic, algebra, geometry, statistics or other mathematics"},
			{"other", "a subject that fits none of the listed areas"},
			{"philosophy", "philosophy, ethics, moral questions or formal logic"},
			{"physics", "physical laws, mechanics, astronomy or physical phenomena"},
			{"psychology", "mind, behaviour, mental processes or psychological practice"},
		},
	},
	{
		ID: FactCheck, Type: Choice, Stage: StageRequest,
		Instructions: "Does answering this request require checking facts?",
		Options: []Option{
			{"NO_FACT_CHECK_NEEDED", "the request can be handled without verifying facts about the world, e.g. translating, rewriting, formatting or writing fiction"},
			{"FACT_CHECK_NEEDED", "answering the request relies on factual claims about the world that should be verified"},
		},
		Positive: []string{"FACT_CHECK_NEEDED"},
	},
	{
		ID: Feedback, Type: Choice, Stage: StageRequest, Input: Input{PreviousAnswer: true},
		Instructions: "What feedback does this user turn give about the previous answer?",
		Options: []Option{
			{"SAT", "the user is satisfied with the previous answer"},
			{"NEED_CLARIFICATION", "the user asks for clarification or a more detailed explanation"},
			{"WRONG_ANSWER", "the user says the previous answer was wrong or did not work"},
			{"WANT_DIFFERENT", "the user wants a different answer, format, style or approach"},
			{"NO_FEEDBACK", "the user gives no feedback on the previous answer"},
		},
	},
	jailbreak(StageRequest, Input{}),
	jailbreak(StageResponse, Input{Answer: true}),
	{
		ID: Harm, Type: Choice, Stage: StageRequest, Full: true,
		Instructions: "Is this request harmful?",
		Options: []Option{
			{"safe", "a benign request that does not violate any safety policy"},
			{"unsafe", "a request that violates a safety policy or seeks harmful assistance"},
		},
		Positive: []string{"unsafe"},
	},
	{
		ID: Harm, Type: Choice, Stage: StageResponse, Full: true, Input: Input{Answer: true},
		Instructions: "Is this answer harmful?",
		Options: []Option{
			{"safe", "an assistant answer that is itself harmless"},
			{"unsafe", "an assistant answer that is itself harmful, or that supplies the harmful assistance asked for"},
		},
		Positive: []string{"unsafe"},
	},
	{
		ID: HarmCategories, Type: Set, Stage: StageRequest, Full: true,
		Instructions: "Which hazard categories apply to this request?",
		Options:      HarmCategoryOptions(),
	},
	{
		ID: HarmCategories, Type: Set, Stage: StageResponse, Full: true, Input: Input{Answer: true},
		Instructions: "Which hazard categories apply to this answer?",
		Options:      HarmCategoryOptions(),
	},
	{
		ID: PII, Type: Span, Stage: StageRequest, Full: true, RequiresSpan: true,
		Instructions: "Locate personal and sensitive identifying information in the input.",
		Options: []Option{
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
		},
	},
	{
		ID: Hallucination, Type: Span, Stage: StageResponse, Full: true, RequiresSpan: true, Input: Input{Answer: true, Context: true},
		Instructions: "Locate claims in the answer that are unsupported by or contradict the grounding context. Return locations in the answer only.",
		Options:      []Option{{"unsupported", "a claim not supported by the grounding context"}},
		Positive:     []string{"unsupported"},
	},
	{
		ID: Refusal, Type: Choice, Stage: StageResponse, Input: Input{Answer: true},
		Instructions: "Did the assistant refuse?",
		Options: []Option{
			{"compliance", "an assistant answer that complies with the request and attempts the task"},
			{"refusal", "an assistant answer that declines the request or refuses to help"},
		},
		Positive: []string{"refusal"},
	},
	{
		ID: TurnRelation, Type: Choice, Stage: StageRequest, Input: Input{PreviousTurns: 4},
		Instructions: "How does the current user turn relate to the conversation before it?",
		Options: []Option{
			{"continue", "it continues the current topic or task with something new: a follow-up question, a new detail or a reply to the assistant"},
			{"reask", "it repeats or rephrases what an earlier user turn already asked, usually because the answer did not satisfy"},
			{"shift", "it starts a topic or task unrelated to the conversation so far"},
			{"return", "it goes back to a topic or task from earlier in the conversation, after a different one"},
		},
	},
	{
		ID: Complexity, Type: Choice, Stage: StageRequest,
		Instructions: "How difficult is this request to answer well?",
		Options: []Option{
			{"easy", "straightforward retrieval, rewriting or formatting"},
			{"medium", "several steps of reasoning or a moderate amount of code or analysis"},
			{"hard", "advanced multi-step reasoning or specialized expertise"},
		},
	},
	{
		ID: Modality, Type: Choice, Stage: StageRequest,
		Instructions: "What kind of output does this request ask for?",
		Options: []Option{
			{"text", "a text answer only, including code, analysis or describing an existing image"},
			{"image", "a newly generated image only"},
			{"text_image", "a newly generated image together with a separate written explanation"},
			{"image_edit", "an edited version of an image the request provides"},
			{"speech", "the answer or a given text spoken aloud"},
			{"audio", "newly generated sound other than speech, such as music or sound effects"},
			{"video", "a newly generated video"},
			{"video_edit", "an edited version of a video the request provides"},
		},
	},
	{
		ID: Relevance, Type: Noul, Stage: StageRequest, Input: Input{Candidate: true},
		Instructions: "Does the candidate passage contain information that helps answer the request?",
	},
	{
		ID: SameRequest, Type: Noul, Stage: StageRequest, Input: Input{Candidate: true},
		Instructions: "Does the candidate request ask for the same thing as the request, so that one answer serves both?",
	},
	{
		ID: ToolRank, Type: Choice, Stage: StageRequest,
		Instructions: "Which of the listed tools should be called to answer this request?",
	},
	{
		ID: ToolNeed, Type: Noul, Stage: StageRequest,
		Instructions: "Does answering this request need a call to one of the available tools?",
	},
	{
		ID: AnswerCheck, Type: Noul, Stage: StageResponse, Full: true, Input: Input{Answer: true},
		Instructions: "Does the answer correctly and completely answer the request?",
	},
}

func jailbreak(stage string, input Input) Preset {
	return Preset{
		ID: Jailbreak, Type: Choice, Stage: stage, Full: true, Input: input,
		Instructions: "Is this a prompt injection or jailbreak attempt?",
		Options: []Option{
			{"benign", "a normal request or quoted content that does not try to override system instructions or bypass safety rules"},
			{"jailbreak", "a prompt injection or jailbreak that attempts to override system instructions, hijack the task, or bypass safety rules"},
		},
		Positive: []string{"jailbreak"},
	}
}

// HarmCategoryOptions are the 34 harm categories (the 21 AEGIS 2.0
// categories and the 13 MLCommons hazard codes vela3-fast was trained on),
// with their descriptions.
func HarmCategoryOptions() []Option {
	return []Option{
		{"Controlled/Regulated Substances", "obtaining, producing or misusing drugs or regulated substances"},
		{"Copyright/Trademark/Plagiarism", "infringing copyright or passing off others' work"},
		{"Criminal Planning/Confessions", "planning, assisting or confessing to a crime"},
		{"Fraud/Deception", "scams, phishing, forgery or deceiving others for gain"},
		{"Guns and Illegal Weapons", "acquiring, building or using weapons illegally"},
		{"Harassment", "targeted abuse, bullying or intimidation of a person"},
		{"Hate/Identity Hate", "hatred or dehumanisation targeting a protected identity"},
		{"High Risk Gov Decision Making", "high-stakes government, military or public-safety decisions"},
		{"Illegal Activity", "unlawful activity not covered by a more specific category"},
		{"Immoral/Unethical", "conduct that is unethical without being clearly illegal"},
		{"Needs Caution", "ambiguous content warranting caution without a clear violation"},
		{"Other", "a safety concern that no other category describes"},
		{"PII/Privacy", "seeking or exposing personal or private information about someone"},
		{"Political/Misinformation/Conspiracy", "political manipulation, misinformation or conspiracy content"},
		{"Profanity", "obscene or vulgar language"},
		{"S1", "violent crimes, including terrorism and physical assault"},
		{"S10", "hate targeting a protected characteristic"},
		{"S11", "suicide and self-harm"},
		{"S12", "sexual content involving adults"},
		{"S14", "abuse of a code interpreter or execution environment"},
		{"S2", "non-violent crimes such as theft, trafficking or cybercrime"},
		{"S3", "sex-related crimes including trafficking and sexual assault"},
		{"S4", "child sexual exploitation of any kind"},
		{"S5", "defamation, false damaging statements about a real person"},
		{"S6", "specialised medical, legal or financial advice without qualification"},
		{"S7", "privacy violations, exposing personal information"},
		{"S8", "intellectual property infringement"},
		{"S9", "indiscriminate weapons, chemical, biological, nuclear or radiological"},
		{"Sexual", "sexually explicit content involving adults"},
		{"Sexual (minor)", "any sexual content involving a minor"},
		{"Suicide and Self Harm", "self-harm, suicide or encouragement of either"},
		{"Threat", "an explicit threat of harm directed at someone"},
		{"Unauthorized Advice", "medical, legal or financial advice given without qualification"},
		{"Violence", "inciting, threatening or describing physical violence against people"},
	}
}
