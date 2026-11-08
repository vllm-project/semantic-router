package config

import (
	"slices"
	"strings"
	"testing"
)

func systemOneConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"decider": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Revision: strings.Repeat("a", 40)},
		"bert":    {Provider: "http", ExternalModel: "bert"},
	}
	threshold := 1.0
	cfg.SystemOneRules = []SystemOneRule{
		{Name: "hard", Deployment: "decider", Question: SystemOneQuestion{Type: SystemOneQuestionNoul, Instructions: "Hard?"}},
		{Name: "kind", Deployment: "decider", Question: SystemOneQuestion{Type: SystemOneQuestionChoice, Instructions: "Kind?", Choices: []SystemOneOption{{Key: "code"}, {Key: "math"}}}},
		{Name: "level", Deployment: "decider", Predicate: &NumericPredicate{GTE: &threshold}, Question: SystemOneQuestion{Type: SystemOneQuestionScore, Instructions: "Level?", Levels: []string{"low", "mid", "high"}}},
	}
	return cfg
}

func TestModelRuntimeDeploymentValidation(t *testing.T) {
	valid := []ModelDeployment{
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B"},
		{Provider: ModelRuntimeProvider, Artifact: "/models/kai", Device: "rocm:1"},
		{Provider: ModelRuntimeProvider, Endpoint: "unix:///run/vllm-sr/kai.sock"},
		{Provider: ModelRuntimeProvider, Endpoint: "http://decision-runtime:8100"},
		{Provider: ModelRuntimeProvider, Endpoint: "http://shared-runtime:8100", ServedName: "vela-domain"},
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-PII", Device: "xpu:0", Input: ModelInputBudget{MaxTokens: 32768, Overflow: "window"}},
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "mps", Input: ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}},
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", MaxNumSeqs: intPtr(1), MaxNumBatchedTokens: intPtr(8), BatchWindowMS: ptrFloat64(0), Dtype: "auto", EnableSharedContext: boolPtr(true)},
	}
	for _, deployment := range valid {
		if err := deployment.WithDefaults().validate(&RouterConfig{}); err != nil {
			t.Fatalf("%+v: %v", deployment, err)
		}
	}
	invalid := map[string]ModelDeployment{
		"missing artifact":      {Provider: ModelRuntimeProvider},
		"relative artifact":     {Provider: ModelRuntimeProvider, Artifact: "./models/kai"},
		"short revision":        {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Revision: "881bee41"},
		"local revision":        {Provider: ModelRuntimeProvider, Artifact: "/models/kai", Revision: strings.Repeat("a", 40)},
		"malformed device":      {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Device: "cuda:x"},
		"no rows per forward":   {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", MaxNumSeqs: intPtr(0)},
		"tiny token budget":     {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", MaxNumBatchedTokens: intPtr(4)},
		"negative window":       {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", BatchWindowMS: ptrFloat64(-1)},
		"unknown dtype":         {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Dtype: "fp16"},
		"bad endpoint":          {Provider: ModelRuntimeProvider, Endpoint: "tcp://host:1"},
		"relative socket":       {Provider: ModelRuntimeProvider, Endpoint: "unix://run/x.sock"},
		"negative budget":       {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Input: ModelInputBudget{MaxTokens: -1}},
		"bad overflow":          {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Input: ModelInputBudget{Overflow: "drop"}},
		"attached device":       {Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100", Device: "cuda"},
		"managed served name":   {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", ServedName: "x"},
		"engine args elsewhere": {Provider: "http", ExternalModel: "x", Dtype: "float32"},
		"served name elsewhere": {Provider: "http", ExternalModel: "x", ServedName: "x"},
	}
	for name, deployment := range invalid {
		if err := deployment.WithDefaults().validate(&RouterConfig{}); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
}

func TestModelRuntimeDeploymentDefaults(t *testing.T) {
	deployment := ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x"}.WithDefaults()
	if deployment.Device != "auto" || deployment.HasEngineArgs() || !deployment.Managed() {
		t.Fatalf("defaults = %+v", deployment)
	}
	if (ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://x:1"}).Managed() {
		t.Fatal("an endpoint deployment is attached, not managed")
	}
}

func TestSystemOneSignalContracts(t *testing.T) {
	if err := validateSystemOneContracts(systemOneConfig()); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*RouterConfig){
		"unknown deployment": func(cfg *RouterConfig) { cfg.SystemOneRules[0].Deployment = "missing" },
		"wrong provider":     func(cfg *RouterConfig) { cfg.SystemOneRules[0].Deployment = "bert" },
		"duplicate name":     func(cfg *RouterConfig) { cfg.SystemOneRules[1].Name = "hard" },
		"one choice": func(cfg *RouterConfig) {
			cfg.SystemOneRules[1].Question.Choices = cfg.SystemOneRules[1].Question.Choices[:1]
		},
		"duplicate choice": func(cfg *RouterConfig) { cfg.SystemOneRules[1].Question.Choices[1].Key = "code" },
		"noul keys":        func(cfg *RouterConfig) { cfg.SystemOneRules[0].Question.Choices = []SystemOneOption{{Key: "yes"}} },
		"score predicate":  func(cfg *RouterConfig) { cfg.SystemOneRules[2].Predicate = nil },
		"score levels":     func(cfg *RouterConfig) { cfg.SystemOneRules[2].Question.Levels = []string{"only"} },
		"type":             func(cfg *RouterConfig) { cfg.SystemOneRules[0].Question.Type = "rank" },
		"instructions":     func(cfg *RouterConfig) { cfg.SystemOneRules[0].Question.Instructions = " " },
		"timeout":          func(cfg *RouterConfig) { cfg.SystemOneRules[0].TimeoutMs = MaxSystemOneTimeoutMs + 1 },
		"colon in name":    func(cfg *RouterConfig) { cfg.SystemOneRules[0].Name = "a:b" },
		"input budget": func(cfg *RouterConfig) {
			decider := cfg.ModelDeployments["decider"]
			decider.Input = ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}
			cfg.ModelDeployments["decider"] = decider
		},
	}
	for name, mutate := range cases {
		cfg := systemOneConfig()
		mutate(cfg)
		if err := validateSystemOneContracts(cfg); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
}

// Every rule field is checked against the question it matches and the stage
// it reads, and an error names the rule by its name.
func TestSystemOneRuleFieldErrors(t *testing.T) {
	cases := map[string]struct {
		mutate func(*RouterConfig)
		want   string
	}{
		"labels on a noul":            {func(cfg *RouterConfig) { cfg.SystemOneRules[0].Labels = []string{"true"} }, "a noul rule takes none"},
		"labels on a score":           {func(cfg *RouterConfig) { cfg.SystemOneRules[2].Labels = []string{"1"} }, "a score rule takes none"},
		"ignore_labels on a choice":   {func(cfg *RouterConfig) { cfg.SystemOneRules[1].IgnoreLabels = []string{"code"} }, "ignore_labels apply only to set and span rules"},
		"label not an option":         {func(cfg *RouterConfig) { cfg.SystemOneRules[1].Labels = []string{"chat"} }, `labels: "chat" is not an option of the question`},
		"label twice":                 {func(cfg *RouterConfig) { cfg.SystemOneRules[1].Labels = []string{"code", "code"} }, `labels lists "code" twice`},
		"ignored label not an option": {func(cfg *RouterConfig) { cfg.SystemOneRules[3].IgnoreLabels = []string{"refunds"} }, `ignore_labels: "refunds" is not an option`},
		"labels and ignore_labels overlap": {func(cfg *RouterConfig) {
			cfg.SystemOneRules[3].Labels, cfg.SystemOneRules[3].IgnoreLabels = []string{"billing"}, []string{"billing"}
		}, `"billing" is both in labels and in ignore_labels`},
		"every label ignored": {func(cfg *RouterConfig) {
			cfg.SystemOneRules[3].IgnoreLabels = []string{"billing", "shipping"}
		}, "ignore_labels leaves no label to count"},
		"previous turns above 32":     {func(cfg *RouterConfig) { cfg.SystemOneRules[0].Input.PreviousTurns = 33 }, "input.previous_turns must be within [0, 32]"},
		"negative previous turns":     {func(cfg *RouterConfig) { cfg.SystemOneRules[0].Input.PreviousTurns = -1 }, "input.previous_turns"},
		"answer at the request stage": {func(cfg *RouterConfig) { cfg.SystemOneRules[0].Input.Answer = true }, "a request-stage rule does not have"},
		"context at the request stage": {func(cfg *RouterConfig) {
			cfg.SystemOneRules[0].Stage, cfg.SystemOneRules[0].Input.Context = SystemOneStageRequest, true
		}, "a request-stage rule does not have"},
		"unknown stage":    {func(cfg *RouterConfig) { cfg.SystemOneRules[0].Stage = "routing" }, "stage must be request or response"},
		"unknown coverage": {func(cfg *RouterConfig) { cfg.SystemOneRules[0].Coverage = "whole" }, "coverage must be truncate or full"},
		"on_unscanned without full coverage": {func(cfg *RouterConfig) {
			cfg.SystemOneRules[0].OnUnscanned = SystemOneUnscannedMatch
		}, "on_unscanned applies only to a rule with coverage: full"},
		"unknown on_unscanned": {func(cfg *RouterConfig) {
			cfg.SystemOneRules[0].Coverage, cfg.SystemOneRules[0].OnUnscanned = SystemOneCoverageFull, "block"
		}, "on_unscanned must be match, no_match or unknown"},
	}
	for name, test := range cases {
		cfg := labelledSystemOneConfig()
		test.mutate(cfg)
		err := validateSystemOneContracts(cfg)
		if err == nil || !strings.Contains(err.Error(), test.want) || !strings.HasPrefix(err.Error(), "recipes[default].signals.systemone[") {
			t.Fatalf("%s: got %v, want %q", name, err, test.want)
		}
	}
	accepted := labelledSystemOneConfig()
	accepted.SystemOneRules[0].Stage = SystemOneStageResponse
	accepted.SystemOneRules[0].Input = SystemOneInput{Answer: true, Context: true, History: true, PreviousTurns: MaxSystemOnePreviousTurns, ToolResults: true}
	accepted.SystemOneRules[0].Coverage, accepted.SystemOneRules[0].OnUnscanned = SystemOneCoverageFull, SystemOneUnscannedNoMatch
	accepted.SystemOneRules[1].Labels = []string{"math"}
	accepted.SystemOneRules[3].IgnoreLabels = []string{"shipping"}
	if err := validateSystemOneContracts(accepted); err != nil {
		t.Fatalf("every field within its limits: %v", err)
	}
}

// labelledSystemOneConfig adds a Vela 2.0 deployment with a set and a span question.
func labelledSystemOneConfig() *RouterConfig {
	cfg := systemOneConfig()
	cfg.ModelDeployments["vela"] = ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.8B"}
	threshold := 0.4
	cfg.SystemOneRules = append(cfg.SystemOneRules,
		SystemOneRule{Name: "topics", Deployment: "vela", Question: SystemOneQuestion{
			Type: SystemOneQuestionSet, Instructions: "Which topics does the request mention?", Threshold: &threshold,
			Labels: []SystemOneOption{{Key: "billing", Description: "payments or invoices"}, {Key: "shipping"}},
		}},
		SystemOneRule{Name: "places", Deployment: "vela", Question: SystemOneQuestion{
			Type: SystemOneQuestionSpan, Instructions: "Which spans name a place?", Head: SystemOneSpanHeadBroad,
			Labels: []SystemOneOption{{Key: "city", Description: "a city"}},
		}},
	)
	return cfg
}

func TestSetAndSpanQuestionContracts(t *testing.T) {
	if err := validateSystemOneContracts(labelledSystemOneConfig()); err != nil {
		t.Fatal(err)
	}
	set, span := 3, 4
	cases := map[string]func(*RouterConfig){
		"no labels":           func(cfg *RouterConfig) { cfg.SystemOneRules[set].Question.Labels = nil },
		"duplicate label":     func(cfg *RouterConfig) { cfg.SystemOneRules[set].Question.Labels[1].Key = "billing" },
		"untrimmed label":     func(cfg *RouterConfig) { cfg.SystemOneRules[span].Question.Labels[0].Key = " city" },
		"choices on a set":    func(cfg *RouterConfig) { cfg.SystemOneRules[set].Question.Choices = []SystemOneOption{{Key: "a"}} },
		"levels on a span":    func(cfg *RouterConfig) { cfg.SystemOneRules[span].Question.Levels = []string{"a", "b"} },
		"threshold above one": func(cfg *RouterConfig) { high := 1.5; cfg.SystemOneRules[set].Question.Threshold = &high },
		"unknown head":        func(cfg *RouterConfig) { cfg.SystemOneRules[span].Question.Head = "wide" },
		"head on a set":       func(cfg *RouterConfig) { cfg.SystemOneRules[set].Question.Head = SystemOneSpanHeadRouter },
		"labels on a choice":  func(cfg *RouterConfig) { cfg.SystemOneRules[1].Question.Labels = []SystemOneOption{{Key: "x"}} },
		"threshold on a noul": func(cfg *RouterConfig) { low := 0.2; cfg.SystemOneRules[0].Question.Threshold = &low },
		"too many labels": func(cfg *RouterConfig) {
			labels := make([]SystemOneOption, MaxSystemOneLabels+1)
			for index := range labels {
				labels[index] = SystemOneOption{Key: strings.Repeat("l", index+1)}
			}
			cfg.SystemOneRules[set].Question.Labels = labels
		},
		"label answer key": func(cfg *RouterConfig) {
			cfg.SystemOneRules = append(cfg.SystemOneRules, SystemOneRule{Name: "topics.billing", Deployment: "vela", Question: SystemOneQuestion{Type: SystemOneQuestionNoul, Instructions: "?"}})
		},
	}
	for name, mutate := range cases {
		cfg := labelledSystemOneConfig()
		mutate(cfg)
		if err := validateSystemOneContracts(cfg); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
	cfg := labelledSystemOneConfig()
	cfg.SystemOneRules = append(cfg.SystemOneRules, SystemOneRule{Name: "topics.billing", Deployment: "decider", Question: SystemOneQuestion{Type: SystemOneQuestionNoul, Instructions: "?"}})
	if err := validateSystemOneContracts(cfg); err != nil {
		t.Fatalf("a label answer key only collides within one deployment's call: %v", err)
	}
}

// A set or span rule matches itself over its counted labels, so a condition
// may name it without a label; a label must be an option that counts.
func TestSetAndSpanConditionsNameACountedLabel(t *testing.T) {
	cfg := labelledSystemOneConfig()
	cfg.SystemOneRules[3].IgnoreLabels = []string{"billing"}
	accept := []RuleNode{
		{Type: SignalTypeSystemOne, Name: "topics"},
		{Type: SignalTypeSystemOne, Name: "places"},
		{Type: SignalTypeSystemOne, Name: "topics", Label: "shipping"},
		{Type: SignalTypeSystemOne, Name: "places", Label: "city"},
	}
	for _, node := range accept {
		if err := validateDecisionLeafNode(cfg, "d", &node); err != nil {
			t.Fatalf("%+v: %v", node, err)
		}
	}
	reject := map[string]struct {
		node RuleNode
		want string
	}{
		"undeclared set label":   {RuleNode{Type: SignalTypeSystemOne, Name: "topics", Label: "refunds"}, "not an option of the question"},
		"undeclared span label":  {RuleNode{Type: SignalTypeSystemOne, Name: "places", Label: "PERSON"}, "not an option of the question"},
		"a choice key on a span": {RuleNode{Type: SignalTypeSystemOne, Name: "places", Label: "code"}, "not an option of the question"},
		"an ignored label":       {RuleNode{Type: SignalTypeSystemOne, Name: "topics", Label: "billing"}, "ignore_labels never count"},
	}
	for name, test := range reject {
		err := validateDecisionLeafNode(cfg, "d", &test.node)
		if err == nil || !strings.Contains(err.Error(), test.want) {
			t.Fatalf("%s: got %v, want %q", name, err, test.want)
		}
	}
}

// A choice rule without labels has no match of its own: its conditions name
// the chosen option. With labels, the rule matches when one of them is chosen.
func TestSystemOneConditionLabels(t *testing.T) {
	cfg := systemOneConfig()
	accept := []RuleNode{
		{Type: SignalTypeSystemOne, Name: "hard"},
		{Type: SignalTypeSystemOne, Name: "level"},
		{Type: SignalTypeSystemOne, Name: "kind", Label: "math"},
		{Type: SignalTypeSystemOne, Name: "hard", OnError: "match"},
	}
	for _, node := range accept {
		if err := validateDecisionLeafNode(cfg, "d", &node); err != nil {
			t.Fatalf("%+v: %v", node, err)
		}
	}
	reject := []RuleNode{
		{Type: SignalTypeSystemOne, Name: "missing"},
		{Type: SignalTypeSystemOne, Name: "kind"},
		{Type: SignalTypeSystemOne, Name: "kind", Label: "chat"},
		{Type: SignalTypeSystemOne, Name: "hard", Label: "true"},
		{Type: SignalTypeKeyword, Name: "kw", Label: "x"},
	}
	for _, node := range reject {
		if err := validateDecisionLeafNode(cfg, "d", &node); err == nil {
			t.Fatalf("%+v: expected a validation error", node)
		}
	}
	cfg.SystemOneRules[1].Labels = []string{"code"}
	if err := validateDecisionLeafNode(cfg, "d", &RuleNode{Type: SignalTypeSystemOne, Name: "kind"}); err != nil {
		t.Fatalf("a choice rule with labels matches itself: %v", err)
	}
}

func TestCountedLabels(t *testing.T) {
	cfg := labelledSystemOneConfig()
	choice, set := cfg.SystemOneRules[1], cfg.SystemOneRules[3]
	if labels := choice.CountedLabels(); labels != nil || choice.MatchesItself() {
		t.Fatalf("a choice rule without labels counts none: %v", labels)
	}
	choice.Labels = []string{"math"}
	if labels := choice.CountedLabels(); !slices.Equal(labels, []string{"math"}) || !choice.MatchesItself() {
		t.Fatalf("choice labels: %v", labels)
	}
	if labels := set.CountedLabels(); !slices.Equal(labels, []string{"billing", "shipping"}) {
		t.Fatalf("a set rule counts every label by default: %v", labels)
	}
	set.IgnoreLabels = []string{"billing"}
	if labels := set.CountedLabels(); !slices.Equal(labels, []string{"shipping"}) {
		t.Fatalf("ignored labels never count: %v", labels)
	}
	if cfg.SystemOneRules[0].CountedLabels() != nil || !cfg.SystemOneRules[0].MatchesItself() {
		t.Fatal("a noul rule counts no label and matches itself")
	}
}

// A span rule needs a deployment that answers span questions, and a rule that
// reads media one whose model reads the medium: the generated catalog tells
// for a built-in decision model, the declared capabilities for a systemone
// deployment, and the card when the generation prepares for anything else.
func TestSystemOneAnswererLoadChecks(t *testing.T) {
	span := SystemOneRule{Name: "accounts", Question: SystemOneQuestion{Type: SystemOneQuestionSpan, Instructions: "Account numbers?", Labels: []SystemOneOption{{Key: "account"}}}}
	image := SystemOneRule{Name: "chart", Input: SystemOneInput{Image: true}, Question: SystemOneQuestion{Type: SystemOneQuestionNoul, Instructions: "Is it a chart?"}}
	video := image
	video.Name, video.Input = "clip", SystemOneInput{Video: true}
	audio := image
	audio.Name, audio.Input = "voice", SystemOneInput{Audio: true}
	deployments := map[string]ModelDeployment{
		"kai":      {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B"},
		"vela2":    {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B"},
		"fast":     {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/vela3-fast"},
		"lite":     {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/vela3-lite"},
		"d3":       {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/d3"},
		"attached": {Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100"},
		"custom":   {Provider: ModelRuntimeProvider, Artifact: "/models/custom"},
		"remote":   {Provider: SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra", Capabilities: &DeploymentCapabilities{QuestionTypes: []string{"choice", "noul", "score"}, Modalities: []string{"text", "image"}}},
		"silent":   {Provider: SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra"},
	}
	cases := []struct {
		rule       SystemOneRule
		deployment string
		code       string
	}{
		{span, "kai", "span_required"},
		{span, "d3", "span_required"},
		{span, "remote", "span_required"},
		{span, "vela2", ""},
		{span, "fast", ""},
		{span, "lite", ""},
		{span, "attached", ""},
		{span, "custom", ""},
		{span, "silent", ""},
		{image, "fast", "input_modality_unsupported"},
		{image, "kai", "input_modality_unsupported"},
		{image, "d3", ""},
		{image, "lite", ""},
		{image, "remote", ""},
		{image, "silent", ""},
		{video, "d3", ""},
		{video, "remote", "input_modality_unsupported"},
		{audio, "d3", "input_modality_unsupported"},
		{audio, "attached", ""},
	}
	for _, test := range cases {
		cfg := &RouterConfig{}
		cfg.ModelDeployments = deployments
		rule := test.rule
		rule.Deployment = test.deployment
		cfg.SystemOneRules = []SystemOneRule{rule}
		err := validateSystemOneContracts(cfg)
		if test.code == "" {
			if err != nil {
				t.Fatalf("%s on %s: %v", rule.Name, test.deployment, err)
			}
			continue
		}
		prefix := "recipes[default].signals.systemone[" + rule.Name + "]: " + test.code + ": "
		if err == nil || !strings.HasPrefix(err.Error(), prefix) || !strings.Contains(err.Error(), `deployment "`+test.deployment+`"`) || strings.Contains(err.Error(), "\n") {
			t.Fatalf("%s on %s: got %v, want one line starting %q", rule.Name, test.deployment, err, prefix)
		}
	}
	err := CheckSystemOneAnswerer("recipes[default].signals.systemone[accounts]", "primary", span, []string{"choice", "noul", "score"}, []string{"text"})
	want := `recipes[default].signals.systemone[accounts]: span_required: resolves to deployment "primary", which cannot answer span questions; bind it to a span-capable deployment (rule deployment)`
	if err == nil || err.Error() != want {
		t.Fatalf("span_required: %v", err)
	}

	recipe := &RouterConfig{RoutingScope: "support"}
	recipe.ModelDeployments = deployments
	span.Deployment = "kai"
	recipe.SystemOneRules = []SystemOneRule{span}
	err = wrapRoutingProfileValidationError("support", validateSystemOneContracts(recipe))
	if err == nil || !strings.HasPrefix(err.Error(), "recipes[support].signals.systemone[accounts]: span_required: ") {
		t.Fatalf("a recipe's rule: %v", err)
	}
}

func TestDeclaredModalitiesOfASystemOneDeployment(t *testing.T) {
	base := ModelDeployment{Provider: SystemOneProvider, Endpoint: "https://api.example.com/v1/systemone", Model: "vllm-sr/d3-ultra", Auth: &DeploymentAuth{APIKeyEnv: "KEY"}}
	for _, modalities := range [][]string{{"text", "image", "video"}, {"text"}} {
		deployment := base
		deployment.Capabilities = &DeploymentCapabilities{Modalities: modalities}
		if err := deployment.validateSystemOne(); err != nil {
			t.Fatalf("%v: %v", modalities, err)
		}
	}
	for want, modalities := range map[string][]string{
		"carries no audio part": {"text", "audio"},
		`lists "smell"`:         {"smell"},
		`lists "image" twice`:   {"image", "image"},
	} {
		deployment := base
		deployment.Capabilities = &DeploymentCapabilities{Modalities: modalities}
		if err := deployment.validateSystemOne(); err == nil || !strings.Contains(err.Error(), want) {
			t.Fatalf("%v: got %v, want %q", modalities, err, want)
		}
	}
}

// A response-stage rule is asked about the model's answer, after routing: a
// decision cannot read it, directly or through a projection.
func TestResponseStageSystemOneRulesDoNotRoute(t *testing.T) {
	cfg := systemOneConfig()
	cfg.SystemOneRules[0].Stage = SystemOneStageResponse
	cfg.SystemOneRules[0].Input.Answer = true
	if err := validateSystemOneContracts(cfg); err != nil {
		t.Fatal(err)
	}
	cfg.Decisions = []Decision{{Name: "route", Rules: RuleNode{Type: SignalTypeSystemOne, Name: "hard"}}}
	err := validateDecisionsReadRequestStageSignals(cfg)
	if err == nil || !strings.Contains(err.Error(), `reads systemone rule "hard"`) || !strings.Contains(err.Error(), "response stage") {
		t.Fatalf("a decision reading a response-stage rule: %v", err)
	}
	if cfg.SignalStageOf(SignalTypeSystemOne, "hard") != SignalStageResponse || cfg.SignalStageOf(SignalTypeSystemOne, "kind:code") != SignalStageRequest {
		t.Fatal("a systemone rule is observed at its own stage")
	}
}

func TestSystemOneSelectorContract(t *testing.T) {
	refs := []ModelRef{{Model: "large"}, {Model: "small"}}
	ok := &AlgorithmConfig{Type: DecisionAlgorithmSystemOne, SystemOne: &SystemOneSelectionConfig{
		Deployment: "decider", Instructions: "Which model?", Candidates: map[string]string{"large": "Best quality"},
	}}
	if err := validateSystemOneSelectorConfig("d", refs, ok); err != nil {
		t.Fatal(err)
	}
	ok.SystemOne.Deployment = ""
	if err := validateSystemOneSelectorConfig("d", refs, ok); err != nil {
		t.Fatalf("a selector without a deployment asks the decision model: %v", err)
	}
	modelFree := &AlgorithmConfig{Type: DecisionAlgorithmSystemOne, SystemOne: &SystemOneSelectionConfig{Instructions: "Which?"}}
	if err := validateSystemOneSelectorConfig("d", nil, modelFree); err != nil {
		t.Fatalf("a model-free built-in Recipe receives its candidates through assignments: %v", err)
	}
	modelFree.SystemOne.Candidates = map[string]string{"large": "Best quality"}
	if err := validateSystemOneSelectorConfig("d", nil, modelFree); err == nil {
		t.Fatal("candidates must name assigned modelRefs")
	}
	cases := map[string]func(*AlgorithmConfig) []ModelRef{
		"blank deployment":  func(a *AlgorithmConfig) []ModelRef { a.SystemOne.Deployment = " "; return refs },
		"padded deployment": func(a *AlgorithmConfig) []ModelRef { a.SystemOne.Deployment = " decider"; return refs },
		"missing block":     func(a *AlgorithmConfig) []ModelRef { a.SystemOne = nil; return refs },
		"one candidate":     func(a *AlgorithmConfig) []ModelRef { return refs[:1] },
		"duplicate model":   func(a *AlgorithmConfig) []ModelRef { return []ModelRef{{Model: "x"}, {Model: "x"}} },
		"unknown candidate": func(a *AlgorithmConfig) []ModelRef {
			a.SystemOne.Candidates = map[string]string{"other": "x"}
			return refs
		},
		"empty instructions": func(a *AlgorithmConfig) []ModelRef { a.SystemOne.Instructions = ""; return refs },
	}
	for name, mutate := range cases {
		algorithm := &AlgorithmConfig{Type: DecisionAlgorithmSystemOne, SystemOne: &SystemOneSelectionConfig{Deployment: "decider", Instructions: "Which?"}}
		candidates := mutate(algorithm)
		if err := validateSystemOneSelectorConfig("d", candidates, algorithm); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
}

func TestModelRuntimeDeploymentsInUse(t *testing.T) {
	cfg := systemOneConfig()
	cfg.ModelDeployments["selector"] = ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://selector:8100"}
	cfg.ModelDeployments["idle"] = ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/idle"}
	cfg.Decisions = []Decision{{Name: "route", Algorithm: &AlgorithmConfig{Type: DecisionAlgorithmSystemOne, SystemOne: &SystemOneSelectionConfig{Deployment: "selector"}}}}
	used := ModelRuntimeDeploymentsInUse(cfg)
	if len(used) != 2 || used["decider"].Device != "auto" || used["selector"].Endpoint == "" {
		t.Fatalf("in use = %+v", used)
	}
	if _, idle := used["idle"]; idle {
		t.Fatal("unreferenced deployments must not start")
	}
}

func TestSystemOneRuleDefaults(t *testing.T) {
	rule := SystemOneRule{Question: SystemOneQuestion{Type: SystemOneQuestionNoul}}
	predicate := rule.EffectivePredicate()
	if predicate == nil || predicate.GTE == nil || *predicate.GTE != DefaultSystemOneNoulThreshold {
		t.Fatalf("noul default predicate = %+v", predicate)
	}
	if (SystemOneRule{Question: SystemOneQuestion{Type: SystemOneQuestionChoice}}).EffectivePredicate() != nil {
		t.Fatal("a choice rule has no default predicate")
	}
	if rule.EffectiveTimeout().Milliseconds() != DefaultSystemOneTimeoutMs {
		t.Fatal("default timeout")
	}
	if rule.EffectiveStage() != SystemOneStageRequest || rule.FullCoverage() || rule.EffectiveOnUnscanned() != SystemOneUnscannedUnknown {
		t.Fatal("a rule reads the request, truncated, and an unscanned answer is unknown")
	}
	if media := (SystemOneInput{Image: true, Video: true}).Media(); !slices.Equal(media, []string{"image", "video"}) {
		t.Fatalf("media = %v", media)
	}
}
