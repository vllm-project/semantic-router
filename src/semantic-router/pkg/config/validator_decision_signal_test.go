package config

import (
	"strings"
	"testing"
)

func decisionSignalConfig() *RouterConfig {
	cfg := &RouterConfig{}
	cfg.ModelDeployments = map[string]ModelDeployment{
		"decider": {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B", Revision: strings.Repeat("a", 40)},
		"bert":    {Provider: "http", ExternalModel: "bert"},
	}
	threshold := 1.0
	cfg.DecisionRules = []DecisionSignalRule{
		{Name: "hard", Deployment: "decider", Question: DecisionQuestion{Type: DecisionQuestionNoul, Instructions: "Hard?"}},
		{Name: "kind", Deployment: "decider", Question: DecisionQuestion{Type: DecisionQuestionChoice, Instructions: "Kind?", Choices: []DecisionChoice{{Key: "code"}, {Key: "math"}}}},
		{Name: "level", Deployment: "decider", Predicate: &NumericPredicate{GTE: &threshold}, Question: DecisionQuestion{Type: DecisionQuestionScore, Instructions: "Level?", Levels: []string{"low", "mid", "high"}}},
	}
	return cfg
}

func TestModelRuntimeDeploymentValidation(t *testing.T) {
	valid := []ModelDeployment{
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Decision-2.0-Kai-0.6B"},
		{Provider: ModelRuntimeProvider, Artifact: "/models/kai", Device: "rocm:1", Profile: "shared_context"},
		{Provider: ModelRuntimeProvider, Endpoint: "unix:///run/vllm-sr/kai.sock"},
		{Provider: ModelRuntimeProvider, Endpoint: "http://decision-runtime:8100", Device: "cuda"},
		{Provider: ModelRuntimeProvider, Endpoint: "http://shared-runtime:8100", ServedName: "vela-domain"},
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-PII", Device: "xpu:0", Process: "encoders", Input: ModelInputBudget{MaxTokens: 32768, Overflow: "window"}},
		{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-1.0-Encoder-307M-Domain", Device: "mps", Input: ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}},
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
		"malformed profile":     {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Profile: "Turbo"},
		"bad endpoint":          {Provider: ModelRuntimeProvider, Endpoint: "tcp://host:1"},
		"relative socket":       {Provider: ModelRuntimeProvider, Endpoint: "unix://run/x.sock"},
		"negative budget":       {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Input: ModelInputBudget{MaxTokens: -1}},
		"bad overflow":          {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Input: ModelInputBudget{Overflow: "drop"}},
		"attached process":      {Provider: ModelRuntimeProvider, Endpoint: "http://runtime:8100", Process: "encoders"},
		"managed served name":   {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", ServedName: "x"},
		"bad process":           {Provider: ModelRuntimeProvider, Artifact: "vllm-sr/x", Process: "../escape"},
		"profile elsewhere":     {Provider: "http", ExternalModel: "x", Profile: "exact"},
		"process elsewhere":     {Provider: "http", ExternalModel: "x", Process: "encoders"},
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
	if deployment.Device != "auto" || deployment.Profile != "exact" || !deployment.Managed() {
		t.Fatalf("defaults = %+v", deployment)
	}
	if (ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://x:1"}).Managed() {
		t.Fatal("an endpoint deployment is attached, not managed")
	}
}

func TestDecisionSignalContracts(t *testing.T) {
	if err := validateDecisionSignalContracts(decisionSignalConfig()); err != nil {
		t.Fatal(err)
	}
	cases := map[string]func(*RouterConfig){
		"unknown deployment": func(cfg *RouterConfig) { cfg.DecisionRules[0].Deployment = "missing" },
		"wrong provider":     func(cfg *RouterConfig) { cfg.DecisionRules[0].Deployment = "bert" },
		"duplicate name":     func(cfg *RouterConfig) { cfg.DecisionRules[1].Name = "hard" },
		"one choice": func(cfg *RouterConfig) {
			cfg.DecisionRules[1].Question.Choices = cfg.DecisionRules[1].Question.Choices[:1]
		},
		"duplicate choice": func(cfg *RouterConfig) { cfg.DecisionRules[1].Question.Choices[1].Key = "code" },
		"noul keys":        func(cfg *RouterConfig) { cfg.DecisionRules[0].Question.Choices = []DecisionChoice{{Key: "yes"}} },
		"score predicate":  func(cfg *RouterConfig) { cfg.DecisionRules[2].Predicate = nil },
		"score levels":     func(cfg *RouterConfig) { cfg.DecisionRules[2].Question.Levels = []string{"only"} },
		"type":             func(cfg *RouterConfig) { cfg.DecisionRules[0].Question.Type = "rank" },
		"instructions":     func(cfg *RouterConfig) { cfg.DecisionRules[0].Question.Instructions = " " },
		"timeout":          func(cfg *RouterConfig) { cfg.DecisionRules[0].TimeoutMs = MaxDecisionTimeoutMs + 1 },
		"colon in name":    func(cfg *RouterConfig) { cfg.DecisionRules[0].Name = "a:b" },
		"input budget": func(cfg *RouterConfig) {
			decider := cfg.ModelDeployments["decider"]
			decider.Input = ModelInputBudget{MaxTokens: 512, Overflow: "truncate"}
			cfg.ModelDeployments["decider"] = decider
		},
	}
	for name, mutate := range cases {
		cfg := decisionSignalConfig()
		mutate(cfg)
		if err := validateDecisionSignalContracts(cfg); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
}

// labelledDecisionConfig adds a Vela 2.0 deployment with a set and a span question.
func labelledDecisionConfig() *RouterConfig {
	cfg := decisionSignalConfig()
	cfg.ModelDeployments["vela"] = ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/Vela-2.0-0.3B"}
	threshold := 0.4
	cfg.DecisionRules = append(cfg.DecisionRules,
		DecisionSignalRule{Name: "topics", Deployment: "vela", Question: DecisionQuestion{
			Type: DecisionQuestionSet, Instructions: "Which topics does the request mention?", Threshold: &threshold,
			Labels: []DecisionChoice{{Key: "billing", Description: "payments or invoices"}, {Key: "shipping"}},
		}},
		DecisionSignalRule{Name: "places", Deployment: "vela", Question: DecisionQuestion{
			Type: DecisionQuestionSpan, Instructions: "Which spans name a place?", Head: DecisionSpanHeadBroad,
			Labels: []DecisionChoice{{Key: "city", Description: "a city"}},
		}},
	)
	return cfg
}

func TestSetAndSpanQuestionContracts(t *testing.T) {
	if err := validateDecisionSignalContracts(labelledDecisionConfig()); err != nil {
		t.Fatal(err)
	}
	set, span := 3, 4
	cases := map[string]func(*RouterConfig){
		"no labels":           func(cfg *RouterConfig) { cfg.DecisionRules[set].Question.Labels = nil },
		"duplicate label":     func(cfg *RouterConfig) { cfg.DecisionRules[set].Question.Labels[1].Key = "billing" },
		"untrimmed label":     func(cfg *RouterConfig) { cfg.DecisionRules[span].Question.Labels[0].Key = " city" },
		"choices on a set":    func(cfg *RouterConfig) { cfg.DecisionRules[set].Question.Choices = []DecisionChoice{{Key: "a"}} },
		"levels on a span":    func(cfg *RouterConfig) { cfg.DecisionRules[span].Question.Levels = []string{"a", "b"} },
		"threshold above one": func(cfg *RouterConfig) { high := 1.5; cfg.DecisionRules[set].Question.Threshold = &high },
		"unknown head":        func(cfg *RouterConfig) { cfg.DecisionRules[span].Question.Head = "wide" },
		"head on a set":       func(cfg *RouterConfig) { cfg.DecisionRules[set].Question.Head = DecisionSpanHeadRouter },
		"labels on a choice":  func(cfg *RouterConfig) { cfg.DecisionRules[1].Question.Labels = []DecisionChoice{{Key: "x"}} },
		"threshold on a noul": func(cfg *RouterConfig) { low := 0.2; cfg.DecisionRules[0].Question.Threshold = &low },
		"too many labels": func(cfg *RouterConfig) {
			labels := make([]DecisionChoice, MaxDecisionLabels+1)
			for index := range labels {
				labels[index] = DecisionChoice{Key: strings.Repeat("l", index+1)}
			}
			cfg.DecisionRules[set].Question.Labels = labels
		},
		"label answer key": func(cfg *RouterConfig) {
			cfg.DecisionRules = append(cfg.DecisionRules, DecisionSignalRule{Name: "topics.billing", Deployment: "vela", Question: DecisionQuestion{Type: DecisionQuestionNoul, Instructions: "?"}})
		},
	}
	for name, mutate := range cases {
		cfg := labelledDecisionConfig()
		mutate(cfg)
		if err := validateDecisionSignalContracts(cfg); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
	cfg := labelledDecisionConfig()
	cfg.DecisionRules = append(cfg.DecisionRules, DecisionSignalRule{Name: "topics.billing", Deployment: "decider", Question: DecisionQuestion{Type: DecisionQuestionNoul, Instructions: "?"}})
	if err := validateDecisionSignalContracts(cfg); err != nil {
		t.Fatalf("a label answer key only collides within one deployment's call: %v", err)
	}
}

func TestSetAndSpanConditionsNameADeclaredLabel(t *testing.T) {
	cfg := labelledDecisionConfig()
	accept := []RuleNode{
		{Type: SignalTypeDecision, Name: "topics", Label: "shipping"},
		{Type: SignalTypeDecision, Name: "places", Label: "city"},
	}
	for _, node := range accept {
		if err := validateDecisionLeafNode(cfg, "d", &node); err != nil {
			t.Fatalf("%+v: %v", node, err)
		}
	}
	reject := map[string]RuleNode{
		"set without label":      {Type: SignalTypeDecision, Name: "topics"},
		"span without label":     {Type: SignalTypeDecision, Name: "places"},
		"undeclared set label":   {Type: SignalTypeDecision, Name: "topics", Label: "refunds"},
		"undeclared span label":  {Type: SignalTypeDecision, Name: "places", Label: "PERSON"},
		"a choice key on a span": {Type: SignalTypeDecision, Name: "places", Label: "code"},
	}
	for name, node := range reject {
		err := validateDecisionLeafNode(cfg, "d", &node)
		if err == nil || !strings.Contains(err.Error(), "declared label") {
			t.Fatalf("%s: expected a declared-label error, got %v", name, err)
		}
	}
}

func TestDecisionConditionLabels(t *testing.T) {
	cfg := decisionSignalConfig()
	accept := []RuleNode{
		{Type: SignalTypeDecision, Name: "hard"},
		{Type: SignalTypeDecision, Name: "level"},
		{Type: SignalTypeDecision, Name: "kind", Label: "math"},
		{Type: SignalTypeDecision, Name: "hard", OnError: "match"},
	}
	for _, node := range accept {
		if err := validateDecisionLeafNode(cfg, "d", &node); err != nil {
			t.Fatalf("%+v: %v", node, err)
		}
	}
	reject := []RuleNode{
		{Type: SignalTypeDecision, Name: "missing"},
		{Type: SignalTypeDecision, Name: "kind"},
		{Type: SignalTypeDecision, Name: "kind", Label: "chat"},
		{Type: SignalTypeDecision, Name: "hard", Label: "true"},
		{Type: SignalTypeKeyword, Name: "kw", Label: "x"},
	}
	for _, node := range reject {
		if err := validateDecisionLeafNode(cfg, "d", &node); err == nil {
			t.Fatalf("%+v: expected a validation error", node)
		}
	}
}

func TestDecisionSelectorContract(t *testing.T) {
	refs := []ModelRef{{Model: "large"}, {Model: "small"}}
	ok := &AlgorithmConfig{Type: DecisionAlgorithmDecision, Decision: &DecisionSelectionConfig{
		Deployment: "decider", Instructions: "Which model?", Candidates: map[string]string{"large": "Best quality"},
	}}
	if err := validateDecisionSelectorConfig("d", refs, ok); err != nil {
		t.Fatal(err)
	}
	ok.Decision.Deployment = ""
	if err := validateDecisionSelectorConfig("d", refs, ok); err != nil {
		t.Fatalf("a selector without a deployment asks the decision model: %v", err)
	}
	cases := map[string]func(*AlgorithmConfig) []ModelRef{
		"blank deployment":  func(a *AlgorithmConfig) []ModelRef { a.Decision.Deployment = " "; return refs },
		"padded deployment": func(a *AlgorithmConfig) []ModelRef { a.Decision.Deployment = " decider"; return refs },
		"missing block":     func(a *AlgorithmConfig) []ModelRef { a.Decision = nil; return refs },
		"one candidate":     func(a *AlgorithmConfig) []ModelRef { return refs[:1] },
		"duplicate model":   func(a *AlgorithmConfig) []ModelRef { return []ModelRef{{Model: "x"}, {Model: "x"}} },
		"unknown candidate": func(a *AlgorithmConfig) []ModelRef {
			a.Decision.Candidates = map[string]string{"other": "x"}
			return refs
		},
		"empty instructions": func(a *AlgorithmConfig) []ModelRef { a.Decision.Instructions = ""; return refs },
	}
	for name, mutate := range cases {
		algorithm := &AlgorithmConfig{Type: DecisionAlgorithmDecision, Decision: &DecisionSelectionConfig{Deployment: "decider", Instructions: "Which?"}}
		candidates := mutate(algorithm)
		if err := validateDecisionSelectorConfig("d", candidates, algorithm); err == nil {
			t.Fatalf("%s: expected a validation error", name)
		}
	}
}

func TestModelRuntimeDeploymentsInUse(t *testing.T) {
	cfg := decisionSignalConfig()
	cfg.ModelDeployments["selector"] = ModelDeployment{Provider: ModelRuntimeProvider, Endpoint: "http://selector:8100"}
	cfg.ModelDeployments["idle"] = ModelDeployment{Provider: ModelRuntimeProvider, Artifact: "vllm-sr/idle"}
	cfg.Decisions = []Decision{{Name: "route", Algorithm: &AlgorithmConfig{Type: "decision", Decision: &DecisionSelectionConfig{Deployment: "selector"}}}}
	used := ModelRuntimeDeploymentsInUse(cfg)
	if len(used) != 2 || used["decider"].Profile != "exact" || used["selector"].Endpoint == "" {
		t.Fatalf("in use = %+v", used)
	}
	if _, idle := used["idle"]; idle {
		t.Fatal("unreferenced deployments must not start")
	}
}

func TestNoulDefaultPredicate(t *testing.T) {
	rule := DecisionSignalRule{Question: DecisionQuestion{Type: DecisionQuestionNoul}}
	predicate := rule.EffectivePredicate()
	if predicate == nil || predicate.GTE == nil || *predicate.GTE != DefaultDecisionNoulThreshold {
		t.Fatalf("noul default predicate = %+v", predicate)
	}
	if (DecisionSignalRule{Question: DecisionQuestion{Type: DecisionQuestionChoice}}).EffectivePredicate() != nil {
		t.Fatal("a choice rule has no default predicate")
	}
	if rule.EffectiveTimeout().Milliseconds() != DefaultDecisionTimeoutMs {
		t.Fatal("default timeout")
	}
}
