package authoring

// CurrentVersion is the current version marker for the first canonical
// authoring config slice introduced by TD001.
const CurrentVersion = "v0.1"

// Config is the first versioned authoring contract for the TD001 migration.
// It intentionally covers only the initial vertical slice rather than the
// entire runtime RouterConfig surface.
type Config struct {
	Version   string     `yaml:"version"`
	Listeners []Listener `yaml:"listeners,omitempty"`
	Signals   Signals    `yaml:"signals,omitempty"`
	Decisions []Decision `yaml:"decisions,omitempty"`
	Providers Providers  `yaml:"providers"`
}

type Listener struct {
	Name    string `yaml:"name"`
	Address string `yaml:"address"`
	Port    int    `yaml:"port"`
	Timeout string `yaml:"timeout,omitempty"`
}

type Signals struct {
	Keywords []KeywordSignal `yaml:"keywords,omitempty"`
}

type KeywordSignal struct {
	Name          string   `yaml:"name"`
	Operator      string   `yaml:"operator"`
	Keywords      []string `yaml:"keywords"`
	CaseSensitive bool     `yaml:"case_sensitive,omitempty"`
}

type Providers struct {
	Models                 []Model                    `yaml:"models,omitempty"`
	DefaultModel           string                     `yaml:"default_model,omitempty"`
	ReasoningFamilies      map[string]ReasoningFamily `yaml:"reasoning_families,omitempty"`
	DefaultReasoningEffort string                     `yaml:"default_reasoning_effort,omitempty"`
}

type Model struct {
	Name            string        `yaml:"name"`
	Endpoints       []Endpoint    `yaml:"endpoints,omitempty"`
	AccessKey       string        `yaml:"access_key,omitempty"`
	ReasoningFamily string        `yaml:"reasoning_family,omitempty"`
	Pricing         *ModelPricing `yaml:"pricing,omitempty"`
	ParamSize       string        `yaml:"param_size,omitempty"`
	APIFormat       string        `yaml:"api_format,omitempty"`
	Description     string        `yaml:"description,omitempty"`
	Capabilities    []string      `yaml:"capabilities,omitempty"`
	QualityScore    float64       `yaml:"quality_score,omitempty"`
}

type Endpoint struct {
	Name     string `yaml:"name"`
	Weight   int    `yaml:"weight,omitempty"`
	Endpoint string `yaml:"endpoint"`
	Protocol string `yaml:"protocol,omitempty"`
}

type ModelPricing struct {
	Currency        string  `yaml:"currency,omitempty"`
	PromptPer1M     float64 `yaml:"prompt_per_1m,omitempty"`
	CompletionPer1M float64 `yaml:"completion_per_1m,omitempty"`
}

type ReasoningFamily struct {
	Type      string `yaml:"type"`
	Parameter string `yaml:"parameter"`
}

type Decision struct {
	Name        string     `yaml:"name"`
	Description string     `yaml:"description,omitempty"`
	Priority    int        `yaml:"priority,omitempty"`
	Rules       Rules      `yaml:"rules"`
	ModelRefs   []ModelRef `yaml:"modelRefs,omitempty"`
}

type Rules struct {
	Operator   string      `yaml:"operator,omitempty"`
	Conditions []Condition `yaml:"conditions,omitempty"`
}

type Condition struct {
	Type       string      `yaml:"type,omitempty"`
	Name       string      `yaml:"name,omitempty"`
	Operator   string      `yaml:"operator,omitempty"`
	Conditions []Condition `yaml:"conditions,omitempty"`
}

type ModelRef struct {
	Model           string `yaml:"model"`
	UseReasoning    bool   `yaml:"use_reasoning"`
	ReasoningEffort string `yaml:"reasoning_effort,omitempty"`
	LoRAName        string `yaml:"lora_name,omitempty"`
}
