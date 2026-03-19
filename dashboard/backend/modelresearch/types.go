package modelresearch

import (
	"time"

	modelinventory "github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelinventory"
)

type CampaignStatus string

const (
	StatusPending   CampaignStatus = "pending"
	StatusRunning   CampaignStatus = "running"
	StatusCompleted CampaignStatus = "completed"
	StatusFailed    CampaignStatus = "failed"
	StatusStopped   CampaignStatus = "stopped"
	StatusBlocked   CampaignStatus = "blocked"
)

type GoalTemplate string

const (
	GoalImproveAccuracy GoalTemplate = "improve_accuracy"
	GoalExploreSignal   GoalTemplate = "explore_signal"
)

type EventKind string

const (
	EventStatus   EventKind = "status"
	EventLog      EventKind = "log"
	EventProgress EventKind = "progress"
	EventMetric   EventKind = "metric"
)

type Budget struct {
	MaxTrials int `json:"max_trials"`
}

type Overrides struct {
	APIBaseOverride      string         `json:"api_base_override,omitempty"`
	RequestModelOverride string         `json:"request_model_override,omitempty"`
	DatasetOverride      string         `json:"dataset_override,omitempty"`
	HyperparameterHints  map[string]any `json:"hyperparameter_hints,omitempty"`
	AllowCPUDryRun       bool           `json:"allow_cpu_dry_run,omitempty"`
}

type CreateCampaignRequest struct {
	Name               string       `json:"name"`
	GoalTemplate       GoalTemplate `json:"goal_template"`
	Target             string       `json:"target"`
	Budget             Budget       `json:"budget"`
	SuccessThresholdPP float64      `json:"success_threshold_pp"`
	Overrides          Overrides    `json:"overrides,omitempty"`
}

type CampaignEvent struct {
	Timestamp  time.Time `json:"timestamp"`
	Kind       EventKind `json:"kind"`
	Level      string    `json:"level,omitempty"`
	Message    string    `json:"message"`
	Percent    int       `json:"percent,omitempty"`
	TrialIndex int       `json:"trial_index,omitempty"`
}

type Baseline struct {
	Label        string   `json:"label"`
	Source       string   `json:"source"`
	RuntimeName  string   `json:"runtime_name,omitempty"`
	ModelPath    string   `json:"model_path,omitempty"`
	ModelID      string   `json:"model_id,omitempty"`
	State        string   `json:"state,omitempty"`
	Description  string   `json:"description,omitempty"`
	Categories   []string `json:"categories,omitempty"`
	RequestModel string   `json:"request_model,omitempty"`
}

type MetricSnapshot struct {
	Source        string  `json:"source"`
	Dataset       string  `json:"dataset"`
	Accuracy      float64 `json:"accuracy"`
	F1            float64 `json:"f1,omitempty"`
	Precision     float64 `json:"precision,omitempty"`
	Recall        float64 `json:"recall,omitempty"`
	LatencyAvgMS  float64 `json:"latency_avg_ms,omitempty"`
	OutputPath    string  `json:"output_path,omitempty"`
	ModelID       string  `json:"model_id,omitempty"`
	ImprovementPP float64 `json:"improvement_pp,omitempty"`
}

type TrialResult struct {
	Index         int               `json:"index"`
	Name          string            `json:"name"`
	Status        CampaignStatus    `json:"status"`
	StartedAt     time.Time         `json:"started_at"`
	CompletedAt   *time.Time        `json:"completed_at,omitempty"`
	Params        map[string]any    `json:"params,omitempty"`
	ModelPath     string            `json:"model_path,omitempty"`
	UseLoRA       bool              `json:"use_lora,omitempty"`
	PrimaryMetric string            `json:"primary_metric"`
	Eval          *MetricSnapshot   `json:"eval,omitempty"`
	RuntimeEval   *MetricSnapshot   `json:"runtime_eval,omitempty"`
	Error         string            `json:"error,omitempty"`
	Artifacts     map[string]string `json:"artifacts,omitempty"`
}

type Campaign struct {
	ID                  string                             `json:"id"`
	Name                string                             `json:"name"`
	Status              CampaignStatus                     `json:"status"`
	GoalTemplate        GoalTemplate                       `json:"goal_template"`
	Target              string                             `json:"target"`
	Platform            string                             `json:"platform"`
	PrimaryMetric       string                             `json:"primary_metric"`
	SuccessThresholdPP  float64                            `json:"success_threshold_pp"`
	Budget              Budget                             `json:"budget"`
	CreatedAt           time.Time                          `json:"created_at"`
	UpdatedAt           time.Time                          `json:"updated_at"`
	CompletedAt         *time.Time                         `json:"completed_at,omitempty"`
	DefaultAPIBase      string                             `json:"default_api_base"`
	APIBase             string                             `json:"api_base"`
	DefaultRequestModel string                             `json:"default_request_model"`
	RequestModel        string                             `json:"request_model"`
	Overrides           Overrides                          `json:"overrides,omitempty"`
	Recipe              RecipeSummary                      `json:"recipe"`
	Baseline            Baseline                           `json:"baseline"`
	BaselineEval        *MetricSnapshot                    `json:"baseline_eval,omitempty"`
	RuntimeBaseline     *MetricSnapshot                    `json:"runtime_baseline,omitempty"`
	BestTrial           *TrialResult                       `json:"best_trial,omitempty"`
	Trials              []TrialResult                      `json:"trials,omitempty"`
	Events              []CampaignEvent                    `json:"events,omitempty"`
	ArtifactDir         string                             `json:"artifact_dir"`
	ConfigFragmentPath  string                             `json:"config_fragment_path,omitempty"`
	LastError           string                             `json:"last_error,omitempty"`
	RuntimeModels       *modelinventory.ModelsInfoResponse `json:"runtime_models,omitempty"`
}

type RecipeSummary struct {
	Key                         string         `json:"key"`
	Label                       string         `json:"label"`
	GoalTemplates               []GoalTemplate `json:"goal_templates"`
	DefaultDataset              string         `json:"default_dataset"`
	DatasetHint                 string         `json:"dataset_hint"`
	DefaultSuccessThresholdPP   float64        `json:"default_success_threshold_pp"`
	PrimaryMetric               string         `json:"primary_metric"`
	SupportsDatasetOverride     bool           `json:"supports_dataset_override"`
	SupportsHyperparameterHints bool           `json:"supports_hyperparameter_hints"`
	Baseline                    Baseline       `json:"baseline"`
}

type RecipesResponse struct {
	DefaultAPIBase      string                             `json:"default_api_base"`
	DefaultRequestModel string                             `json:"default_request_model"`
	DefaultPlatform     string                             `json:"default_platform"`
	RuntimeModels       *modelinventory.ModelsInfoResponse `json:"runtime_models,omitempty"`
	Recipes             []RecipeSummary                    `json:"recipes"`
}
