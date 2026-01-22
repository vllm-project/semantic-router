// Package models defines data structures for the evaluation system.
package models

import (
	"time"
)

// EvaluationStatus represents the current state of an evaluation task.
type EvaluationStatus string

const (
	StatusPending   EvaluationStatus = "pending"
	StatusRunning   EvaluationStatus = "running"
	StatusCompleted EvaluationStatus = "completed"
	StatusFailed    EvaluationStatus = "failed"
	StatusCancelled EvaluationStatus = "cancelled"
)

// EvaluationDimension represents a type of evaluation to run.
type EvaluationDimension string

const (
	DimensionHallucination EvaluationDimension = "hallucination"
	DimensionReasoning     EvaluationDimension = "reasoning"
	DimensionAccuracy      EvaluationDimension = "accuracy"
	DimensionLatency       EvaluationDimension = "latency"
	DimensionCost          EvaluationDimension = "cost"
	DimensionSecurity      EvaluationDimension = "security"
)

// EvaluationConfig holds the configuration for an evaluation task.
type EvaluationConfig struct {
	Dimensions    []EvaluationDimension `json:"dimensions"`
	Datasets      map[string][]string   `json:"datasets"`        // dimension -> dataset names
	MaxSamples    int                   `json:"max_samples"`     // max samples per dataset
	Endpoint      string                `json:"endpoint"`        // router endpoint URL
	Model         string                `json:"model"`           // model to evaluate (empty for auto-discovery)
	Concurrent    int                   `json:"concurrent"`      // concurrent requests
	SamplesPerCat int                   `json:"samples_per_cat"` // samples per category (for reasoning)
}

// EvaluationTask represents an evaluation task stored in the database.
type EvaluationTask struct {
	ID              string           `json:"id"`
	Name            string           `json:"name"`
	Description     string           `json:"description"`
	Status          EvaluationStatus `json:"status"`
	CreatedAt       time.Time        `json:"created_at"`
	StartedAt       *time.Time       `json:"started_at,omitempty"`
	CompletedAt     *time.Time       `json:"completed_at,omitempty"`
	Config          EvaluationConfig `json:"config"`
	ErrorMessage    string           `json:"error_message,omitempty"`
	ProgressPercent int              `json:"progress_percent"`
	CurrentStep     string           `json:"current_step,omitempty"`
}

// EvaluationResult holds the results for a specific dimension/dataset combination.
type EvaluationResult struct {
	ID             string                 `json:"id"`
	TaskID         string                 `json:"task_id"`
	Dimension      EvaluationDimension    `json:"dimension"`
	DatasetName    string                 `json:"dataset_name"`
	Metrics        map[string]interface{} `json:"metrics"`
	RawResultsPath string                 `json:"raw_results_path,omitempty"`
}

// EvaluationHistoryEntry tracks metric values over time.
type EvaluationHistoryEntry struct {
	ID          int64     `json:"id"`
	ResultID    string    `json:"result_id"`
	MetricName  string    `json:"metric_name"`
	MetricValue float64   `json:"metric_value"`
	RecordedAt  time.Time `json:"recorded_at"`
}

// HallucinationMetrics holds parsed metrics from the hallucination benchmark.
type HallucinationMetrics struct {
	DatasetName                 string  `json:"dataset_name"`
	Endpoint                    string  `json:"endpoint"`
	Model                       string  `json:"model"`
	TotalSamples                int     `json:"total_samples"`
	SuccessfulRequests          int     `json:"successful_requests"`
	FailedRequests              int     `json:"failed_requests"`
	TotalHallucinationsDetected int     `json:"total_hallucinations_detected"`
	TruePositives               int     `json:"true_positives"`
	FalsePositives              int     `json:"false_positives"`
	TrueNegatives               int     `json:"true_negatives"`
	FalseNegatives              int     `json:"false_negatives"`
	Precision                   float64 `json:"precision"`
	Recall                      float64 `json:"recall"`
	F1Score                     float64 `json:"f1_score"`
	Accuracy                    float64 `json:"accuracy"`
	AvgLatencyMs                float64 `json:"avg_latency_ms"`
	P50LatencyMs                float64 `json:"p50_latency_ms"`
	P99LatencyMs                float64 `json:"p99_latency_ms"`
	FactCheckNeededCount        int     `json:"fact_check_needed_count"`
	DetectionSkippedCount       int     `json:"detection_skipped_count"`
	AvgContextLength            float64 `json:"avg_context_length"`
	EstimatedDetectionTimeMs    float64 `json:"estimated_detection_time_ms"`
	ActualDetectionTimeMs       float64 `json:"actual_detection_time_ms"`
	TimeSavedMs                 float64 `json:"time_saved_ms"`
	EfficiencyGainPercent       float64 `json:"efficiency_gain_percent"`
}

// ReasoningModeMetrics holds metrics for a single mode (standard or reasoning).
type ReasoningModeMetrics struct {
	ModeName             string  `json:"mode_name"`
	TotalQuestions       int     `json:"total_questions"`
	CorrectAnswers       int     `json:"correct_answers"`
	FailedQueries        int     `json:"failed_queries"`
	Accuracy             float64 `json:"accuracy"`
	AvgResponseTimeSec   float64 `json:"avg_response_time_sec"`
	AvgPromptTokens      float64 `json:"avg_prompt_tokens"`
	AvgCompletionTokens  float64 `json:"avg_completion_tokens"`
	AvgTotalTokens       float64 `json:"avg_total_tokens"`
	TokenUsageRatio      float64 `json:"token_usage_ratio"`
	TimePerOutputTokenMs float64 `json:"time_per_output_token_ms"`
}

// ReasoningImprovementSummary holds the comparison between modes.
type ReasoningImprovementSummary struct {
	AccuracyDelta             float64 `json:"accuracy_delta"`
	AccuracyImprovementPct    float64 `json:"accuracy_improvement_pct"`
	TokenUsageRatioDelta      float64 `json:"token_usage_ratio_delta"`
	TimePerOutputTokenDeltaMs float64 `json:"time_per_output_token_delta_ms"`
	ResponseTimeDeltaSec      float64 `json:"response_time_delta_sec"`
}

// ReasoningComparison holds the comparison results for a dataset.
type ReasoningComparison struct {
	Dataset            string                                     `json:"dataset"`
	Model              string                                     `json:"model"`
	Timestamp          string                                     `json:"timestamp"`
	StandardMode       ReasoningModeMetrics                       `json:"standard_mode"`
	ReasoningMode      ReasoningModeMetrics                       `json:"reasoning_mode"`
	ImprovementSummary ReasoningImprovementSummary                `json:"improvement_summary"`
	CategoryBreakdown  map[string]map[string]ReasoningModeMetrics `json:"category_breakdown,omitempty"`
}

// ReasoningMetrics holds the full output from the reasoning mode benchmark.
type ReasoningMetrics struct {
	GeneratedAt             string                `json:"generated_at"`
	Issue                   string                `json:"issue"`
	Title                   string                `json:"title"`
	Comparisons             []ReasoningComparison `json:"comparisons"`
	VSRConfigRecommendation interface{}           `json:"vsr_config_recommendation,omitempty"`
}

// DatasetInfo provides information about an available dataset.
type DatasetInfo struct {
	Name        string              `json:"name"`
	Description string              `json:"description"`
	Dimension   EvaluationDimension `json:"dimension"`
	SampleCount int                 `json:"sample_count,omitempty"`
}

// CreateTaskRequest is the request body for creating a new task.
type CreateTaskRequest struct {
	Name        string           `json:"name"`
	Description string           `json:"description"`
	Config      EvaluationConfig `json:"config"`
}

// RunTaskRequest is the request body for running a task.
type RunTaskRequest struct {
	TaskID string `json:"task_id"`
}

// ProgressUpdate represents a real-time progress update for SSE.
type ProgressUpdate struct {
	TaskID          string `json:"task_id"`
	ProgressPercent int    `json:"progress_percent"`
	CurrentStep     string `json:"current_step"`
	Message         string `json:"message,omitempty"`
	Timestamp       int64  `json:"timestamp"`
}

// ExportFormat specifies the output format for reports.
type ExportFormat string

const (
	ExportJSON ExportFormat = "json"
	ExportCSV  ExportFormat = "csv"
	ExportPDF  ExportFormat = "pdf"
)
