// Package evaluation provides evaluation task management functionality.
package evaluation

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/google/uuid"
	_ "github.com/mattn/go-sqlite3"

	"github.com/vllm-project/semantic-router/dashboard/backend/models"
)

// DB handles SQLite database operations for evaluations.
type DB struct {
	db *sql.DB
	mu sync.RWMutex
}

// NewDB creates a new database connection and initializes the schema.
func NewDB(dbPath string) (*DB, error) {
	// Ensure directory exists
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("failed to create database directory: %w", err)
	}

	db, err := sql.Open("sqlite3", dbPath+"?_journal_mode=WAL&_busy_timeout=5000")
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	// Test connection
	if err := db.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping database: %w", err)
	}

	evalDB := &DB{db: db}
	if err := evalDB.initSchema(); err != nil {
		return nil, fmt.Errorf("failed to initialize schema: %w", err)
	}

	log.Printf("Evaluation database initialized at: %s", dbPath)
	return evalDB, nil
}

// initSchema creates the database tables if they don't exist.
func (d *DB) initSchema() error {
	schema := `
	-- Tasks table
	CREATE TABLE IF NOT EXISTS evaluation_tasks (
		id TEXT PRIMARY KEY,
		name TEXT NOT NULL,
		description TEXT,
		status TEXT NOT NULL DEFAULT 'pending',
		created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		started_at TIMESTAMP,
		completed_at TIMESTAMP,
		config_json TEXT NOT NULL,
		error_message TEXT,
		progress_percent INTEGER DEFAULT 0,
		current_step TEXT
	);

	-- Create index on status for filtering
	CREATE INDEX IF NOT EXISTS idx_tasks_status ON evaluation_tasks(status);
	CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON evaluation_tasks(created_at DESC);

	-- Results table
	CREATE TABLE IF NOT EXISTS evaluation_results (
		id TEXT PRIMARY KEY,
		task_id TEXT NOT NULL,
		dimension TEXT NOT NULL,
		dataset_name TEXT NOT NULL,
		metrics_json TEXT NOT NULL,
		raw_results_path TEXT,
		FOREIGN KEY (task_id) REFERENCES evaluation_tasks(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_results_task_id ON evaluation_results(task_id);

	-- Historical tracking
	CREATE TABLE IF NOT EXISTS evaluation_history (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		result_id TEXT NOT NULL,
		metric_name TEXT NOT NULL,
		metric_value REAL NOT NULL,
		recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
		FOREIGN KEY (result_id) REFERENCES evaluation_results(id) ON DELETE CASCADE
	);

	CREATE INDEX IF NOT EXISTS idx_history_result_id ON evaluation_history(result_id);
	CREATE INDEX IF NOT EXISTS idx_history_recorded_at ON evaluation_history(recorded_at DESC);
	`

	_, err := d.db.Exec(schema)
	return err
}

// Close closes the database connection.
func (d *DB) Close() error {
	return d.db.Close()
}

// CreateTask creates a new evaluation task.
func (d *DB) CreateTask(task *models.EvaluationTask) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if task.ID == "" {
		task.ID = uuid.New().String()
	}
	task.CreatedAt = time.Now()
	task.Status = models.StatusPending

	configJSON, err := json.Marshal(task.Config)
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	query := `
		INSERT INTO evaluation_tasks (id, name, description, status, created_at, config_json, progress_percent, current_step)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?)
	`

	_, err = d.db.Exec(query, task.ID, task.Name, task.Description, task.Status, task.CreatedAt, string(configJSON), 0, "")
	if err != nil {
		return fmt.Errorf("failed to insert task: %w", err)
	}

	return nil
}

// GetTask retrieves a task by ID.
func (d *DB) GetTask(id string) (*models.EvaluationTask, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	query := `
		SELECT id, name, description, status, created_at, started_at, completed_at, config_json, error_message, progress_percent, current_step
		FROM evaluation_tasks
		WHERE id = ?
	`

	var task models.EvaluationTask
	var configJSON string
	var startedAt, completedAt sql.NullTime
	var errorMessage, currentStep sql.NullString

	err := d.db.QueryRow(query, id).Scan(
		&task.ID, &task.Name, &task.Description, &task.Status, &task.CreatedAt,
		&startedAt, &completedAt, &configJSON, &errorMessage, &task.ProgressPercent, &currentStep,
	)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("failed to query task: %w", err)
	}

	if startedAt.Valid {
		task.StartedAt = &startedAt.Time
	}
	if completedAt.Valid {
		task.CompletedAt = &completedAt.Time
	}
	if errorMessage.Valid {
		task.ErrorMessage = errorMessage.String
	}
	if currentStep.Valid {
		task.CurrentStep = currentStep.String
	}

	if err := json.Unmarshal([]byte(configJSON), &task.Config); err != nil {
		return nil, fmt.Errorf("failed to unmarshal config: %w", err)
	}

	return &task, nil
}

// ListTasks retrieves all tasks, optionally filtered by status.
func (d *DB) ListTasks(status string) ([]*models.EvaluationTask, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	var query string
	var args []interface{}

	if status != "" {
		query = `
			SELECT id, name, description, status, created_at, started_at, completed_at, config_json, error_message, progress_percent, current_step
			FROM evaluation_tasks
			WHERE status = ?
			ORDER BY created_at DESC
		`
		args = append(args, status)
	} else {
		query = `
			SELECT id, name, description, status, created_at, started_at, completed_at, config_json, error_message, progress_percent, current_step
			FROM evaluation_tasks
			ORDER BY created_at DESC
		`
	}

	rows, err := d.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("failed to query tasks: %w", err)
	}
	defer rows.Close()

	var tasks []*models.EvaluationTask
	for rows.Next() {
		var task models.EvaluationTask
		var configJSON string
		var startedAt, completedAt sql.NullTime
		var errorMessage, currentStep sql.NullString

		err := rows.Scan(
			&task.ID, &task.Name, &task.Description, &task.Status, &task.CreatedAt,
			&startedAt, &completedAt, &configJSON, &errorMessage, &task.ProgressPercent, &currentStep,
		)
		if err != nil {
			return nil, fmt.Errorf("failed to scan task: %w", err)
		}

		if startedAt.Valid {
			task.StartedAt = &startedAt.Time
		}
		if completedAt.Valid {
			task.CompletedAt = &completedAt.Time
		}
		if errorMessage.Valid {
			task.ErrorMessage = errorMessage.String
		}
		if currentStep.Valid {
			task.CurrentStep = currentStep.String
		}

		if err := json.Unmarshal([]byte(configJSON), &task.Config); err != nil {
			return nil, fmt.Errorf("failed to unmarshal config: %w", err)
		}

		tasks = append(tasks, &task)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating tasks: %w", err)
	}

	return tasks, nil
}

// UpdateTaskStatus updates the status and optional error message for a task.
func (d *DB) UpdateTaskStatus(id string, status models.EvaluationStatus, errorMessage string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	var query string
	var args []interface{}

	now := time.Now()

	switch status {
	case models.StatusRunning:
		query = `UPDATE evaluation_tasks SET status = ?, started_at = ?, error_message = NULL WHERE id = ?`
		args = []interface{}{status, now, id}
	case models.StatusCompleted, models.StatusFailed, models.StatusCancelled:
		query = `UPDATE evaluation_tasks SET status = ?, completed_at = ?, error_message = ? WHERE id = ?`
		args = []interface{}{status, now, errorMessage, id}
	default:
		query = `UPDATE evaluation_tasks SET status = ?, error_message = ? WHERE id = ?`
		args = []interface{}{status, errorMessage, id}
	}

	result, err := d.db.Exec(query, args...)
	if err != nil {
		return fmt.Errorf("failed to update task status: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("task not found: %s", id)
	}

	return nil
}

// UpdateTaskProgress updates the progress percentage and current step.
func (d *DB) UpdateTaskProgress(id string, percent int, currentStep string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	query := `UPDATE evaluation_tasks SET progress_percent = ?, current_step = ? WHERE id = ?`

	result, err := d.db.Exec(query, percent, currentStep, id)
	if err != nil {
		return fmt.Errorf("failed to update task progress: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("task not found: %s", id)
	}

	return nil
}

// DeleteTask deletes a task and its associated results.
func (d *DB) DeleteTask(id string) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	// SQLite foreign keys with ON DELETE CASCADE should handle results
	query := `DELETE FROM evaluation_tasks WHERE id = ?`

	result, err := d.db.Exec(query, id)
	if err != nil {
		return fmt.Errorf("failed to delete task: %w", err)
	}

	rowsAffected, err := result.RowsAffected()
	if err != nil {
		return fmt.Errorf("failed to get rows affected: %w", err)
	}
	if rowsAffected == 0 {
		return fmt.Errorf("task not found: %s", id)
	}

	return nil
}

// SaveResult saves an evaluation result.
func (d *DB) SaveResult(result *models.EvaluationResult) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	if result.ID == "" {
		result.ID = uuid.New().String()
	}

	metricsJSON, err := json.Marshal(result.Metrics)
	if err != nil {
		return fmt.Errorf("failed to marshal metrics: %w", err)
	}

	query := `
		INSERT INTO evaluation_results (id, task_id, dimension, dataset_name, metrics_json, raw_results_path)
		VALUES (?, ?, ?, ?, ?, ?)
	`

	_, err = d.db.Exec(query, result.ID, result.TaskID, result.Dimension, result.DatasetName, string(metricsJSON), result.RawResultsPath)
	if err != nil {
		return fmt.Errorf("failed to insert result: %w", err)
	}

	return nil
}

// GetResults retrieves all results for a task.
func (d *DB) GetResults(taskID string) ([]*models.EvaluationResult, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	query := `
		SELECT id, task_id, dimension, dataset_name, metrics_json, raw_results_path
		FROM evaluation_results
		WHERE task_id = ?
	`

	rows, err := d.db.Query(query, taskID)
	if err != nil {
		return nil, fmt.Errorf("failed to query results: %w", err)
	}
	defer rows.Close()

	var results []*models.EvaluationResult
	for rows.Next() {
		var result models.EvaluationResult
		var metricsJSON string
		var rawResultsPath sql.NullString

		err := rows.Scan(&result.ID, &result.TaskID, &result.Dimension, &result.DatasetName, &metricsJSON, &rawResultsPath)
		if err != nil {
			return nil, fmt.Errorf("failed to scan result: %w", err)
		}

		if rawResultsPath.Valid {
			result.RawResultsPath = rawResultsPath.String
		}

		if err := json.Unmarshal([]byte(metricsJSON), &result.Metrics); err != nil {
			return nil, fmt.Errorf("failed to unmarshal metrics: %w", err)
		}

		results = append(results, &result)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating results: %w", err)
	}

	return results, nil
}

// SaveHistoryEntry saves a historical metric entry.
func (d *DB) SaveHistoryEntry(entry *models.EvaluationHistoryEntry) error {
	d.mu.Lock()
	defer d.mu.Unlock()

	query := `
		INSERT INTO evaluation_history (result_id, metric_name, metric_value, recorded_at)
		VALUES (?, ?, ?, ?)
	`

	result, err := d.db.Exec(query, entry.ResultID, entry.MetricName, entry.MetricValue, time.Now())
	if err != nil {
		return fmt.Errorf("failed to insert history entry: %w", err)
	}

	id, err := result.LastInsertId()
	if err != nil {
		return fmt.Errorf("failed to get last insert id: %w", err)
	}
	entry.ID = id

	return nil
}

// GetHistoryForMetric retrieves historical values for a specific metric across results.
func (d *DB) GetHistoryForMetric(metricName string, limit int) ([]*models.EvaluationHistoryEntry, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	query := `
		SELECT id, result_id, metric_name, metric_value, recorded_at
		FROM evaluation_history
		WHERE metric_name = ?
		ORDER BY recorded_at DESC
		LIMIT ?
	`

	rows, err := d.db.Query(query, metricName, limit)
	if err != nil {
		return nil, fmt.Errorf("failed to query history: %w", err)
	}
	defer rows.Close()

	var entries []*models.EvaluationHistoryEntry
	for rows.Next() {
		var entry models.EvaluationHistoryEntry
		err := rows.Scan(&entry.ID, &entry.ResultID, &entry.MetricName, &entry.MetricValue, &entry.RecordedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan history entry: %w", err)
		}
		entries = append(entries, &entry)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating history: %w", err)
	}

	return entries, nil
}

// GetHistoryForResult retrieves all historical entries for a specific result.
func (d *DB) GetHistoryForResult(resultID string) ([]*models.EvaluationHistoryEntry, error) {
	d.mu.RLock()
	defer d.mu.RUnlock()

	query := `
		SELECT id, result_id, metric_name, metric_value, recorded_at
		FROM evaluation_history
		WHERE result_id = ?
		ORDER BY recorded_at DESC
	`

	rows, err := d.db.Query(query, resultID)
	if err != nil {
		return nil, fmt.Errorf("failed to query history: %w", err)
	}
	defer rows.Close()

	var entries []*models.EvaluationHistoryEntry
	for rows.Next() {
		var entry models.EvaluationHistoryEntry
		err := rows.Scan(&entry.ID, &entry.ResultID, &entry.MetricName, &entry.MetricValue, &entry.RecordedAt)
		if err != nil {
			return nil, fmt.Errorf("failed to scan history entry: %w", err)
		}
		entries = append(entries, &entry)
	}

	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("error iterating history: %w", err)
	}

	return entries, nil
}
