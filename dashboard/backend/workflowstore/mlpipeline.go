package workflowstore

import (
	"database/sql"
	"encoding/json"
	"fmt"
	"time"
)

// MLJobRecord mirrors dashboard ML pipeline job JSON for persistence.
type MLJobRecord struct {
	ID          string    `json:"id"`
	Type        string    `json:"type"`
	Status      string    `json:"status"`
	CreatedAt   time.Time `json:"created_at"`
	CompletedAt time.Time `json:"completed_at,omitempty"`
	Error       string    `json:"error,omitempty"`
	OutputFiles []string  `json:"output_files,omitempty"`
	Progress    int       `json:"progress"`
	CurrentStep string    `json:"current_step"`
}

// MLProgressEvent is a typed, durable progress record (not log-derived).
type MLProgressEvent struct {
	ID         int64     `json:"id"`
	JobID      string    `json:"job_id"`
	Step       string    `json:"step"`
	Percent    int       `json:"percent"`
	Message    string    `json:"message"`
	RecordedAt time.Time `json:"recorded_at"`
}

// PutMLJob inserts or replaces a job row.
func (s *Store) PutMLJob(j MLJobRecord) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	outFiles, _ := json.Marshal(j.OutputFiles)
	completed := sqlNullTime(j.CompletedAt)
	_, err := s.db.Exec(`
INSERT INTO ml_pipeline_jobs (id, job_type, status, created_at, completed_at, error, output_files_json, progress, current_step)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(id) DO UPDATE SET
  job_type = excluded.job_type,
  status = excluded.status,
  completed_at = excluded.completed_at,
  error = excluded.error,
  output_files_json = excluded.output_files_json,
  progress = excluded.progress,
  current_step = excluded.current_step`,
		j.ID, j.Type, j.Status, j.CreatedAt.UTC().Format(time.RFC3339Nano),
		completed, nullStr(j.Error), string(outFiles), j.Progress, j.CurrentStep)
	return err
}

// AppendMLProgressEvent records a progress update for a job.
func (s *Store) AppendMLProgressEvent(jobID, step string, percent int, message string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		`INSERT INTO ml_pipeline_progress_events (job_id, step, percent, message, recorded_at) VALUES (?, ?, ?, ?, ?)`,
		jobID, step, percent, message, time.Now().UTC().Format(time.RFC3339Nano))
	return err
}

// UpdateMLJobProgress updates percent and current_step (and ensures row exists).
func (s *Store) UpdateMLJobProgress(jobID string, percent int, currentStep string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	_, err := s.db.Exec(
		`UPDATE ml_pipeline_jobs SET progress = ?, current_step = ? WHERE id = ?`,
		percent, currentStep, jobID)
	return err
}

// GetMLJob loads one job by id.
func (s *Store) GetMLJob(id string) (*MLJobRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	row := s.db.QueryRow(`
SELECT id, job_type, status, created_at, completed_at, error, output_files_json, progress, current_step
FROM ml_pipeline_jobs WHERE id = ?`, id)
	return scanMLJob(row)
}

// ListMLJobs returns jobs newest first.
func (s *Store) ListMLJobs() ([]MLJobRecord, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`
SELECT id, job_type, status, created_at, completed_at, error, output_files_json, progress, current_step
FROM ml_pipeline_jobs ORDER BY created_at DESC`)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []MLJobRecord
	for rows.Next() {
		j, err := scanMLJobRows(rows)
		if err != nil {
			return nil, err
		}
		out = append(out, *j)
	}
	return out, rows.Err()
}

// ListMLProgressEvents returns recent events for a job (oldest first within limit).
func (s *Store) ListMLProgressEvents(jobID string, limit int) ([]MLProgressEvent, error) {
	if limit <= 0 {
		limit = 500
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	rows, err := s.db.Query(`
SELECT id, job_id, step, percent, message, recorded_at
FROM ml_pipeline_progress_events
WHERE job_id = ?
ORDER BY id ASC
LIMIT ?`, jobID, limit)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var out []MLProgressEvent
	for rows.Next() {
		var e MLProgressEvent
		var recorded string
		if err := rows.Scan(&e.ID, &e.JobID, &e.Step, &e.Percent, &e.Message, &recorded); err != nil {
			return nil, err
		}
		t, err := time.Parse(time.RFC3339Nano, recorded)
		if err != nil {
			t, _ = time.Parse(time.RFC3339, recorded)
		}
		e.RecordedAt = t
		out = append(out, e)
	}
	return out, rows.Err()
}

// RecoverInterruptedMLJobs marks running jobs as failed after process restart.
func (s *Store) RecoverInterruptedMLJobs(message string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC().Format(time.RFC3339Nano)
	_, err := s.db.Exec(`
UPDATE ml_pipeline_jobs
SET status = 'failed', completed_at = ?, error = ?
WHERE status = 'running'`, now, message)
	return err
}

// MLWorkflowStats returns counts for health endpoints.
func (s *Store) MLWorkflowStats() (total int, running int, err error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	err = s.db.QueryRow(`SELECT COUNT(*) FROM ml_pipeline_jobs`).Scan(&total)
	if err != nil {
		return 0, 0, err
	}
	err = s.db.QueryRow(`SELECT COUNT(*) FROM ml_pipeline_jobs WHERE status = 'running'`).Scan(&running)
	if err != nil {
		return total, 0, err
	}
	return total, running, nil
}

func scanMLJob(row *sql.Row) (*MLJobRecord, error) {
	var j MLJobRecord
	var created, completed, outJSON, errStr sql.NullString
	err := row.Scan(&j.ID, &j.Type, &j.Status, &created, &completed, &errStr, &outJSON, &j.Progress, &j.CurrentStep)
	if err != nil {
		if err == sql.ErrNoRows {
			return nil, nil
		}
		return nil, err
	}
	j.CreatedAt, _ = parseRFC3339Any(created.String)
	if completed.Valid && completed.String != "" {
		j.CompletedAt, _ = parseRFC3339Any(completed.String)
	}
	if errStr.Valid {
		j.Error = errStr.String
	}
	if outJSON.Valid && outJSON.String != "" {
		_ = json.Unmarshal([]byte(outJSON.String), &j.OutputFiles)
	}
	return &j, nil
}

func scanMLJobRows(rows *sql.Rows) (*MLJobRecord, error) {
	var j MLJobRecord
	var created, completed, outJSON, errStr sql.NullString
	err := rows.Scan(&j.ID, &j.Type, &j.Status, &created, &completed, &errStr, &outJSON, &j.Progress, &j.CurrentStep)
	if err != nil {
		return nil, err
	}
	j.CreatedAt, _ = parseRFC3339Any(created.String)
	if completed.Valid && completed.String != "" {
		j.CompletedAt, _ = parseRFC3339Any(completed.String)
	}
	if errStr.Valid {
		j.Error = errStr.String
	}
	if outJSON.Valid && outJSON.String != "" {
		_ = json.Unmarshal([]byte(outJSON.String), &j.OutputFiles)
	}
	return &j, nil
}

func parseRFC3339Any(s string) (time.Time, error) {
	if s == "" {
		return time.Time{}, fmt.Errorf("empty")
	}
	t, err := time.Parse(time.RFC3339Nano, s)
	if err != nil {
		return time.Parse(time.RFC3339, s)
	}
	return t, nil
}

func sqlNullTime(t time.Time) interface{} {
	if t.IsZero() {
		return nil
	}
	return t.UTC().Format(time.RFC3339Nano)
}

func nullStr(s string) interface{} {
	if s == "" {
		return nil
	}
	return s
}
