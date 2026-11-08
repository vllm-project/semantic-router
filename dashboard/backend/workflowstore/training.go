package workflowstore

import (
	"context"
	"database/sql"
	"encoding/json"

	c "github.com/vllm-project/semantic-router/src/semantic-router/pkg/trainingcontract"
)

// TrainingRecord keeps ownership private to management. Its public body is a v2 resource.
type TrainingRecord struct {
	Kind, ID, Owner string
	Body            json.RawMessage
}
type TrainingSubmission struct{ Owner, Key, Digest, RunID string }

func (s *Store) initTrainingSchema() error {
	_, err := s.db.Exec(`
 CREATE TABLE IF NOT EXISTS training_resources (
  kind TEXT NOT NULL, id TEXT PRIMARY KEY, owner TEXT NOT NULL, body BLOB NOT NULL
 );
 CREATE INDEX IF NOT EXISTS training_resources_owner ON training_resources(owner,kind);
 CREATE TABLE IF NOT EXISTS training_submissions (
  owner TEXT NOT NULL, key TEXT NOT NULL, digest TEXT NOT NULL, run_id TEXT NOT NULL,
  PRIMARY KEY(owner,key), FOREIGN KEY(run_id) REFERENCES training_resources(id)
 );
 CREATE TABLE IF NOT EXISTS training_events (
  sequence INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT NOT NULL, body BLOB NOT NULL,
  FOREIGN KEY(run_id) REFERENCES training_resources(id)
 );
 CREATE INDEX IF NOT EXISTS training_events_run ON training_events(run_id,sequence);`)
	return err
}

// TrainingTx serializes read-modify-write operations, including across Store instances.
// The connection uses BEGIN IMMEDIATE; no worker or file I/O runs inside this transaction.
type TrainingTx struct {
	tx  *sql.Tx
	ctx context.Context
}

func (s *Store) UpdateTraining(ctx context.Context, fn func(*TrainingTx) error) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return err
	}
	defer func() { _ = tx.Rollback() }()
	if err := fn(&TrainingTx{tx: tx, ctx: ctx}); err != nil {
		return err
	}
	return tx.Commit()
}

func (t *TrainingTx) Get(kind, id, owner string) (TrainingRecord, error) {
	var r TrainingRecord
	err := t.tx.QueryRowContext(t.ctx, `SELECT kind,id,owner,body FROM training_resources WHERE kind=? AND id=? AND owner=?`, kind, id, owner).Scan(&r.Kind, &r.ID, &r.Owner, &r.Body)
	return r, err
}

func (s *Store) GetTraining(ctx context.Context, kind, id, owner string) (TrainingRecord, error) {
	var r TrainingRecord
	err := s.db.QueryRowContext(ctx, `SELECT kind,id,owner,body FROM training_resources WHERE kind=? AND id=? AND owner=?`, kind, id, owner).Scan(&r.Kind, &r.ID, &r.Owner, &r.Body)
	return r, err
}

func (t *TrainingTx) Insert(r TrainingRecord) error {
	_, err := t.tx.ExecContext(t.ctx, `INSERT INTO training_resources(kind,id,owner,body) VALUES(?,?,?,?)`, r.Kind, r.ID, r.Owner, []byte(r.Body))
	return err
}

func (t *TrainingTx) SaveRun(r TrainingRecord) error {
	_, err := t.tx.ExecContext(t.ctx, `UPDATE training_resources SET body=? WHERE kind='runs' AND id=? AND owner=?`, []byte(r.Body), r.ID, r.Owner)
	return err
}

func (t *TrainingTx) FindSubmission(owner, key string) (*TrainingSubmission, error) {
	r := TrainingSubmission{Owner: owner, Key: key}
	err := t.tx.QueryRowContext(t.ctx, `SELECT digest,run_id FROM training_submissions WHERE owner=? AND key=?`, owner, key).Scan(&r.Digest, &r.RunID)
	if err == sql.ErrNoRows {
		return nil, nil
	}
	return &r, err
}

func (t *TrainingTx) Submit(r TrainingSubmission) error {
	_, err := t.tx.ExecContext(t.ctx, `INSERT INTO training_submissions(owner,key,digest,run_id) VALUES(?,?,?,?)`, r.Owner, r.Key, r.Digest, r.RunID)
	return err
}

func (t *TrainingTx) Event(e c.Event) error {
	body, err := json.Marshal(e)
	if err != nil {
		return err
	}
	_, err = t.tx.ExecContext(t.ctx, `INSERT INTO training_events(run_id,body) VALUES(?,?)`, e.RunID, body)
	return err
}

// ListTrainingAll is for the management recovery coordinator, never a public owner override.
func (s *Store) ListTrainingAll(ctx context.Context, kind string) ([]TrainingRecord, error) {
	return s.listTraining(ctx, `SELECT kind,id,owner,body FROM training_resources WHERE kind=? ORDER BY rowid`, kind)
}

func (s *Store) ListTraining(ctx context.Context, kind, owner string) ([]TrainingRecord, error) {
	return s.listTraining(ctx, `SELECT kind,id,owner,body FROM training_resources WHERE kind=? AND owner=? ORDER BY rowid`, kind, owner)
}

func (s *Store) listTraining(ctx context.Context, query string, args ...any) ([]TrainingRecord, error) {
	rows, err := s.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer func() { _ = rows.Close() }()
	out := []TrainingRecord{}
	for rows.Next() {
		var r TrainingRecord
		if err := rows.Scan(&r.Kind, &r.ID, &r.Owner, &r.Body); err != nil {
			return nil, err
		}
		out = append(out, r)
	}
	return out, rows.Err()
}

func (s *Store) TrainingEvents(ctx context.Context, runID string, after int64) (c.EventPage, error) {
	page := c.EventPage{Events: []c.Event{}, NextAfter: after}
	rows, err := s.db.QueryContext(ctx, `SELECT sequence,body FROM training_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT 1000`, runID, after)
	if err != nil {
		return page, err
	}
	defer func() { _ = rows.Close() }()
	for rows.Next() {
		var seq int64
		var body []byte
		if err := rows.Scan(&seq, &body); err != nil {
			return page, err
		}
		var e c.Event
		if err := json.Unmarshal(body, &e); err != nil {
			return page, err
		}
		e.Sequence = seq
		page.Events = append(page.Events, e)
		page.NextAfter = seq
	}
	return page, rows.Err()
}
