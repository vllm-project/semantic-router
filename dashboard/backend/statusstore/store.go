// Package statusstore persists the Dashboard's bounded public service history.
package statusstore

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	_ "github.com/mattn/go-sqlite3"
)

const (
	// RetentionHours is the public status window, including the current UTC hour.
	// Ninety points keep the availability track legible while exposing hour-level
	// incidents instead of flattening a full day into one state.
	RetentionHours = 90

	StateUnknown     State = "unknown"
	StateOperational State = "operational"
	StateStarting    State = "starting"
	StateUnavailable State = "unavailable"
)

const schema = `
CREATE TABLE IF NOT EXISTS service_status_hourly (
	service_name TEXT NOT NULL,
	observed_at TEXT NOT NULL,
	status TEXT NOT NULL CHECK (status IN ('operational', 'starting', 'unavailable')),
	severity INTEGER NOT NULL CHECK (severity BETWEEN 0 AND 2),
	PRIMARY KEY (service_name, observed_at),
	CHECK (
		(status = 'operational' AND severity = 0) OR
		(status = 'starting' AND severity = 1) OR
		(status = 'unavailable' AND severity = 2)
	)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS service_status_hourly_observed_at_idx
	ON service_status_hourly(observed_at);
`

// State is one hourly service availability state. Unknown is returned for hours
// without a server-side observation and is never persisted as an observation.
type State string

// Observation is one current, server-observed service state.
type Observation struct {
	Service string
	State   State
}

// Hour is one UTC hour in the public history window.
type Hour struct {
	ObservedAt string `json:"observedAt"`
	Status     State  `json:"status"`
}

// ServiceHistory is a dense hourly series for one public service.
type ServiceHistory struct {
	Name  string `json:"name"`
	Hours []Hour `json:"hours"`
}

// History is the bounded server-owned history returned by /api/status.
type History struct {
	WindowHours int              `json:"windowHours"`
	Through     string           `json:"through"`
	Services    []ServiceHistory `json:"services"`
}

// Store is a concurrency-safe SQLite status history store.
type Store struct {
	db  *sql.DB
	mu  sync.Mutex
	now func() time.Time
}

// Open opens or creates a status history database at dbPath.
func Open(dbPath string) (*Store, error) {
	if strings.TrimSpace(dbPath) == "" {
		return nil, fmt.Errorf("statusstore: database path is required")
	}
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("statusstore: create dir: %w", err)
	}
	db, err := sql.Open(
		"sqlite3",
		dbPath+"?_journal_mode=WAL&_busy_timeout=5000&_foreign_keys=1",
	)
	if err != nil {
		return nil, fmt.Errorf("statusstore: open: %w", err)
	}
	db.SetMaxOpenConns(1)
	if err := db.Ping(); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("statusstore: ping: %w", err)
	}
	if _, err := db.Exec(schema); err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("statusstore: initialize schema: %w", err)
	}
	return &Store{db: db, now: time.Now}, nil
}

// Close releases the SQLite handle.
func (s *Store) Close() error {
	if s == nil || s.db == nil {
		return nil
	}
	return s.db.Close()
}

// Record atomically records current states, keeps each hour's worst state, and
// prunes observations outside the bounded history window.
func (s *Store) Record(ctx context.Context, observations []Observation) error {
	if s == nil || s.db == nil {
		return fmt.Errorf("statusstore: store is not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	currentHour := utcHour(s.now())
	start := currentHour.Add(-time.Duration(RetentionHours-1) * time.Hour)
	normalized, err := normalizeObservations(observations)
	if err != nil {
		return err
	}

	tx, err := s.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("statusstore: begin observation: %w", err)
	}
	defer func() { _ = tx.Rollback() }()

	for _, observation := range normalized {
		severity, _ := stateSeverity(observation.State)
		if _, err := tx.ExecContext(
			ctx,
			`INSERT INTO service_status_hourly(service_name, observed_at, status, severity)
			 VALUES(?, ?, ?, ?)
			 ON CONFLICT(service_name, observed_at) DO UPDATE SET
			 status = CASE
			   WHEN excluded.severity > service_status_hourly.severity THEN excluded.status
			   ELSE service_status_hourly.status
			 END,
			 severity = MAX(service_status_hourly.severity, excluded.severity)`,
			observation.Service,
			currentHour.Format(time.RFC3339),
			observation.State,
			severity,
		); err != nil {
			return fmt.Errorf("statusstore: record observation: %w", err)
		}
	}

	if _, err := tx.ExecContext(
		ctx,
		`DELETE FROM service_status_hourly WHERE observed_at < ? OR observed_at > ?`,
		start.Format(time.RFC3339),
		currentHour.Format(time.RFC3339),
	); err != nil {
		return fmt.Errorf("statusstore: prune history: %w", err)
	}
	if err := tx.Commit(); err != nil {
		return fmt.Errorf("statusstore: commit observation: %w", err)
	}
	return nil
}

// Observe records a state and returns the current dense series. Request
// handlers use Read and never write.
func (s *Store) Observe(ctx context.Context, observations []Observation) (History, error) {
	if err := s.Record(ctx, observations); err != nil {
		return History{}, err
	}
	return s.Read(ctx, nil)
}

// Read returns the current dense history without recording a client-driven
// observation. A lifecycle-owned sampler is responsible for writes.
func (s *Store) Read(ctx context.Context, serviceNames []string) (History, error) {
	if s == nil || s.db == nil {
		return History{}, fmt.Errorf("statusstore: store is not initialized")
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	currentHour := utcHour(s.now())
	start := currentHour.Add(-time.Duration(RetentionHours-1) * time.Hour)
	observed, err := readObservedHours(ctx, s.db, start, currentHour)
	if err != nil {
		return History{}, err
	}
	for _, name := range serviceNames {
		name = strings.TrimSpace(name)
		if name != "" && observed[name] == nil {
			observed[name] = map[string]State{}
		}
	}
	return denseHistory(start, currentHour, observed), nil
}

// UnknownHistory returns an honest all-unknown window for a set of services.
// It is used only when durable history cannot be read; it never invents uptime.
func UnknownHistory(now time.Time, serviceNames []string) History {
	currentHour := utcHour(now)
	start := currentHour.Add(-time.Duration(RetentionHours-1) * time.Hour)
	observed := make(map[string]map[string]State, len(serviceNames))
	for _, name := range serviceNames {
		name = strings.TrimSpace(name)
		if name != "" {
			observed[name] = map[string]State{}
		}
	}
	return denseHistory(start, currentHour, observed)
}

func normalizeObservations(observations []Observation) ([]Observation, error) {
	byService := make(map[string]Observation, len(observations))
	for _, observation := range observations {
		observation.Service = strings.TrimSpace(observation.Service)
		if observation.Service == "" || len(observation.Service) > 128 {
			return nil, fmt.Errorf("statusstore: invalid service name")
		}
		severity, valid := stateSeverity(observation.State)
		if !valid {
			return nil, fmt.Errorf("statusstore: invalid state %q", observation.State)
		}
		if previous, exists := byService[observation.Service]; exists {
			previousSeverity, _ := stateSeverity(previous.State)
			if previousSeverity >= severity {
				continue
			}
		}
		byService[observation.Service] = observation
	}
	normalized := make([]Observation, 0, len(byService))
	for _, observation := range byService {
		normalized = append(normalized, observation)
	}
	sort.Slice(normalized, func(i, j int) bool {
		return normalized[i].Service < normalized[j].Service
	})
	return normalized, nil
}

func stateSeverity(state State) (int, bool) {
	switch state {
	case StateOperational:
		return 0, true
	case StateStarting:
		return 1, true
	case StateUnavailable:
		return 2, true
	default:
		return 0, false
	}
}

func readObservedHours(
	ctx context.Context,
	db *sql.DB,
	start time.Time,
	currentHour time.Time,
) (map[string]map[string]State, error) {
	rows, err := db.QueryContext(
		ctx,
		`SELECT service_name, observed_at, status
		 FROM service_status_hourly
		 WHERE observed_at >= ? AND observed_at <= ?
		 ORDER BY service_name, observed_at`,
		start.Format(time.RFC3339),
		currentHour.Format(time.RFC3339),
	)
	if err != nil {
		return nil, fmt.Errorf("statusstore: read history: %w", err)
	}
	defer func() { _ = rows.Close() }()

	observed := map[string]map[string]State{}
	for rows.Next() {
		var serviceName, observedAt string
		var state State
		if err := rows.Scan(&serviceName, &observedAt, &state); err != nil {
			return nil, fmt.Errorf("statusstore: scan history: %w", err)
		}
		if observed[serviceName] == nil {
			observed[serviceName] = map[string]State{}
		}
		observed[serviceName][observedAt] = state
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("statusstore: iterate history: %w", err)
	}
	return observed, nil
}

func denseHistory(start, currentHour time.Time, observed map[string]map[string]State) History {
	names := make([]string, 0, len(observed))
	for name := range observed {
		names = append(names, name)
	}
	sort.Strings(names)

	services := make([]ServiceHistory, 0, len(names))
	for _, name := range names {
		hours := make([]Hour, 0, RetentionHours)
		for hour := start; !hour.After(currentHour); hour = hour.Add(time.Hour) {
			observedAt := hour.Format(time.RFC3339)
			state := observed[name][observedAt]
			if state == "" {
				state = StateUnknown
			}
			hours = append(hours, Hour{ObservedAt: observedAt, Status: state})
		}
		services = append(services, ServiceHistory{Name: name, Hours: hours})
	}

	return History{
		WindowHours: RetentionHours,
		Through:     currentHour.Format(time.RFC3339),
		Services:    services,
	}
}

func utcHour(value time.Time) time.Time {
	value = value.UTC()
	return time.Date(value.Year(), value.Month(), value.Day(), value.Hour(), 0, 0, 0, time.UTC)
}
