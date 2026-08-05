package looper

import (
	"context"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultWorkflowStateRedisAddress = "localhost:6379"
	defaultWorkflowStateKeyPrefix    = "vllm-sr:flow:state:"
	defaultWorkflowStateFileDir      = "vllm-sr-flow-state"
)

type workflowToolStateStore interface {
	Put(ctx context.Context, state *workflowPendingToolState) (string, error)
	Take(ctx context.Context, id string) (*workflowPendingToolState, bool, error)
	Clear(ctx context.Context) error
}

type workflowStepResultState struct {
	Step             workflowPlanStep                   `json:"step"`
	Responses        []*ModelResponse                   `json:"responses,omitempty"`
	Failed           []FusionFailedModel                `json:"failed,omitempty"`
	ToolTrajectories map[string][]workflowAgentToolTurn `json:"tool_trajectories,omitempty"`
}

func (r workflowStepResult) MarshalJSON() ([]byte, error) {
	return json.Marshal(workflowStepResultState{
		Step:             r.step,
		Responses:        r.responses,
		Failed:           r.failed,
		ToolTrajectories: cloneWorkflowToolTrajectories(r.toolTrajectories),
	})
}

func (r *workflowStepResult) UnmarshalJSON(data []byte) error {
	var state workflowStepResultState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}
	r.step = state.Step
	r.responses = state.Responses
	r.failed = state.Failed
	r.toolTrajectories = cloneWorkflowToolTrajectories(state.ToolTrajectories)
	return nil
}

func newWorkflowToolStateStoreFromConfig(flow config.FlowRuntimeConfig) workflowToolStateStore {
	stateCfg := flow.State.WithDefaults()
	switch stateCfg.StoreBackend {
	case config.WorkflowStateBackendMemory:
		return newWorkflowMemoryToolStateStore(stateCfg.TTL())
	case config.WorkflowStateBackendRedis:
		return newWorkflowRedisToolStateStore(stateCfg.Redis, stateCfg.TTL())
	case config.WorkflowStateBackendFile:
		return newWorkflowFileToolStateStore(stateCfg.File.Directory, stateCfg.TTL())
	default:
		logging.ComponentWarnEvent("looper", "workflow_state_backend_unknown", map[string]interface{}{
			"backend":  stateCfg.StoreBackend,
			"fallback": config.WorkflowStateBackendFile,
		})
		return newWorkflowFileToolStateStore(stateCfg.File.Directory, stateCfg.TTL())
	}
}

func normalizeWorkflowToolStateForStore(state *workflowPendingToolState) {
	if state.ID == "" {
		state.ID = newWorkflowToolStateID()
	}
	if state.CreatedAt.IsZero() {
		state.CreatedAt = time.Now().UTC()
	}
}

func workflowToolStateExpired(state *workflowPendingToolState, ttl time.Duration, now time.Time) bool {
	if state == nil || ttl <= 0 {
		return false
	}
	return now.Sub(state.CreatedAt) > ttl
}

type workflowMemoryToolStateStore struct {
	mu     sync.Mutex
	ttl    time.Duration
	states map[string]*workflowPendingToolState
}

func newWorkflowMemoryToolStateStore(ttl time.Duration) *workflowMemoryToolStateStore {
	return &workflowMemoryToolStateStore{
		ttl:    ttl,
		states: map[string]*workflowPendingToolState{},
	}
}

func (s *workflowMemoryToolStateStore) Put(_ context.Context, state *workflowPendingToolState) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked(time.Now().UTC())
	normalizeWorkflowToolStateForStore(state)
	s.states[state.ID] = state
	return state.ID, nil
}

func (s *workflowMemoryToolStateStore) Take(_ context.Context, id string) (*workflowPendingToolState, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked(time.Now().UTC())
	state, ok := s.states[id]
	if ok {
		delete(s.states, id)
	}
	return state, ok, nil
}

func (s *workflowMemoryToolStateStore) Clear(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.states = map[string]*workflowPendingToolState{}
	return nil
}

func (s *workflowMemoryToolStateStore) cleanupLocked(now time.Time) {
	for id, state := range s.states {
		if workflowToolStateExpired(state, s.ttl, now) {
			delete(s.states, id)
		}
	}
}

type workflowFileToolStateStore struct {
	dir string
	ttl time.Duration
}

func newWorkflowFileToolStateStore(dir string, ttl time.Duration) *workflowFileToolStateStore {
	return &workflowFileToolStateStore{
		dir: workflowStateFileDir(dir),
		ttl: ttl,
	}
}

func workflowStateFileDir(dir string) string {
	if strings.TrimSpace(dir) != "" {
		return dir
	}
	base, err := os.UserCacheDir()
	if err != nil || strings.TrimSpace(base) == "" {
		base = os.TempDir()
	}
	return filepath.Join(base, defaultWorkflowStateFileDir)
}

func (s *workflowFileToolStateStore) Put(_ context.Context, state *workflowPendingToolState) (string, error) {
	normalizeWorkflowToolStateForStore(state)
	if err := os.MkdirAll(s.dir, 0o700); err != nil {
		return "", fmt.Errorf("create workflow state directory: %w", err)
	}
	s.cleanupExpired(time.Now().UTC())
	data, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal workflow state: %w", err)
	}
	path, err := s.pathForID(state.ID)
	if err != nil {
		return "", err
	}
	tmp := path + ".tmp-" + newWorkflowToolStateID()
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		return "", fmt.Errorf("write workflow state: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		return "", fmt.Errorf("commit workflow state: %w", err)
	}
	return state.ID, nil
}

func (s *workflowFileToolStateStore) Take(_ context.Context, id string) (*workflowPendingToolState, bool, error) {
	path, err := s.pathForID(id)
	if err != nil {
		return nil, false, err
	}
	consumePath := path + ".take-" + newWorkflowToolStateID()
	if renameErr := os.Rename(path, consumePath); renameErr != nil {
		if errors.Is(renameErr, os.ErrNotExist) {
			return nil, false, nil
		}
		return nil, false, fmt.Errorf("claim workflow state: %w", renameErr)
	}
	defer os.Remove(consumePath)

	data, err := os.ReadFile(consumePath)
	if err != nil {
		return nil, false, fmt.Errorf("read workflow state: %w", err)
	}
	var state workflowPendingToolState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, false, fmt.Errorf("parse workflow state: %w", err)
	}
	if workflowToolStateExpired(&state, s.ttl, time.Now().UTC()) {
		return nil, false, nil
	}
	return &state, true, nil
}

func (s *workflowFileToolStateStore) Clear(_ context.Context) error {
	entries, err := os.ReadDir(s.dir)
	if errors.Is(err, os.ErrNotExist) {
		return nil
	}
	if err != nil {
		return fmt.Errorf("list workflow states: %w", err)
	}
	for _, entry := range entries {
		if entry.IsDir() {
			continue
		}
		name := entry.Name()
		if strings.HasSuffix(name, ".json") || strings.Contains(name, ".take-") || strings.Contains(name, ".tmp-") {
			if err := os.Remove(filepath.Join(s.dir, name)); err != nil && !errors.Is(err, os.ErrNotExist) {
				return fmt.Errorf("remove workflow state %q: %w", name, err)
			}
		}
	}
	return nil
}

func (s *workflowFileToolStateStore) cleanupExpired(now time.Time) {
	entries, err := os.ReadDir(s.dir)
	if err != nil {
		return
	}
	for _, entry := range entries {
		if entry.IsDir() || !strings.HasSuffix(entry.Name(), ".json") {
			continue
		}
		path := filepath.Join(s.dir, entry.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var state workflowPendingToolState
		if err := json.Unmarshal(data, &state); err != nil || workflowToolStateExpired(&state, s.ttl, now) {
			_ = os.Remove(path)
		}
	}
}

func (s *workflowFileToolStateStore) pathForID(id string) (string, error) {
	if !validWorkflowStateID(id) {
		return "", fmt.Errorf("invalid workflow state id %q", id)
	}
	return filepath.Join(s.dir, id+".json"), nil
}

func validWorkflowStateID(id string) bool {
	if id == "" {
		return false
	}
	for _, ch := range id {
		if (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9') || ch == '-' || ch == '_' {
			continue
		}
		return false
	}
	return true
}

type workflowRedisToolStateStore struct {
	client    *redis.Client
	keyPrefix string
	ttl       time.Duration
}

func newWorkflowRedisToolStateStore(cfg config.WorkflowStateRedisConfig, ttl time.Duration) *workflowRedisToolStateStore {
	address := strings.TrimSpace(cfg.Address)
	if address == "" {
		address = defaultWorkflowStateRedisAddress
	}
	keyPrefix := strings.TrimSpace(cfg.KeyPrefix)
	if keyPrefix == "" {
		keyPrefix = defaultWorkflowStateKeyPrefix
	}
	poolSize := cfg.PoolSize
	if poolSize <= 0 {
		poolSize = 10
	}
	maxRetries := cfg.MaxRetries
	if maxRetries <= 0 {
		maxRetries = 3
	}
	opts := &redis.Options{
		Addr:       address,
		DB:         cfg.DB,
		Password:   cfg.Password,
		PoolSize:   poolSize,
		MaxRetries: maxRetries,
	}
	if cfg.UseTLS {
		opts.TLSConfig = &tls.Config{InsecureSkipVerify: cfg.TLSSkipVerify}
	}
	return &workflowRedisToolStateStore{
		client:    redis.NewClient(opts),
		keyPrefix: keyPrefix,
		ttl:       ttl,
	}
}

func (s *workflowRedisToolStateStore) Put(ctx context.Context, state *workflowPendingToolState) (string, error) {
	normalizeWorkflowToolStateForStore(state)
	data, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal workflow state: %w", err)
	}
	if err := s.client.Set(ctx, s.key(state.ID), data, s.ttl).Err(); err != nil {
		return "", fmt.Errorf("store workflow state in redis: %w", err)
	}
	return state.ID, nil
}

func (s *workflowRedisToolStateStore) Take(ctx context.Context, id string) (*workflowPendingToolState, bool, error) {
	if !validWorkflowStateID(id) {
		return nil, false, fmt.Errorf("invalid workflow state id %q", id)
	}
	result, err := workflowRedisTakeScript.Run(ctx, s.client, []string{s.key(id)}).Result()
	if errors.Is(err, redis.Nil) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, fmt.Errorf("take workflow state from redis: %w", err)
	}
	data, ok := workflowRedisScriptBytes(result)
	if !ok {
		return nil, false, fmt.Errorf("take workflow state from redis returned %T", result)
	}
	var state workflowPendingToolState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, false, fmt.Errorf("parse workflow state: %w", err)
	}
	if workflowToolStateExpired(&state, s.ttl, time.Now().UTC()) {
		return nil, false, nil
	}
	return &state, true, nil
}

func (s *workflowRedisToolStateStore) Clear(ctx context.Context) error {
	var cursor uint64
	for {
		keys, next, err := s.client.Scan(ctx, cursor, s.keyPrefix+"*", 100).Result()
		if err != nil {
			return fmt.Errorf("scan workflow states in redis: %w", err)
		}
		if len(keys) > 0 {
			if err := s.client.Del(ctx, keys...).Err(); err != nil {
				return fmt.Errorf("clear workflow states in redis: %w", err)
			}
		}
		if next == 0 {
			return nil
		}
		cursor = next
	}
}

func (s *workflowRedisToolStateStore) key(id string) string {
	return s.keyPrefix + id
}

func workflowRedisScriptBytes(value interface{}) ([]byte, bool) {
	switch typed := value.(type) {
	case string:
		return []byte(typed), true
	case []byte:
		return typed, true
	default:
		return nil, false
	}
}

var workflowRedisTakeScript = redis.NewScript(`
local value = redis.call("GET", KEYS[1])
if value then
  redis.call("DEL", KEYS[1])
end
return value
`)
