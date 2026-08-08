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
	"sync/atomic"
	"time"

	"github.com/redis/go-redis/v9"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

const (
	defaultWorkflowStateRedisAddress = "localhost:6379"
	defaultWorkflowStateKeyPrefix    = "vllm-sr:flow:state:"
	defaultWorkflowStateFileDir      = "vllm-sr-flow-state"

	// maxStatePayloadBytes caps the serialised size of a single workflow
	// tool state entry. This prevents a runaway workflow with huge tool
	// trajectories from blowing up memory or Redis bandwidth.
	maxStatePayloadBytes = 512 * 1024 // 512 KiB

	// maxMemoryStateEntries caps how many in-flight states the memory
	// backend will hold. After cleanup, if the map is still at capacity
	// we reject the Put to prevent unbounded growth.
	maxMemoryStateEntries = 10_000

	// maxAggregateStateBytes caps the total serialized bytes stored across all
	// in-flight memory/file states to prevent OOMs even if entries are small.
	maxAggregateStateBytes = 100 * 1024 * 1024 // 100 MiB

	// workflowStateSweeperInterval is how often the background goroutine
	// proactively purges expired entries. Keeps memory/file state bounded
	// even when no new requests arrive.
	workflowStateSweeperInterval = 60 * time.Second
)

type workflowToolStateStore interface {
	Put(ctx context.Context, state *workflowPendingToolState) (string, error)
	Take(ctx context.Context, id string) (*workflowPendingToolState, bool, error)
	Clear(ctx context.Context) error
	Close() error
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

// checkPayloadSize rejects payloads that exceed the hard cap. Applied in
// every backend's Put path after json.Marshal so the limit is on wire bytes.
func checkPayloadSize(data []byte) error {
	if len(data) > maxStatePayloadBytes {
		return fmt.Errorf("workflow state payload %d bytes exceeds limit %d", len(data), maxStatePayloadBytes)
	}
	return nil
}

type memoryStateEntry struct {
	state *workflowPendingToolState
	size  int64
}

type workflowMemoryToolStateStore struct {
	mu           sync.Mutex
	ttl          time.Duration
	states       map[string]memoryStateEntry
	currentBytes int64
	done         chan struct{}
	closeOnce    sync.Once
	wg           sync.WaitGroup
}

func newWorkflowMemoryToolStateStore(ttl time.Duration) *workflowMemoryToolStateStore {
	s := &workflowMemoryToolStateStore{
		ttl:    ttl,
		states: map[string]memoryStateEntry{},
		done:   make(chan struct{}),
	}
	s.wg.Add(1)
	go s.sweepLoop()
	return s
}

func (s *workflowMemoryToolStateStore) sweepLoop() {
	defer s.wg.Done()
	ticker := time.NewTicker(workflowStateSweeperInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case now := <-ticker.C:
			s.mu.Lock()
			s.cleanupLocked(now.UTC())
			s.mu.Unlock()
		}
	}
}

func (s *workflowMemoryToolStateStore) Put(_ context.Context, state *workflowPendingToolState) (string, error) {
	normalizeWorkflowToolStateForStore(state)
	// Marshal early to enforce payload cap before touching the map.
	data, err := json.Marshal(state)
	if err != nil {
		return "", fmt.Errorf("marshal workflow state: %w", err)
	}
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return "", sizeErr
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked(time.Now().UTC())
	if len(s.states) >= maxMemoryStateEntries {
		return "", fmt.Errorf("workflow memory state store at capacity (%d entries)", maxMemoryStateEntries)
	}
	var oldSize int64
	if old, exists := s.states[state.ID]; exists {
		oldSize = old.size
	}
	if s.currentBytes+int64(len(data))-oldSize > maxAggregateStateBytes {
		return "", fmt.Errorf("workflow memory state store at capacity (%d bytes, max %d)", s.currentBytes+int64(len(data))-oldSize, maxAggregateStateBytes)
	}
	s.states[state.ID] = memoryStateEntry{
		state: state,
		size:  int64(len(data)),
	}
	s.currentBytes += int64(len(data)) - oldSize
	return state.ID, nil
}

func (s *workflowMemoryToolStateStore) Take(_ context.Context, id string) (*workflowPendingToolState, bool, error) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.cleanupLocked(time.Now().UTC())
	entry, ok := s.states[id]
	if ok {
		s.currentBytes -= entry.size
		delete(s.states, id)
	}
	if !ok {
		return nil, false, nil
	}
	return entry.state, true, nil
}

func (s *workflowMemoryToolStateStore) Clear(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.states = map[string]memoryStateEntry{}
	s.currentBytes = 0
	return nil
}

func (s *workflowMemoryToolStateStore) Close() error {
	s.closeOnce.Do(func() {
		close(s.done)
		s.wg.Wait()
	})
	return nil
}

func (s *workflowMemoryToolStateStore) cleanupLocked(now time.Time) {
	for id, entry := range s.states {
		if workflowToolStateExpired(entry.state, s.ttl, now) {
			s.currentBytes -= entry.size
			delete(s.states, id)
		}
	}
}

type workflowFileToolStateStore struct {
	dir          string
	ttl          time.Duration
	done         chan struct{}
	closeOnce    sync.Once
	wg           sync.WaitGroup
	currentBytes int64
}

func newWorkflowFileToolStateStore(dir string, ttl time.Duration) *workflowFileToolStateStore {
	storeDir := workflowStateFileDir(dir)
	var initialBytes int64
	if entries, err := os.ReadDir(storeDir); err == nil {
		for _, e := range entries {
			name := e.Name()
			if !e.IsDir() && (strings.HasSuffix(name, ".json") || strings.Contains(name, ".take-") || strings.Contains(name, ".tmp-")) {
				if info, err := e.Info(); err == nil {
					initialBytes += info.Size()
				}
			}
		}
	}
	s := &workflowFileToolStateStore{
		dir:          storeDir,
		ttl:          ttl,
		done:         make(chan struct{}),
		currentBytes: initialBytes,
	}
	s.wg.Add(1)
	go s.sweepLoop()
	return s
}

func (s *workflowFileToolStateStore) sweepLoop() {
	defer s.wg.Done()
	ticker := time.NewTicker(workflowStateSweeperInterval)
	defer ticker.Stop()
	for {
		select {
		case <-s.done:
			return
		case now := <-ticker.C:
			s.cleanupExpired(now.UTC())
		}
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
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return "", sizeErr
	}

	path, err := s.pathForID(state.ID)
	if err != nil {
		return "", err
	}

	var oldSize int64
	if info, err := os.Stat(path); err == nil {
		oldSize = info.Size()
	}

	newSize := int64(len(data))
	diff := newSize - oldSize

	newCurrent := atomic.AddInt64(&s.currentBytes, diff)
	if newCurrent > maxAggregateStateBytes {
		atomic.AddInt64(&s.currentBytes, -diff) // Rollback
		return "", fmt.Errorf("workflow file state store at capacity (%d bytes, max %d)", newCurrent, maxAggregateStateBytes)
	}

	tmp := path + ".tmp-" + newWorkflowToolStateID()
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		atomic.AddInt64(&s.currentBytes, -(newSize - oldSize))
		return "", fmt.Errorf("write workflow state: %w", err)
	}
	if err := os.Rename(tmp, path); err != nil {
		_ = os.Remove(tmp)
		atomic.AddInt64(&s.currentBytes, -(newSize - oldSize))
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
	defer func() {
		if info, statErr := os.Stat(consumePath); statErr == nil {
			atomic.AddInt64(&s.currentBytes, -info.Size())
		}
		os.Remove(consumePath)
	}()

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
		if !strings.HasSuffix(name, ".json") && !strings.Contains(name, ".take-") && !strings.Contains(name, ".tmp-") {
			continue
		}
		path := filepath.Join(s.dir, name)
		if info, statErr := os.Stat(path); statErr == nil {
			if rmErr := os.Remove(path); rmErr != nil && !errors.Is(rmErr, os.ErrNotExist) {
				return fmt.Errorf("remove workflow state %q: %w", name, rmErr)
			} else if rmErr == nil {
				atomic.AddInt64(&s.currentBytes, -info.Size())
			}
		}
	}
	return nil
}

func (s *workflowFileToolStateStore) Close() error {
	s.closeOnce.Do(func() {
		close(s.done)
		s.wg.Wait()
	})
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
			if err := os.Remove(path); err == nil {
				atomic.AddInt64(&s.currentBytes, -int64(len(data)))
			}
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
	if sizeErr := checkPayloadSize(data); sizeErr != nil {
		return "", sizeErr
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

func (s *workflowRedisToolStateStore) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
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
