package responsestore

import (
	"context"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/responseapi"
)

// MemoryStore is an in-memory implementation of CombinedStore.
type MemoryStore struct {
	mu            sync.RWMutex
	responses     map[responseObjectKey]*responseapi.StoredResponse
	conversations map[conversationObjectKey]*responseapi.StoredConversation
	enabled       bool
	ttl           time.Duration
	maxResponses  int
	maxConvs      int
}

type responseObjectKey struct {
	owner responseapi.ResponseOwner
	id    string
}

type conversationObjectKey struct {
	owner responseapi.ResponseOwner
	id    string
}

// NewMemoryStore creates a new in-memory store.
func NewMemoryStore(config StoreConfig) (*MemoryStore, error) {
	ttl := DefaultTTL
	if config.TTLSeconds > 0 {
		ttl = time.Duration(config.TTLSeconds) * time.Second
	}
	maxResponses := config.Memory.MaxResponses
	if maxResponses <= 0 {
		maxResponses = 10000
	}
	maxConvs := config.Memory.MaxConversations
	if maxConvs <= 0 {
		maxConvs = 1000
	}
	store := &MemoryStore{
		responses:     make(map[responseObjectKey]*responseapi.StoredResponse),
		conversations: make(map[conversationObjectKey]*responseapi.StoredConversation),
		enabled:       config.Enabled,
		ttl:           ttl,
		maxResponses:  maxResponses,
		maxConvs:      maxConvs,
	}
	go store.cleanupExpired()
	return store, nil
}

func (m *MemoryStore) IsEnabled() bool { return m.enabled }

func (m *MemoryStore) CheckConnection(ctx context.Context) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	return nil
}

func (m *MemoryStore) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.responses = nil
	m.conversations = nil
	return nil
}

func (m *MemoryStore) StoreResponse(ctx context.Context, owner responseapi.ResponseOwner, response *responseapi.StoredResponse) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil || response == nil || response.ID == "" || response.Owner != owner {
		return ErrInvalidInput
	}
	key := responseObjectKey{owner: owner, id: response.ID}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.responses[key]; exists {
		return ErrAlreadyExists
	}
	if len(m.responses) >= m.maxResponses {
		m.evictOldestResponse()
	}
	if response.TTL.IsZero() {
		response.TTL = time.Now().Add(m.ttl)
	}
	stored := *response
	m.responses[key] = &stored

	// Add to conversation index if ConversationID is set
	if response.ConversationID != "" {
		conversationKey := conversationObjectKey{owner: owner, id: response.ConversationID}
		if conv, exists := m.conversations[conversationKey]; exists && conv.Owner == owner {
			conv.ResponseIDs = append(conv.ResponseIDs, response.ID)
			conv.UpdatedAt = time.Now().Unix()
		}
	}

	return nil
}

func (m *MemoryStore) GetResponse(ctx context.Context, owner responseapi.ResponseOwner, responseID string) (*responseapi.StoredResponse, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return nil, ErrInvalidInput
	}
	if responseID == "" {
		return nil, ErrInvalidID
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	response, exists := m.responses[responseObjectKey{owner: owner, id: responseID}]
	if !exists || response.Owner != owner {
		return nil, ErrNotFound
	}
	if !response.TTL.IsZero() && time.Now().After(response.TTL) {
		return nil, ErrNotFound
	}
	result := *response
	return &result, nil
}

func (m *MemoryStore) UpdateResponse(ctx context.Context, owner responseapi.ResponseOwner, response *responseapi.StoredResponse) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil || response == nil || response.ID == "" || response.Owner != owner {
		return ErrInvalidInput
	}
	key := responseObjectKey{owner: owner, id: response.ID}
	m.mu.Lock()
	defer m.mu.Unlock()
	if stored, exists := m.responses[key]; !exists || stored.Owner != owner {
		return ErrNotFound
	}
	stored := *response
	m.responses[key] = &stored
	return nil
}

func (m *MemoryStore) DeleteResponse(ctx context.Context, owner responseapi.ResponseOwner, responseID string) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return ErrInvalidInput
	}
	if responseID == "" {
		return ErrInvalidID
	}
	key := responseObjectKey{owner: owner, id: responseID}
	m.mu.Lock()
	defer m.mu.Unlock()
	if response, exists := m.responses[key]; !exists || response.Owner != owner {
		return ErrNotFound
	}
	delete(m.responses, key)
	return nil
}

func (m *MemoryStore) GetConversationChain(ctx context.Context, owner responseapi.ResponseOwner, responseID string) ([]*responseapi.StoredResponse, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return nil, ErrInvalidInput
	}
	if responseID == "" {
		return nil, ErrInvalidID
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	var chain []*responseapi.StoredResponse
	currentID := responseID
	for currentID != "" {
		response, exists := m.responses[responseObjectKey{owner: owner, id: currentID}]
		if !exists || response.Owner != owner || (!response.TTL.IsZero() && time.Now().After(response.TTL)) {
			break
		}
		result := *response
		chain = append([]*responseapi.StoredResponse{&result}, chain...)
		currentID = response.PreviousResponseID
	}
	if len(chain) == 0 {
		return nil, ErrNotFound
	}
	return chain, nil
}

func (m *MemoryStore) ListResponsesByConversation(ctx context.Context, owner responseapi.ResponseOwner, conversationID string, opts ListOptions) ([]*responseapi.StoredResponse, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return nil, ErrInvalidInput
	}
	if conversationID == "" {
		return nil, ErrInvalidID
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	conv, exists := m.conversations[conversationObjectKey{owner: owner, id: conversationID}]
	if !exists || conv.Owner != owner {
		return nil, ErrNotFound
	}
	var responses []*responseapi.StoredResponse
	for _, respID := range conv.ResponseIDs {
		if resp, exists := m.responses[responseObjectKey{owner: owner, id: respID}]; exists && resp.Owner == owner {
			if resp.TTL.IsZero() || time.Now().Before(resp.TTL) {
				result := *resp
				responses = append(responses, &result)
			}
		}
	}
	responses = ApplyListOptions(responses, opts)
	return responses, nil
}

func (m *MemoryStore) CreateConversation(ctx context.Context, owner responseapi.ResponseOwner, conversation *responseapi.StoredConversation) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil || conversation == nil || conversation.ID == "" || conversation.Owner != owner {
		return ErrInvalidInput
	}
	key := conversationObjectKey{owner: owner, id: conversation.ID}
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, exists := m.conversations[key]; exists {
		return ErrAlreadyExists
	}
	if len(m.conversations) >= m.maxConvs {
		m.evictOldestConversation()
	}
	if conversation.TTL.IsZero() {
		conversation.TTL = time.Now().Add(m.ttl)
	}
	stored := *conversation
	m.conversations[key] = &stored
	return nil
}

func (m *MemoryStore) GetConversation(ctx context.Context, owner responseapi.ResponseOwner, conversationID string) (*responseapi.StoredConversation, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return nil, ErrInvalidInput
	}
	if conversationID == "" {
		return nil, ErrInvalidID
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	conv, exists := m.conversations[conversationObjectKey{owner: owner, id: conversationID}]
	if !exists || conv.Owner != owner {
		return nil, ErrNotFound
	}
	if !conv.TTL.IsZero() && time.Now().After(conv.TTL) {
		return nil, ErrNotFound
	}
	result := *conv
	return &result, nil
}

func (m *MemoryStore) UpdateConversation(ctx context.Context, owner responseapi.ResponseOwner, conversation *responseapi.StoredConversation) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil || conversation == nil || conversation.ID == "" || conversation.Owner != owner {
		return ErrInvalidInput
	}
	key := conversationObjectKey{owner: owner, id: conversation.ID}
	m.mu.Lock()
	defer m.mu.Unlock()
	if stored, exists := m.conversations[key]; !exists || stored.Owner != owner {
		return ErrNotFound
	}
	stored := *conversation
	m.conversations[key] = &stored
	return nil
}

func (m *MemoryStore) DeleteConversation(ctx context.Context, owner responseapi.ResponseOwner, conversationID string, deleteResponses bool) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return ErrInvalidInput
	}
	if conversationID == "" {
		return ErrInvalidID
	}
	key := conversationObjectKey{owner: owner, id: conversationID}
	m.mu.Lock()
	defer m.mu.Unlock()
	conv, exists := m.conversations[key]
	if !exists || conv.Owner != owner {
		return ErrNotFound
	}
	if deleteResponses {
		for _, respID := range conv.ResponseIDs {
			delete(m.responses, responseObjectKey{owner: owner, id: respID})
		}
	}
	delete(m.conversations, key)
	return nil
}

func (m *MemoryStore) ListConversations(ctx context.Context, owner responseapi.ResponseOwner, opts ListOptions) ([]*responseapi.StoredConversation, error) {
	if !m.enabled {
		return nil, ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return nil, ErrInvalidInput
	}
	m.mu.RLock()
	defer m.mu.RUnlock()
	var convs []*responseapi.StoredConversation
	for _, conv := range m.conversations {
		if conv.Owner == owner && (conv.TTL.IsZero() || time.Now().Before(conv.TTL)) {
			result := *conv
			convs = append(convs, &result)
		}
	}
	sort.Slice(convs, func(i, j int) bool {
		if opts.Order == "asc" {
			return convs[i].CreatedAt < convs[j].CreatedAt
		}
		return convs[i].CreatedAt > convs[j].CreatedAt
	})
	convs = ApplyConvListOptions(convs, opts)
	return convs, nil
}

func (m *MemoryStore) AddResponseToConversation(ctx context.Context, owner responseapi.ResponseOwner, conversationID, responseID string) error {
	if !m.enabled {
		return ErrStoreDisabled
	}
	if validateOwner(owner) != nil {
		return ErrInvalidInput
	}
	if conversationID == "" || responseID == "" {
		return ErrInvalidID
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	conv, exists := m.conversations[conversationObjectKey{owner: owner, id: conversationID}]
	if !exists || conv.Owner != owner {
		return ErrNotFound
	}
	conv.ResponseIDs = append(conv.ResponseIDs, responseID)
	conv.UpdatedAt = time.Now().Unix()
	return nil
}

// Helper methods

func (m *MemoryStore) cleanupExpired() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()
	for range ticker.C {
		m.mu.Lock()
		now := time.Now()
		for id, resp := range m.responses {
			if !resp.TTL.IsZero() && now.After(resp.TTL) {
				delete(m.responses, id)
			}
		}
		for id, conv := range m.conversations {
			if !conv.TTL.IsZero() && now.After(conv.TTL) {
				delete(m.conversations, id)
			}
		}
		m.mu.Unlock()
	}
}

func (m *MemoryStore) evictOldestResponse() {
	var oldestKey responseObjectKey
	found := false
	var oldestTime int64 = 1<<63 - 1
	for key, resp := range m.responses {
		if resp.CreatedAt < oldestTime {
			oldestTime = resp.CreatedAt
			oldestKey = key
			found = true
		}
	}
	if found {
		delete(m.responses, oldestKey)
	}
}

func (m *MemoryStore) evictOldestConversation() {
	var oldestKey conversationObjectKey
	found := false
	var oldestTime int64 = 1<<63 - 1
	for key, conv := range m.conversations {
		if conv.CreatedAt < oldestTime {
			oldestTime = conv.CreatedAt
			oldestKey = key
			found = true
		}
	}
	if found {
		delete(m.conversations, oldestKey)
	}
}
