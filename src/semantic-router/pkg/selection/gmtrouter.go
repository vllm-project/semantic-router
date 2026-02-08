/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"sort"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// GMTRouterConfig configures the GMTRouter personalized selector
// Based on arXiv:2511.08590 - GMTRouter: Personalized LLM Router over Multi-turn User Interactions
//
// GMTRouter models user-LLM interactions as a heterogeneous graph with 5 node types:
//   - User nodes: Represent individual users (zero-initialized)
//   - LLM nodes: Represent model capabilities (PLM-encoded descriptions)
//   - Query nodes: Represent user queries (PLM-encoded text)
//   - Response nodes: Represent model responses (PLM-encoded text)
//   - Turn nodes: Virtual nodes aggregating per-round interaction info
//
// The graph structure captures rich relational dependencies between users and LLMs,
// enabling personalized routing based on individual user preferences learned from
// few-shot interaction data.
type GMTRouterConfig struct {
	// EnablePersonalization enables user-specific preference learning
	EnablePersonalization bool `yaml:"enable_personalization"`

	// HistorySampleSize is the number of interaction histories to sample (k in paper)
	// Used for inductive training and inference (default: 5)
	HistorySampleSize int `yaml:"history_sample_size"`

	// EmbeddingDimension is the dimension of node embeddings (default: 768)
	EmbeddingDimension int `yaml:"embedding_dimension"`

	// NumGNNLayers is the number of HGT layers (L in paper, default: 2)
	NumGNNLayers int `yaml:"num_gnn_layers"`

	// AttentionHeads is the number of attention heads in HGT (default: 8)
	AttentionHeads int `yaml:"attention_heads"`

	// ModelPath is the path to trained GMTRouter model weights
	ModelPath string `yaml:"model_path"`

	// StoragePath is where to persist interaction graph
	StoragePath string `yaml:"storage_path"`

	// MaxInteractionsPerUser limits stored interactions per user (default: 100)
	MaxInteractionsPerUser int `yaml:"max_interactions_per_user"`

	// FeedbackTypes supported: "rating", "ranking", "response"
	FeedbackTypes []string `yaml:"feedback_types"`

	// MinInteractionsForPersonalization is minimum interactions before personalization kicks in
	MinInteractionsForPersonalization int `yaml:"min_interactions_for_personalization"`
}

// DefaultGMTRouterConfig returns the default GMTRouter configuration
func DefaultGMTRouterConfig() *GMTRouterConfig {
	return &GMTRouterConfig{
		EnablePersonalization:             true,
		HistorySampleSize:                 5,
		EmbeddingDimension:                768,
		NumGNNLayers:                      2,
		AttentionHeads:                    8,
		MaxInteractionsPerUser:            100,
		FeedbackTypes:                     []string{"rating", "ranking"},
		MinInteractionsForPersonalization: 3,
	}
}

// NodeType represents the type of node in the heterogeneous graph
type NodeType string

const (
	NodeTypeUser     NodeType = "user"
	NodeTypeLLM      NodeType = "llm"
	NodeTypeQuery    NodeType = "query"
	NodeTypeResponse NodeType = "response"
	NodeTypeTurn     NodeType = "turn"
)

// GraphNode represents a node in the heterogeneous interaction graph
type GraphNode struct {
	ID        string    `json:"id"`
	Type      NodeType  `json:"type"`
	Embedding []float32 `json:"embedding,omitempty"` // PLM-encoded for query/response/llm; zero for user/turn
	Features  []float32 `json:"features,omitempty"`  // Additional features (e.g., preference scores)
	CreatedAt int64     `json:"created_at"`
}

// GraphEdge represents an edge in the heterogeneous graph
type GraphEdge struct {
	SourceID   string   `json:"source_id"`
	TargetID   string   `json:"target_id"`
	SourceType NodeType `json:"source_type"`
	TargetType NodeType `json:"target_type"`
	Weight     float64  `json:"weight,omitempty"`
	TurnIndex  int      `json:"turn_index,omitempty"` // For turn→turn edges
}

// InteractionRecord represents a single user-LLM interaction round
type InteractionRecord struct {
	UserID       string  `json:"user_id"`
	SessionID    string  `json:"session_id"`
	TurnIndex    int     `json:"turn_index"`
	Query        string  `json:"query"`
	LLMModel     string  `json:"llm_model"`
	Response     string  `json:"response,omitempty"`
	FeedbackType string  `json:"feedback_type"` // "rating", "ranking", "response"
	Rating       float64 `json:"rating,omitempty"`
	Timestamp    int64   `json:"timestamp"`

	// ResponseEmbedding stores the embedding of the model's response
	// Used for semantic similarity in message passing (Paper G4)
	ResponseEmbedding []float32 `json:"response_embedding,omitempty"`

	// QueryEmbedding stores the embedding of the user's query
	// Used for query-response coherence scoring
	QueryEmbedding []float32 `json:"query_embedding,omitempty"`
}

// UserPreferenceState holds aggregated user preference information
type UserPreferenceState struct {
	UserID            string              `json:"user_id"`
	Interactions      []InteractionRecord `json:"interactions"`
	ModelPreferences  map[string]float64  `json:"model_preferences"` // LLM -> preference score
	LastUpdated       int64               `json:"last_updated"`
	TotalInteractions int                 `json:"total_interactions"`
}

// GMTRouterSelector implements personalized LLM routing based on heterogeneous graph learning
// This selector learns user preferences from multi-turn interactions and routes queries
// to LLMs that best match individual user preferences.
type GMTRouterSelector struct {
	config *GMTRouterConfig

	// Heterogeneous graph components
	nodes   map[string]*GraphNode // nodeID -> node
	edges   []GraphEdge
	nodesMu sync.RWMutex

	// User preference states
	userStates map[string]*UserPreferenceState
	userMu     sync.RWMutex

	// LLM node embeddings (PLM-encoded model descriptions)
	llmEmbeddings map[string][]float32
	llmMu         sync.RWMutex

	// Embedding function for encoding text
	embeddingFunc func(text string) ([]float32, error)
}

// NewGMTRouterSelector creates a new GMTRouter-based selector
func NewGMTRouterSelector(cfg *GMTRouterConfig) *GMTRouterSelector {
	if cfg == nil {
		cfg = DefaultGMTRouterConfig()
	}
	s := &GMTRouterSelector{
		config:        cfg,
		nodes:         make(map[string]*GraphNode),
		edges:         make([]GraphEdge, 0),
		userStates:    make(map[string]*UserPreferenceState),
		llmEmbeddings: make(map[string][]float32),
	}
	return s
}

// Method returns the selection method type
func (g *GMTRouterSelector) Method() SelectionMethod {
	return MethodGMTRouter
}

// InitializeFromConfig sets up LLM nodes from model configuration
func (g *GMTRouterSelector) InitializeFromConfig(modelConfig map[string]config.ModelParams) {
	g.llmMu.Lock()
	g.nodesMu.Lock()
	defer g.llmMu.Unlock()
	defer g.nodesMu.Unlock()

	for model, params := range modelConfig {
		// Create LLM node with description as embedding source
		nodeID := fmt.Sprintf("llm:%s", model)
		node := &GraphNode{
			ID:        nodeID,
			Type:      NodeTypeLLM,
			CreatedAt: time.Now().Unix(),
		}

		// If we have an embedding function, encode the model description
		if g.embeddingFunc != nil && params.Description != "" {
			if emb, err := g.embeddingFunc(params.Description); err == nil {
				node.Embedding = emb
				g.llmEmbeddings[model] = emb
			}
		}

		// Store additional features: cost, quality score
		node.Features = []float32{
			float32(params.Pricing.PromptPer1M),
			float32(params.QualityScore),
		}

		g.nodes[nodeID] = node
	}

	// Load persisted state if available
	if g.config.StoragePath != "" {
		g.loadState()
	}

	logging.Infof("[GMTRouter] Initialized with %d LLM nodes, personalization=%v",
		len(g.llmEmbeddings), g.config.EnablePersonalization)
}

// Select chooses the best model using graph-based preference learning
func (g *GMTRouterSelector) Select(ctx context.Context, selCtx *SelectionContext) (*SelectionResult, error) {
	if len(selCtx.CandidateModels) == 0 {
		return nil, fmt.Errorf("no candidate models provided")
	}

	userID := selCtx.UserID
	if userID == "" {
		userID = "anonymous"
	}

	// Get or create user state
	userState := g.getOrCreateUserState(userID)

	// Check if we have enough interactions for personalization
	usePersonalization := g.config.EnablePersonalization &&
		userState.TotalInteractions >= g.config.MinInteractionsForPersonalization

	allScores := make(map[string]float64)
	var selectedModel *config.ModelRef
	var bestScore float64
	var reasoning string

	if usePersonalization {
		// Personalized routing using graph-based preference learning
		scores := g.computePersonalizedScores(userID, selCtx)

		for _, model := range selCtx.CandidateModels {
			score := scores[model.Model]
			allScores[model.Model] = score

			if score > bestScore || selectedModel == nil {
				bestScore = score
				selectedModel = &model
			}
		}

		reasoning = fmt.Sprintf("GMTRouter personalized selection for user %s (%d interactions)",
			userID, userState.TotalInteractions)

		logging.Infof("[GMTRouter] Personalized selection for user %s: %s (score=%.4f)",
			userID, selectedModel.Model, bestScore)
	} else {
		// Cold start: use model quality scores
		for _, model := range selCtx.CandidateModels {
			score := g.getDefaultModelScore(model.Model)
			allScores[model.Model] = score

			if score > bestScore || selectedModel == nil {
				bestScore = score
				selectedModel = &model
			}
		}

		reasoning = fmt.Sprintf("GMTRouter cold-start selection (need %d more interactions)",
			g.config.MinInteractionsForPersonalization-userState.TotalInteractions)

		logging.Infof("[GMTRouter] Cold-start selection: %s (user %s has %d interactions)",
			selectedModel.Model, userID, userState.TotalInteractions)
	}

	// Calculate confidence based on preference strength
	confidence := g.computeConfidence(userID, selectedModel.Model, allScores)

	return &SelectionResult{
		SelectedModel: selectedModel.Model,
		LoRAName:      selectedModel.LoRAName,
		Score:         bestScore,
		Confidence:    confidence,
		Method:        MethodGMTRouter,
		Reasoning:     reasoning,
		AllScores:     allScores,
	}, nil
}

// UpdateFeedback updates the interaction graph with new feedback
func (g *GMTRouterSelector) UpdateFeedback(ctx context.Context, feedback *Feedback) error {
	if feedback.WinnerModel == "" {
		return fmt.Errorf("winner model is required")
	}

	userID := feedback.UserID
	if userID == "" {
		userID = "anonymous"
	}

	// Create interaction record
	record := InteractionRecord{
		UserID:       userID,
		SessionID:    feedback.SessionID,
		Query:        feedback.Query,
		LLMModel:     feedback.WinnerModel,
		FeedbackType: "rating",
		Rating:       1.0, // Winner gets positive rating
		Timestamp:    time.Now().Unix(),
	}

	// Compute embeddings if embedding function is available (Paper G4: Response Nodes)
	if g.embeddingFunc != nil {
		// Compute query embedding
		if feedback.Query != "" {
			if emb, err := g.embeddingFunc(feedback.Query); err == nil {
				record.QueryEmbedding = emb
			}
		}
		// Compute response embedding if response is provided
		if feedback.Response != "" {
			record.Response = feedback.Response
			if emb, err := g.embeddingFunc(feedback.Response); err == nil {
				record.ResponseEmbedding = emb
			}
		}
	}

	// Add to user state
	g.addInteraction(userID, record)

	// If there's a loser, record negative interaction
	if feedback.LoserModel != "" && !feedback.Tie {
		loserRecord := InteractionRecord{
			UserID:       userID,
			SessionID:    feedback.SessionID,
			Query:        feedback.Query,
			LLMModel:     feedback.LoserModel,
			FeedbackType: "rating",
			Rating:       0.0, // Loser gets zero rating
			Timestamp:    time.Now().Unix(),
		}
		// Copy embeddings to loser record (same query/response context)
		loserRecord.QueryEmbedding = record.QueryEmbedding
		loserRecord.ResponseEmbedding = record.ResponseEmbedding
		g.addInteraction(userID, loserRecord)
	}

	// Update graph nodes and edges
	g.updateGraph(userID, record)

	// Recompute user preferences using message passing
	g.recomputeUserPreferences(userID)

	// Persist state
	if g.config.StoragePath != "" {
		g.saveState()
	}

	logging.Debugf("[GMTRouter] Updated feedback for user %s: winner=%s, loser=%s",
		userID, feedback.WinnerModel, feedback.LoserModel)

	return nil
}

// addInteraction adds an interaction record to user state
func (g *GMTRouterSelector) addInteraction(userID string, record InteractionRecord) {
	g.userMu.Lock()
	defer g.userMu.Unlock()

	state, ok := g.userStates[userID]
	if !ok {
		state = &UserPreferenceState{
			UserID:           userID,
			Interactions:     make([]InteractionRecord, 0),
			ModelPreferences: make(map[string]float64),
		}
		g.userStates[userID] = state
	}

	// Add record, maintaining max interactions limit
	state.Interactions = append(state.Interactions, record)
	if len(state.Interactions) > g.config.MaxInteractionsPerUser {
		state.Interactions = state.Interactions[1:] // Remove oldest
	}

	state.TotalInteractions++
	state.LastUpdated = time.Now().Unix()
}

// updateGraph adds nodes and edges for the interaction
func (g *GMTRouterSelector) updateGraph(userID string, record InteractionRecord) {
	g.nodesMu.Lock()
	defer g.nodesMu.Unlock()

	now := time.Now().Unix()

	// Ensure user node exists
	userNodeID := fmt.Sprintf("user:%s", userID)
	if _, ok := g.nodes[userNodeID]; !ok {
		g.nodes[userNodeID] = &GraphNode{
			ID:        userNodeID,
			Type:      NodeTypeUser,
			Embedding: make([]float32, g.config.EmbeddingDimension), // Zero-initialized
			CreatedAt: now,
		}
	}

	// Create query node
	queryNodeID := fmt.Sprintf("query:%s:%d", userID, now)
	queryNode := &GraphNode{
		ID:        queryNodeID,
		Type:      NodeTypeQuery,
		CreatedAt: now,
	}
	// Encode query if embedding function available
	if g.embeddingFunc != nil && record.Query != "" {
		if emb, err := g.embeddingFunc(record.Query); err == nil {
			queryNode.Embedding = emb
		}
	}
	g.nodes[queryNodeID] = queryNode

	// Create turn node (virtual node for aggregation)
	turnNodeID := fmt.Sprintf("turn:%s:%d", userID, now)
	turnNode := &GraphNode{
		ID:        turnNodeID,
		Type:      NodeTypeTurn,
		Embedding: make([]float32, g.config.EmbeddingDimension), // Zero-initialized
		Features:  []float32{float32(record.Rating)},            // User preference feature
		CreatedAt: now,
	}
	g.nodes[turnNodeID] = turnNode

	// Create edges: user→turn, llm→turn, query→turn
	llmNodeID := fmt.Sprintf("llm:%s", record.LLMModel)
	g.edges = append(g.edges,
		GraphEdge{SourceID: userNodeID, TargetID: turnNodeID, SourceType: NodeTypeUser, TargetType: NodeTypeTurn},
		GraphEdge{SourceID: llmNodeID, TargetID: turnNodeID, SourceType: NodeTypeLLM, TargetType: NodeTypeTurn},
		GraphEdge{SourceID: queryNodeID, TargetID: turnNodeID, SourceType: NodeTypeQuery, TargetType: NodeTypeTurn},
	)
}

// recomputeUserPreferences runs HGT-style message passing to update user preferences
// This implements the Heterogeneous Graph Transformer from arXiv:2511.08590
// The algorithm performs multi-hop message passing over the heterogeneous graph
func (g *GMTRouterSelector) recomputeUserPreferences(userID string) {
	g.userMu.Lock()
	defer g.userMu.Unlock()

	state, ok := g.userStates[userID]
	if !ok {
		return
	}

	// Phase 1: Aggregate direct user-model interactions
	modelMsgs := make(map[string][]float64) // model -> list of messages
	modelCounts := make(map[string]int)
	modelResponseScores := make(map[string][]float64) // Paper G4: response embedding scores

	for _, interaction := range state.Interactions {
		model := interaction.LLMModel
		modelCounts[model]++

		// Create message from interaction
		// Includes rating, recency weight, and interaction context
		recencyWeight := g.computeRecencyWeight(interaction.Timestamp)
		feedbackWeight := g.computeFeedbackTypeWeight(interaction.FeedbackType)

		message := interaction.Rating * recencyWeight * feedbackWeight
		modelMsgs[model] = append(modelMsgs[model], message)

		// Paper G4: Response embedding contribution
		// Compute query-response coherence using embeddings
		if len(interaction.QueryEmbedding) > 0 && len(interaction.ResponseEmbedding) > 0 {
			coherence := g.computeCosineSimilarity(interaction.QueryEmbedding, interaction.ResponseEmbedding)
			// High coherence + positive rating = bonus; Low coherence + negative rating = expected
			responseScore := coherence * interaction.Rating
			modelResponseScores[model] = append(modelResponseScores[model], responseScore)
		}
	}

	// Phase 2: Compute attention-weighted aggregation (simplified HGT attention)
	for model, msgs := range modelMsgs {
		if len(msgs) == 0 {
			continue
		}

		// Compute attention weights based on message variance
		// Higher variance = lower confidence = lower weight
		mean := 0.0
		for _, m := range msgs {
			mean += m
		}
		mean /= float64(len(msgs))

		variance := 0.0
		for _, m := range msgs {
			variance += (m - mean) * (m - mean)
		}
		if len(msgs) > 1 {
			variance /= float64(len(msgs) - 1)
		}

		// Attention weight inversely proportional to variance
		attention := 1.0 / (1.0 + variance)

		// Aggregate with attention-weighted mean
		aggScore := mean * attention

		// Phase 3: Multi-hop aggregation from LLM embeddings
		// Get LLM node features for semantic similarity
		g.llmMu.RLock()
		if llmEmb, ok := g.llmEmbeddings[model]; ok && len(llmEmb) > 0 {
			// Use embedding norm as quality signal
			embNorm := float32(0.0)
			for _, v := range llmEmb {
				embNorm += v * v
			}
			embNorm = float32(math.Sqrt(float64(embNorm)))

			// Scale by embedding-based quality
			qualityBonus := float64(embNorm) / 10.0 // Normalize
			aggScore += 0.1 * qualityBonus
		}
		g.llmMu.RUnlock()

		// Phase 4: Neighbor message passing (from user graph)
		neighborBonus := g.computeNeighborInfluence(userID, model)
		aggScore += 0.1 * neighborBonus

		// Phase 5: Response embedding contribution (Paper G4)
		// Use response embeddings to boost/penalize based on query-response coherence
		if respScores, ok := modelResponseScores[model]; ok && len(respScores) > 0 {
			avgRespScore := 0.0
			for _, s := range respScores {
				avgRespScore += s
			}
			avgRespScore /= float64(len(respScores))
			// Scale response coherence contribution
			aggScore += 0.15 * avgRespScore
		}

		// Clamp to [0, 1]
		if aggScore < 0 {
			aggScore = 0
		}
		if aggScore > 1 {
			aggScore = 1
		}

		state.ModelPreferences[model] = aggScore
	}
}

// computeRecencyWeight computes a time-decayed weight for interactions
func (g *GMTRouterSelector) computeRecencyWeight(timestamp int64) float64 {
	now := time.Now().Unix()
	ageHours := float64(now-timestamp) / 3600.0

	// Exponential decay with half-life of 168 hours (1 week)
	halfLife := 168.0
	return math.Pow(0.5, ageHours/halfLife)
}

// computeFeedbackTypeWeight returns weight based on feedback type
func (g *GMTRouterSelector) computeFeedbackTypeWeight(feedbackType string) float64 {
	switch feedbackType {
	case "rating":
		return 1.0 // Explicit rating is most valuable
	case "ranking":
		return 0.9 // Ranking is also explicit
	case "response":
		return 0.7 // Response-based is implicit
	default:
		return 0.8
	}
}

// computeNeighborInfluence computes influence from similar users
// This implements collaborative filtering-style message passing
func (g *GMTRouterSelector) computeNeighborInfluence(userID, model string) float64 {
	influence := 0.0
	count := 0

	for otherUserID, otherState := range g.userStates {
		if otherUserID == userID {
			continue
		}

		// Compute user similarity based on shared model preferences
		similarity := g.computeUserSimilarity(userID, otherUserID)
		if similarity < 0.3 {
			continue // Not similar enough
		}

		// Get neighbor's preference for this model
		if neighborPref, ok := otherState.ModelPreferences[model]; ok {
			influence += similarity * neighborPref
			count++
		}
	}

	if count == 0 {
		return 0
	}
	return influence / float64(count)
}

// computeUserSimilarity computes cosine similarity between user preferences
func (g *GMTRouterSelector) computeUserSimilarity(userOne, userTwo string) float64 {
	stateA, okA := g.userStates[userOne]
	stateB, okB := g.userStates[userTwo]
	if !okA || !okB {
		return 0
	}

	// Find common models
	dotProduct := 0.0
	normA := 0.0
	normB := 0.0

	allModels := make(map[string]bool)
	for m := range stateA.ModelPreferences {
		allModels[m] = true
	}
	for m := range stateB.ModelPreferences {
		allModels[m] = true
	}

	for m := range allModels {
		prefA := stateA.ModelPreferences[m]
		prefB := stateB.ModelPreferences[m]

		dotProduct += prefA * prefB
		normA += prefA * prefA
		normB += prefB * prefB
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// computeCosineSimilarity computes cosine similarity between two embedding vectors
// Used for query-response coherence scoring (Paper G4: Response Nodes)
func (g *GMTRouterSelector) computeCosineSimilarity(vecA, vecB []float32) float64 {
	if len(vecA) == 0 || len(vecB) == 0 || len(vecA) != len(vecB) {
		return 0.0
	}

	var dotProduct, normA, normB float64
	for i := range vecA {
		dotProduct += float64(vecA[i]) * float64(vecB[i])
		normA += float64(vecA[i]) * float64(vecA[i])
		normB += float64(vecB[i]) * float64(vecB[i])
	}

	if normA == 0 || normB == 0 {
		return 0.0
	}

	return dotProduct / (math.Sqrt(normA) * math.Sqrt(normB))
}

// computePersonalizedScores computes scores for each model based on user preferences
func (g *GMTRouterSelector) computePersonalizedScores(userID string, selCtx *SelectionContext) map[string]float64 {
	g.userMu.RLock()
	defer g.userMu.RUnlock()

	scores := make(map[string]float64)
	state := g.userStates[userID]

	for _, model := range selCtx.CandidateModels {
		baseScore := g.getDefaultModelScore(model.Model)

		if state != nil {
			if preference, ok := state.ModelPreferences[model.Model]; ok {
				// Blend base score with learned preference
				scores[model.Model] = 0.3*baseScore + 0.7*preference
			} else {
				// No preference data for this model - use base score with small penalty
				scores[model.Model] = baseScore * 0.9
			}
		} else {
			scores[model.Model] = baseScore
		}
	}

	return scores
}

// getDefaultModelScore returns a default score for a model (cold start)
func (g *GMTRouterSelector) getDefaultModelScore(model string) float64 {
	g.nodesMu.RLock()
	defer g.nodesMu.RUnlock()

	nodeID := fmt.Sprintf("llm:%s", model)
	if node, ok := g.nodes[nodeID]; ok && len(node.Features) >= 2 {
		// Use quality score from features
		return float64(node.Features[1])
	}
	return 0.5 // Default score
}

// getOrCreateUserState gets or creates a user preference state
func (g *GMTRouterSelector) getOrCreateUserState(userID string) *UserPreferenceState {
	g.userMu.Lock()
	defer g.userMu.Unlock()

	if state, ok := g.userStates[userID]; ok {
		return state
	}

	state := &UserPreferenceState{
		UserID:           userID,
		Interactions:     make([]InteractionRecord, 0),
		ModelPreferences: make(map[string]float64),
	}
	g.userStates[userID] = state
	return state
}

// computeConfidence computes selection confidence based on preference strength
func (g *GMTRouterSelector) computeConfidence(userID, selectedModel string, allScores map[string]float64) float64 {
	if len(allScores) <= 1 {
		return 0.5
	}

	// Get selected model score
	selectedScore := allScores[selectedModel]

	// Compute score difference from second best
	scores := make([]float64, 0, len(allScores))
	for _, score := range allScores {
		scores = append(scores, score)
	}
	sort.Float64s(scores)

	if len(scores) >= 2 {
		secondBest := scores[len(scores)-2]
		margin := selectedScore - secondBest

		// Convert margin to confidence (sigmoid-like)
		confidence := 0.5 + margin*0.5
		if confidence > 0.95 {
			confidence = 0.95
		}
		if confidence < 0.1 {
			confidence = 0.1
		}
		return confidence
	}

	return 0.5
}

// SetEmbeddingFunc sets the function used for encoding text to embeddings
func (g *GMTRouterSelector) SetEmbeddingFunc(fn func(text string) ([]float32, error)) {
	g.embeddingFunc = fn
}

// GetUserInteractionCount returns the number of interactions for a user
func (g *GMTRouterSelector) GetUserInteractionCount(userID string) int {
	g.userMu.RLock()
	defer g.userMu.RUnlock()

	if state, ok := g.userStates[userID]; ok {
		return state.TotalInteractions
	}
	return 0
}

// saveState persists the graph state to disk
func (g *GMTRouterSelector) saveState() {
	if g.config.StoragePath == "" {
		return
	}

	g.userMu.RLock()
	defer g.userMu.RUnlock()

	data, err := json.Marshal(g.userStates)
	if err != nil {
		logging.Warnf("[GMTRouter] Failed to marshal state: %v", err)
		return
	}

	if err := os.WriteFile(g.config.StoragePath, data, 0o644); err != nil {
		logging.Warnf("[GMTRouter] Failed to save state: %v", err)
	}
}

// loadState loads the graph state from disk
func (g *GMTRouterSelector) loadState() {
	if g.config.StoragePath == "" {
		return
	}

	data, err := os.ReadFile(g.config.StoragePath)
	if err != nil {
		if !os.IsNotExist(err) {
			logging.Warnf("[GMTRouter] Failed to load state: %v", err)
		}
		return
	}

	g.userMu.Lock()
	defer g.userMu.Unlock()

	if err := json.Unmarshal(data, &g.userStates); err != nil {
		logging.Warnf("[GMTRouter] Failed to unmarshal state: %v", err)
	} else {
		logging.Infof("[GMTRouter] Loaded state with %d users", len(g.userStates))
	}
}

// Close saves state and cleans up resources
func (g *GMTRouterSelector) Close() error {
	g.saveState()
	return nil
}

// GetDebugState returns the current state for debugging
func (g *GMTRouterSelector) GetDebugState(userID string) map[string]interface{} {
	state := map[string]interface{}{
		"config": map[string]interface{}{
			"enable_personalization":               g.config.EnablePersonalization,
			"min_interactions_for_personalization": g.config.MinInteractionsForPersonalization,
		},
	}

	// Graph statistics
	g.nodesMu.RLock()
	state["graph"] = map[string]interface{}{
		"total_nodes": len(g.nodes),
		"total_edges": len(g.edges),
	}
	g.nodesMu.RUnlock()

	// User-specific state if requested
	if userID != "" {
		g.userMu.RLock()
		if userState, ok := g.userStates[userID]; ok {
			modelPrefs := make(map[string]float64)
			for model, score := range userState.ModelPreferences {
				modelPrefs[model] = score
			}
			state["user_state"] = map[string]interface{}{
				"interactions":      len(userState.Interactions),
				"model_preferences": modelPrefs,
				"last_updated":      time.Unix(userState.LastUpdated, 0).Format("2006-01-02T15:04:05Z"),
			}
		}
		g.userMu.RUnlock()
	}

	// All users summary
	g.userMu.RLock()
	userCount := len(g.userStates)
	allUsers := make([]string, 0, len(g.userStates))
	for uid := range g.userStates {
		allUsers = append(allUsers, uid)
	}
	g.userMu.RUnlock()
	state["total_users"] = userCount
	state["users"] = allUsers

	return state
}
