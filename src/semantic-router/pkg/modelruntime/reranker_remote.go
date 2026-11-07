package modelruntime

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"math"
	"net/http"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/connector"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/modelruntime/tasks"
)

// remoteReranker scores query-document pairs with a rerank deployment: POST
// <endpoint>/rerank {model, query, documents}, answered as the vLLM, Cohere
// and Jina rerank APIs answer, {results: [{index, relevance_score}]}. One
// call scores each run of consecutive pairs that share a query.
type remoteReranker struct {
	client   *connector.Client
	model    string
	identity string
}

var rerankOperation = connector.Operation{Name: "rerank", Method: http.MethodPost, Path: "/rerank", RetrySafe: true}

const (
	defaultRerankTimeout    = 5 * time.Second
	defaultRerankBodyLimit  = 4 << 20
	defaultRerankErrorLimit = 8 << 10
)

func newRemoteReranker(spec config.ResolvedModelBinding) (*remoteReranker, error) {
	deployment := spec.Deployment
	client, err := connector.ForDeployment(deployment, connector.Options{
		AttemptTimeout: defaultRerankTimeout, MaxRetries: 1,
		MaxRequestBytes: defaultRerankBodyLimit, MaxResponseBytes: defaultRerankBodyLimit, MaxErrorBytes: defaultRerankErrorLimit,
	})
	if err != nil {
		return nil, fmt.Errorf("rerank deployment %q: %w", spec.Binding.Deployment, err)
	}
	digest := sha256.Sum256([]byte(deployment.Endpoint + "\x00" + deployment.Model))
	return &remoteReranker{client: client, model: deployment.Model, identity: "rerank:" + hex.EncodeToString(digest[:8])}, nil
}

type rerankRequest struct {
	Model     string   `json:"model"`
	Query     string   `json:"query"`
	Documents []string `json:"documents"`
}

type rerankResponse struct {
	Results []struct {
		Index          *int     `json:"index"`
		RelevanceScore *float64 `json:"relevance_score"`
	} `json:"results"`
}

// ScorePairs keeps the pairs' order; a missing, repeated or non-finite score
// fails the call rather than ranking a document by an absent value.
func (r *remoteReranker) ScorePairs(ctx context.Context, _ string, pairs []tasks.QueryDocument) (tasks.RelevanceScores, error) {
	scores := make([]float32, len(pairs))
	for start := 0; start < len(pairs); {
		end := start + 1
		for end < len(pairs) && pairs[end].Query == pairs[start].Query {
			end++
		}
		documents := make([]string, 0, end-start)
		for _, pair := range pairs[start:end] {
			documents = append(documents, pair.Document)
		}
		body, err := json.Marshal(rerankRequest{Model: r.model, Query: pairs[start].Query, Documents: documents})
		if err != nil {
			return tasks.RelevanceScores{}, err
		}
		data, err := r.client.Do(ctx, rerankOperation, body)
		if err != nil {
			return tasks.RelevanceScores{}, err
		}
		var decoded rerankResponse
		if err := json.Unmarshal(data, &decoded); err != nil {
			return tasks.RelevanceScores{}, fmt.Errorf("invalid rerank response: %w", err)
		}
		seen := make([]bool, len(documents))
		for _, result := range decoded.Results {
			if result.Index == nil || result.RelevanceScore == nil || *result.Index < 0 || *result.Index >= len(documents) || seen[*result.Index] ||
				math.IsNaN(*result.RelevanceScore) || math.IsInf(*result.RelevanceScore, 0) {
				return tasks.RelevanceScores{}, fmt.Errorf("invalid rerank response: each document needs one finite relevance_score")
			}
			seen[*result.Index] = true
			scores[start+*result.Index] = float32(*result.RelevanceScore)
		}
		for _, ok := range seen {
			if !ok {
				return tasks.RelevanceScores{}, fmt.Errorf("invalid rerank response: a document has no relevance_score")
			}
		}
		start = end
	}
	return tasks.RelevanceScores{Scores: scores}, nil
}

func (r *remoteReranker) CacheIdentity() string { return r.identity }

// ReportsInputUsage is false: a rerank API reports no per-pair token counts.
func (r *remoteReranker) ReportsInputUsage() bool { return false }

func (r *remoteReranker) Close() error { return r.client.Close() }
