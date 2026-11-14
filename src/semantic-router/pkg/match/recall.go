package match

import (
	"context"
	"fmt"
	"slices"
	"sort"
	"time"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/vectorstore"
)

// recall returns the candidates most similar to the request, most similar
// first: the ones the index recalls, or the listed ones ranked by BM25. Its
// bounds hold whatever the index returned.
func recall(ctx context.Context, spec config.MatchRecall, input Input) ([]Scored, Stage, error) {
	stage := Stage{Name: StageRecall, Method: spec.EffectiveMethod()}
	started := time.Now()
	var recalled []Scored
	var err error
	switch {
	case input.Index != nil:
		if recalled, err = input.Index.Recall(ctx, input.Request, spec); err == nil {
			err = distinctIDs(len(recalled), func(i int) string { return recalled[i].ID })
		}
	case stage.Method == config.MatchRecallBM25:
		recalled = rankBM25(input.Request, input.Candidates)
	default:
		err = fmt.Errorf("match: %s recall needs an index", stage.Method)
	}
	stage.Duration = time.Since(started)
	if err != nil {
		stage.Outcome, stage.Reason = OutcomeFailed, reason(err)
		return nil, stage, err
	}
	stage.Candidates = len(recalled)
	recalled = bounded(recalled, spec)
	stage.Kept, stage.Outcome = len(recalled), OutcomeOK
	if len(recalled) == 0 {
		stage.Outcome = OutcomeEmpty
	}
	return recalled, stage, nil
}

// rankBM25 scores every listed candidate against the query with the BM25
// index vector stores search; one without a query term scores 0.
func rankBM25(query string, candidates []Candidate) []Scored {
	chunks := make(map[string]vectorstore.EmbeddedChunk, len(candidates))
	for _, candidate := range candidates {
		chunks[candidate.ID] = vectorstore.EmbeddedChunk{ID: candidate.ID, Content: candidate.Text}
	}
	scores := vectorstore.NewBM25Index(chunks).Score(query, vectorstore.DefaultBM25K1, vectorstore.DefaultBM25B)
	ranked := make([]Scored, len(candidates))
	for index, candidate := range candidates {
		ranked[index] = Scored{Candidate: candidate, Similarity: scores[candidate.ID]}
	}
	return ranked
}

// bounded orders recalled candidates by similarity, ties in the order they
// came, and keeps those within the recall's bounds, scored by similarity.
func bounded(recalled []Scored, spec config.MatchRecall) []Scored {
	recalled = slices.Clone(recalled)
	sort.SliceStable(recalled, func(i, j int) bool { return recalled[i].Similarity > recalled[j].Similarity })
	kept := recalled[:0]
	for _, candidate := range recalled {
		if spec.MinSimilarity != nil && candidate.Similarity < *spec.MinSimilarity {
			continue
		}
		candidate.Score = candidate.Similarity
		kept = append(kept, candidate)
	}
	if spec.TopK > 0 && len(kept) > spec.TopK {
		kept = kept[:spec.TopK]
	}
	return kept
}
