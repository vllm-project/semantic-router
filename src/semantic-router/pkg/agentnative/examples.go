package agentnative

import (
	"encoding/base64"
	"encoding/json"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

const (
	defaultExamplePageSize = 5
	maximumExamplePageSize = 10
)

// DistributionExamples presents the immutable built-in Recipe distribution as
// bounded, model-free examples. It never installs resources or reads runtime
// assignments.
type DistributionExamples struct {
	revision string
	items    []RecipeExample
}

func NewDistributionExamples(
	distribution routingmanagement.BuiltInRecipeDistribution,
) (*DistributionExamples, error) {
	if err := distribution.Validate(); err != nil {
		return nil, err
	}
	items := make([]RecipeExample, len(distribution.Recipes))
	for index, recipe := range distribution.Recipes {
		items[index] = RecipeExample{
			SourceID: recipe.SourceID, SourceRevision: recipe.SourceRevision,
			Name: recipe.Input.Name, Description: recipe.Input.Description,
			RecipeDigest: recipe.RecipeDigest,
			Document:     append(json.RawMessage(nil), recipe.Input.Document...),
		}
	}
	return &DistributionExamples{revision: distribution.AssetDigest, items: items}, nil
}

func (examples *DistributionExamples) List(query ExampleQuery) (ExamplePage, error) {
	if examples == nil || examples.revision == "" || strings.TrimSpace(query.Search) != query.Search ||
		strings.TrimSpace(query.Name) != query.Name || len(query.Search) > 256 || len(query.Name) > 256 {
		return ExamplePage{}, agentmanagement.ErrInvalid
	}
	pageSize := query.PageSize
	if pageSize == 0 {
		pageSize = defaultExamplePageSize
	}
	if pageSize < 1 || pageSize > maximumExamplePageSize {
		return ExamplePage{}, agentmanagement.ErrInvalid
	}
	search := strings.ToLower(query.Search)
	matching := make([]RecipeExample, 0, len(examples.items))
	for _, example := range examples.items {
		if query.Name != "" && example.Name != query.Name && example.SourceID != query.Name {
			continue
		}
		if search != "" && !strings.Contains(strings.ToLower(example.Name+" "+example.Description), search) {
			continue
		}
		copyExample := example
		if query.Name == "" {
			copyExample.Document = nil
		} else {
			copyExample.Document = append(json.RawMessage(nil), example.Document...)
		}
		matching = append(matching, copyExample)
	}
	offset, err := examples.decodeCursor(query)
	if err != nil || offset > len(matching) {
		return ExamplePage{}, agentmanagement.ErrInvalid
	}
	end := offset + pageSize
	if end > len(matching) {
		end = len(matching)
	}
	page := ExamplePage{
		Revision: examples.revision, Data: append([]RecipeExample{}, matching[offset:end]...),
		HasMore: end < len(matching), PageSize: pageSize,
	}
	if page.HasMore {
		page.NextCursor, err = examples.encodeCursor(query, end)
		if err != nil {
			return ExamplePage{}, err
		}
	}
	return page, nil
}

type exampleCursor struct {
	Revision string `json:"revision"`
	Search   string `json:"search,omitempty"`
	Name     string `json:"name,omitempty"`
	Offset   int    `json:"offset"`
}

func (examples *DistributionExamples) encodeCursor(query ExampleQuery, offset int) (string, error) {
	payload, err := json.Marshal(exampleCursor{
		Revision: examples.revision, Search: query.Search, Name: query.Name, Offset: offset,
	})
	if err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(payload), nil
}

func (examples *DistributionExamples) decodeCursor(query ExampleQuery) (int, error) {
	if query.Cursor == "" {
		return 0, nil
	}
	payload, err := base64.RawURLEncoding.DecodeString(query.Cursor)
	if err != nil || len(payload) > 2048 {
		return 0, agentmanagement.ErrInvalid
	}
	var cursor exampleCursor
	decoder := json.NewDecoder(strings.NewReader(string(payload)))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&cursor); err != nil || cursor.Revision != examples.revision ||
		cursor.Search != query.Search || cursor.Name != query.Name || cursor.Offset < 1 {
		return 0, agentmanagement.ErrInvalid
	}
	return cursor.Offset, nil
}

var _ ExampleSource = (*DistributionExamples)(nil)
