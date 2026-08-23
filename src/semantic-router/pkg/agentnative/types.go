// Package agentnative exposes Router-owned application capabilities as
// immutable Agent tools. It is an adapter layer only: authorization, routing
// resources, Skills, and validation remain owned by their domain services.
package agentnative

import (
	"context"
	"encoding/json"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/accesscontrol"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/agentmanagement"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/controlplane/routingmanagement"
)

const (
	toolCatalogDescribe = agentmanagement.ToolCatalogDescribe
	toolSkillsRead      = agentmanagement.ToolSkillsRead
	toolModelsList      = agentmanagement.ToolModelsList
	toolRecipesExamples = agentmanagement.ToolRecipesExamples
	toolRecipeGet       = agentmanagement.ToolRecipeGet
	toolRecipeValidate  = agentmanagement.ToolRecipeValidate
)

// AgentStore is the exact immutable-session seam needed by router.skills.read.
// A Skill may only be read through the revision pinned by the session Profile.
type AgentStore interface {
	GetSession(context.Context, string, string) (agentmanagement.Session, error)
	GetProfileRevision(context.Context, string, string, int64) (agentmanagement.Profile, error)
	GetSkillRevision(context.Context, string, string, int64) (agentmanagement.Skill, error)
}

// RoutingReader exposes only authorization-scoped reads. Provider bindings and
// credentials are intentionally absent from the values returned by this package.
type RoutingReader interface {
	ListModels(context.Context, string, routingmanagement.PageRequest) (routingmanagement.Page[routingmanagement.Model], error)
	GetRecipe(context.Context, string, string) (routingmanagement.Recipe, error)
}

// ScopeResolver compiles current Management authority into a repository-safe
// result scope. Tool input can never supply or widen this value.
type ScopeResolver interface {
	ResolveResultScope(
		context.Context,
		accesscontrol.ManagementPrincipalID,
		accesscontrol.NamespaceID,
		accesscontrol.Permission,
	) (accesscontrol.ResultScope, error)
}

type ComponentKind string

const (
	ComponentSignal     ComponentKind = "signal"
	ComponentProjection ComponentKind = "projection"
	ComponentDecision   ComponentKind = "decision"
	ComponentAlgorithm  ComponentKind = "algorithm"
	ComponentPlugin     ComponentKind = "plugin"
)

func (kind ComponentKind) valid() bool {
	switch kind {
	case ComponentSignal, ComponentProjection, ComponentDecision, ComponentAlgorithm, ComponentPlugin:
		return true
	default:
		return false
	}
}

type CatalogQuery struct {
	Kind     ComponentKind
	Name     string
	PageSize int
	Cursor   string
}

type ComponentDescriptor struct {
	Kind        ComponentKind   `json:"kind"`
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Tier        string          `json:"tier,omitempty"`
	Execution   string          `json:"execution,omitempty"`
	Schema      json.RawMessage `json:"schema,omitempty"`
}

type CatalogPage struct {
	Revision   string                `json:"revision"`
	Data       []ComponentDescriptor `json:"data"`
	NextCursor string                `json:"nextCursor,omitempty"`
	HasMore    bool                  `json:"hasMore"`
	PageSize   int                   `json:"pageSize"`
}

// CatalogSource is replaceable so future component registries can extend the
// Router without changing Agent prompts, tool definitions, or Dashboard code.
type CatalogSource interface {
	Describe(CatalogQuery) (CatalogPage, error)
}

type ExampleQuery struct {
	Search   string
	Name     string
	PageSize int
	Cursor   string
}

type RecipeExample struct {
	SourceID       string          `json:"sourceId"`
	SourceRevision int64           `json:"sourceRevision"`
	Name           string          `json:"name"`
	Description    string          `json:"description,omitempty"`
	RecipeDigest   string          `json:"recipeDigest"`
	Document       json.RawMessage `json:"document"`
}

type ExamplePage struct {
	Revision   string          `json:"revision"`
	Data       []RecipeExample `json:"data"`
	NextCursor string          `json:"nextCursor,omitempty"`
	HasMore    bool            `json:"hasMore"`
	PageSize   int             `json:"pageSize"`
}

type ExampleSource interface {
	List(ExampleQuery) (ExamplePage, error)
}
