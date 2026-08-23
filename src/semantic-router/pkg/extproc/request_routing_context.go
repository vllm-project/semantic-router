package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type requestRoutingResolution uint8

const (
	routingUnresolved requestRoutingResolution = iota
	routingRecipe
	routingPassthrough
)

// RequestRoutingContext captures the routing boundary selected by the request
// model without duplicating derived recipe names or representing resolved
// passthrough as an ambiguous bool/nil pair.
type RequestRoutingContext struct {
	resolution requestRoutingResolution
	recipe     *config.RoutingRecipe
}

// SelectRecipe records an explicit recipe resolution.
func (c *RequestRoutingContext) SelectRecipe(recipe *config.RoutingRecipe) {
	if c == nil {
		return
	}
	if recipe == nil {
		c.recipe = nil
		c.resolution = routingUnresolved
		return
	}
	c.recipe = recipe
	c.resolution = routingRecipe
}

// SelectPassthrough records that the request named a concrete backend model.
func (c *RequestRoutingContext) SelectPassthrough() {
	if c == nil {
		return
	}
	c.recipe = nil
	c.resolution = routingPassthrough
}

func (c *RequestRoutingContext) IsResolved() bool {
	return c != nil && c.resolution != routingUnresolved
}

func (c *RequestRoutingContext) SelectedRecipe() *config.RoutingRecipe {
	if c == nil || c.resolution != routingRecipe {
		return nil
	}
	return c.recipe
}

func (c *RequestRoutingContext) RecipeName() config.RecipeName {
	recipe := c.SelectedRecipe()
	if recipe == nil {
		return ""
	}
	return recipe.Name
}

// RuntimeScope returns the stable namespace for mutable state owned by the
// effective entrypoint view. Several model-name aliases in one mapping share
// a scope, while separate mappings of the same reusable recipe do not.
func (c *RequestRoutingContext) RuntimeScope() config.RecipeName {
	recipe := c.SelectedRecipe()
	if recipe == nil {
		return ""
	}
	return recipe.RuntimeScope()
}

func (c *RequestRoutingContext) IsPassthrough() bool {
	return c != nil && c.resolution == routingPassthrough
}
