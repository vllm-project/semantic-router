package extproc

import "github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"

type requestRoutingResolution uint8

const (
	routingUnresolved requestRoutingResolution = iota
	routingRecipe
	routingPassthrough
	// routingDenied means the request named a claimed conditional entrypoint
	// alias, but no rule permitted this caller/path. This must never be
	// treated as passthrough or fall back to a default recipe — callers
	// check IsDenied() and reject the request using DeniedStatus/DeniedReason.
	routingDenied
)

// RequestRoutingContext captures the routing boundary selected by the request
// model without duplicating derived recipe names or representing resolved
// passthrough as an ambiguous bool/nil pair.
type RequestRoutingContext struct {
	resolution   requestRoutingResolution
	recipe       *config.RoutingRecipe
	deniedStatus int
	deniedReason string
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

func (c *RequestRoutingContext) IsPassthrough() bool {
	return c != nil && c.resolution == routingPassthrough
}

// SelectDenied records that the request named a claimed entrypoint alias
// this caller/path may not use. status and reason are the external error
// response to return; the caller must return immediately rather than
// continuing to route the request.
func (c *RequestRoutingContext) SelectDenied(status int, reason string) {
	if c == nil {
		return
	}
	c.recipe = nil
	c.resolution = routingDenied
	c.deniedStatus = status
	c.deniedReason = reason
}

func (c *RequestRoutingContext) IsDenied() bool {
	return c != nil && c.resolution == routingDenied
}

func (c *RequestRoutingContext) DeniedStatus() int {
	if c == nil || c.resolution != routingDenied {
		return 0
	}
	return c.deniedStatus
}

func (c *RequestRoutingContext) DeniedReason() string {
	if c == nil || c.resolution != routingDenied {
		return ""
	}
	return c.deniedReason
}
