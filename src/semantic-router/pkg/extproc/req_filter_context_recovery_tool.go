package extproc

import (
	"encoding/json"
	"fmt"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/contextcompression"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/llmprotocol"
)

// The router exposes exactly one reserved retrieval tool per request, whatever
// mix of context actions issued recovery keys. Every action registers its keys
// here so the model sees a single tool whose accepted keys are the union.

// contextRecoveryToolConflict reports whether the request already defines the
// reserved tool name itself. Actions preflight this before removing anything:
// discovering the collision after a commit would leave unreachable content.
func contextRecoveryToolConflict(
	ctx *RequestContext,
	request *llmprotocol.Request,
) bool {
	if request == nil || (ctx != nil && ctx.ContextRecoveryToolOwned) {
		return false
	}
	for _, tool := range request.Tools {
		if tool.Name == contextcompression.RetrieveToolName {
			return true
		}
	}
	return false
}

// registerContextRecoveryKeys records newly issued keys on the request and
// keeps the reserved tool's accepted-key set in sync. Keys stay out of
// receipts and metrics; only their count is ever reported.
func registerContextRecoveryKeys(
	ctx *RequestContext,
	request *llmprotocol.Request,
	keys ...string,
) error {
	if ctx == nil || request == nil || len(keys) == 0 {
		return nil
	}
	if contextRecoveryToolConflict(ctx, request) {
		return fmt.Errorf("request defines reserved tool %q", contextcompression.RetrieveToolName)
	}
	known := stringSet(ctx.ContextCompressionRecoveryKeys)
	for _, key := range keys {
		if key == "" {
			continue
		}
		if _, seen := known[key]; seen {
			continue
		}
		known[key] = struct{}{}
		ctx.ContextCompressionRecoveryKeys = append(ctx.ContextCompressionRecoveryKeys, key)
	}
	if len(ctx.ContextCompressionRecoveryKeys) == 0 {
		return nil
	}
	return applyContextRecoveryTool(ctx, request)
}

// applyContextRecoveryTool installs the reserved tool, or updates the accepted
// keys of the one this request already installed.
func applyContextRecoveryTool(ctx *RequestContext, request *llmprotocol.Request) error {
	schema, err := contextRecoveryToolSchema(ctx.ContextCompressionRecoveryKeys)
	if err != nil {
		return err
	}
	for index := range request.Tools {
		if request.Tools[index].Name != contextcompression.RetrieveToolName {
			continue
		}
		if !ctx.ContextRecoveryToolOwned {
			return fmt.Errorf("request defines reserved tool %q", contextcompression.RetrieveToolName)
		}
		request.Tools[index].InputSchema = schema
		return nil
	}
	request.Tools = append(request.Tools, llmprotocol.Tool{
		Name:        contextcompression.RetrieveToolName,
		Description: "Retrieve original context omitted by vLLM Semantic Router context actions.",
		InputSchema: schema,
	})
	ctx.ContextRecoveryToolOwned = true
	return nil
}

func contextRecoveryToolSchema(keys []string) ([]byte, error) {
	return json.Marshal(map[string]interface{}{
		"type": "object",
		"properties": map[string]interface{}{
			"key": map[string]interface{}{"type": "string", "enum": keys},
		},
		"required": []string{"key"},
	})
}
