# VSR Decision Tracking Headers

This document describes the VSR (Vector Semantic Router) decision tracking headers that are automatically added to successful responses for debugging and monitoring purposes.

## Overview

The semantic router automatically adds response headers to track VSR decision-making information. These headers help developers and operations teams understand how requests are being processed and routed.

**Headers are only added when:**

1. The request is successful (HTTP status 200-299)
2. The request did not hit the cache
3. VSR made routing decisions during request processing

## Headers Added

### `x-vsr-selected-category`

**Description**: The category selected by VSR during classification.

**Example Values**:

- `math`
- `business`
- `biology`
- `computer_science`

**When Added**: When VSR successfully classifies the request into a category.

### `x-vsr-selected-reasoning`

**Description**: Whether reasoning mode was determined to be used for this request.

**Values**:

- `on` - Reasoning mode was enabled
- `off` - Reasoning mode was disabled

**When Added**: When VSR makes a reasoning mode decision (both for auto and explicit model selection).

### `x-vsr-selected-model`

**Description**: The model selected by VSR for processing the request.

**Example Values**:

- `deepseek-v31`
- `phi4`
- `gpt-4`

**When Added**: When VSR selects a model (either through auto-routing or explicit model specification).

## Use Cases

### Debugging

These headers help developers understand:

- Which category VSR classified their request into
- Whether reasoning mode was applied
- Which model was ultimately selected

### Monitoring

Operations teams can use these headers to:

- Track category distribution across requests
- Monitor reasoning mode usage patterns
- Analyze model selection patterns

### Analytics

Product teams can analyze:

- Request categorization accuracy
- Reasoning mode effectiveness
- Model performance by category

## Example Response

```http
HTTP/1.1 200 OK
Content-Type: application/json
x-vsr-selected-category: math
x-vsr-selected-reasoning: on
x-vsr-selected-model: deepseek-v31

{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "deepseek-v31",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "The derivative of x^2 + 3x + 1 is 2x + 3."
      },
      "finish_reason": "stop"
    }
  ]
}
```

## When Headers Are NOT Added

Headers are not added in the following cases:

1. **Cache Hit**: When the response comes from cache, no VSR processing occurs
2. **Error Responses**: When the upstream returns 4xx or 5xx status codes
3. **Missing VSR Information**: When VSR decision information is not available (shouldn't happen in normal operation)
