# Semantic Router Test Suite

This test suite provides a progressive approach to testing the Semantic Router, following the data flow from client request to final response.

## Test Flow

1. **00-client-request-test.py** - Basic client request tests
   - Tests sending requests to the Envoy proxy
   - Verifies basic request formatting and endpoint availability

2. **01-envoy-extproc-test.py** - Envoy request handling tests
   - Tests that Envoy correctly forwards requests to the ExtProc
   - Checks header propagation

3. **02-router-classification-test.py** - Request classification tests
   - Tests BERT embeddings
   - Tests category classification
   - Verifies model selection based on content

4. **03-model-routing-test.py** - Model routing tests
   - Tests that requests are routed to the correct backend model
   - Verifies model header modifications

5. **04-cache-test.py** - Semantic cache tests
   - Tests cache hit/miss behavior
   - Verifies similarity thresholds
   - Tests cache TTL

6. **05-e2e-category-test.py** - End-to-end category-specific tests 
   - Tests math queries route to the math-specialized model
   - Tests creative queries route to the creative-specialized model
   - Tests other domain-specific routing

7. **06-metrics-test.py** - Metrics/monitoring tests
   - Tests Prometheus metrics endpoints
   - Verifies correct metrics are being recorded

## Running Tests

Individual tests can be run with:

```
python tests/XX-test-name.py
```

Or run all tests sequentially with:

```
cd tests && python -m pytest
```

## Prerequisites

- Envoy must be running (make run-envoy)
- Router must be running (make run-router)
- Python dependencies installed 
