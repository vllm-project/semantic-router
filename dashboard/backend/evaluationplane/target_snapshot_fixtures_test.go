package evaluationplane

const modelArmTestYAML = `version: v0.3
global:
  router:
    auto_model_names: [entrypoint-a]
providers:
  defaults:
    default_model: Org/Fast Model
  models:
    - name: Org/Fast Model
      provider_model_id: PrivateOrg/Secret-Upstream-ID
      backend_refs:
        - name: private-primary
          endpoint: private.models.example.test:8000/v1
          api_key: literal-test-secret
      pricing:
        currency: USD
        prompt_per_1m: 1.25
        completion_per_1m: 4.5
    - name: local/omni
      backend_refs:
        - name: local-primary
          endpoint: local-private.example.test:8001
    - name: metadata-only
      provider_model_id: metadata-only-upstream
      pricing:
        currency: USD
        prompt_per_1m: 0.1
        completion_per_1m: 0.2
    - name: non-usd
      provider_model_id: foreign-upstream
      backend_refs:
        - endpoint: foreign-private.example.test:8002
      pricing:
        currency: EUR
        prompt_per_1m: 0.1
        completion_per_1m: 0.2
    - name: ambiguous-unpriced
      backend_refs:
        - endpoint: ambiguous-private.example.test:8003
      pricing:
        prompt_per_1m: 0.1
        completion_per_1m: 0.2
routing:
  modelCards:
    - name: Org/Fast Model
      param_size: 7B
      context_window_size: 32768
      capabilities: [vision, chat, vision, ""]
      modality: ar
    - name: local/omni
      capabilities: [omni, audio, document_understanding, video_understanding]
      modality: omni
    - name: metadata-only
      modality: text
    - name: non-usd
      modality: text
    - name: ambiguous-unpriced
      modality: text
  decisions:
    - name: route
      rules: {}
      modelRefs:
        - model: Org/Fast Model
        - model: local/omni
`

const multiRecipeMixtureTestYAML = `version: v0.3
global:
  router:
    auto_model_names: [primary-auto, default-alias]
  integrations:
    looper:
      endpoint: http://envoy.test/v1/chat/completions
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs: [{endpoint: model-a.test:8000}]
    - name: model-b
      backend_refs: [{endpoint: model-b.test:8000}]
    - name: model-c
      backend_refs: [{endpoint: model-c.test:8000}]
    - name: selector
      backend_refs: [{endpoint: selector.test:8000}]
routing:
  modelCards:
    - {name: model-a, modality: text}
    - {name: model-b, modality: text}
    - {name: model-c, modality: text}
    - {name: selector, modality: text}
  decisions:
    - name: default-route
      rules: {}
      modelRefs: [{model: model-a}, {model: model-b}]
      algorithm:
        type: prompt
        prompt: {model: selector, instructions: "Choose one candidate."}
recipes:
  - name: privacy
    description: "  Private lane  "
    routing:
      decisions:
        - name: privacy-route
          rules: {}
          modelRefs: [{model: model-c}]
entrypoints:
  - model_names: [privacy-primary]
    recipe: privacy
  - model_names: [privacy-alias]
    recipe: privacy
`

const providerDefaultDecisionTestYAML = `version: v0.3
global:
  router:
    auto_model_names: [default-mom]
providers:
  defaults:
    default_model: model-a
  models:
    - name: model-a
      backend_refs: [{endpoint: model-a.test:8000}]
routing:
  modelCards:
    - {name: model-a, modality: text}
  decisions:
    - name: provider-default
      rules: {}
`
