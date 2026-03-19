import { expect, test } from '@playwright/test';
import { mockAuthenticatedAppShell } from './support/auth';

const runtimeModels = {
  models: [
    {
      name: 'feedback_detector',
      type: 'feedback_detection',
      loaded: true,
      state: 'ready',
      resolved_model_path: 'models/mmbert-feedback-detector-merged',
      registry: {
        local_path: 'models/mmbert-feedback-detector-merged',
        purpose: 'classifier',
        description: 'Current MoM feedback detector baseline.',
        num_classes: 2,
      },
    },
    {
      name: 'category_classifier',
      type: 'intent_classification',
      loaded: true,
      state: 'ready',
      resolved_model_path: 'models/mmbert32k-intent-classifier-merged',
      registry: {
        local_path: 'models/mmbert32k-intent-classifier-merged',
        purpose: 'classifier',
        description: 'Current MoM domain classifier baseline.',
        num_classes: 8,
      },
    },
    {
      name: 'pii_classifier',
      type: 'pii_detection',
      loaded: true,
      state: 'ready',
      resolved_model_path: 'models/mmbert32k-pii-detector-merged',
      registry: {
        local_path: 'models/mmbert32k-pii-detector-merged',
        purpose: 'classifier',
        description: 'Current MoM PII detector baseline.',
        num_classes: 32,
      },
    },
  ],
  summary: {
    loaded_models: 3,
    total_models: 3,
  },
};

const recipesResponse = {
  default_api_base: 'http://envoy.internal:8899',
  default_request_model: 'MoM',
  default_platform: 'amd',
  runtime_models: runtimeModels,
  recipes: [
    {
      key: 'feedback',
      label: 'Improve feedback classifier accuracy',
      goal_templates: ['improve_accuracy'],
      default_dataset: 'llm-semantic-router/feedback-detector-dataset',
      dataset_hint: 'Hugging Face dataset id or local path when the feedback trainer supports it.',
      default_success_threshold_pp: 0.5,
      primary_metric: 'accuracy',
      supports_dataset_override: true,
      supports_hyperparameter_hints: true,
      baseline: {
        label: 'Improve feedback classifier accuracy',
        source: 'runtime',
        runtime_name: 'feedback_detector',
        model_path: 'models/mmbert-feedback-detector-merged',
        model_id: 'models/mmbert-feedback-detector-merged',
        request_model: 'MoM',
        description: 'Current MoM feedback detector baseline.',
      },
    },
    {
      key: 'fact-check',
      label: 'Improve fact-check classifier accuracy',
      goal_templates: ['improve_accuracy'],
      default_dataset: 'llm-semantic-router/fact-check-classification-dataset',
      dataset_hint: 'Offline eval accepts a Hugging Face dataset id or local JSON/CSV path.',
      default_success_threshold_pp: 0.5,
      primary_metric: 'accuracy',
      supports_dataset_override: true,
      supports_hyperparameter_hints: true,
      baseline: {
        label: 'Improve fact-check classifier accuracy',
        source: 'fallback',
        model_id: 'llm-semantic-router/mmbert-fact-check-merged',
        request_model: 'MoM',
        description: 'Current MoM fact-check baseline.',
      },
    },
    {
      key: 'jailbreak',
      label: 'Improve jailbreak classifier accuracy',
      goal_templates: ['improve_accuracy'],
      default_dataset: 'llm-semantic-router/jailbreak-detection-dataset',
      dataset_hint: 'Offline eval accepts a Hugging Face dataset id or local JSON/CSV path.',
      default_success_threshold_pp: 0.5,
      primary_metric: 'accuracy',
      supports_dataset_override: true,
      supports_hyperparameter_hints: true,
      baseline: {
        label: 'Improve jailbreak classifier accuracy',
        source: 'fallback',
        model_id: 'llm-semantic-router/mmbert-jailbreak-detector-merged',
        request_model: 'MoM',
        description: 'Current MoM jailbreak detector baseline.',
      },
    },
    {
      key: 'intent',
      label: 'Improve intent classifier accuracy',
      goal_templates: ['improve_accuracy'],
      default_dataset: 'TIGER-Lab/MMLU-Pro',
      dataset_hint: 'Offline eval accepts a local JSON/CSV override. Training continues to use the built-in MMLU-Pro intent corpus.',
      default_success_threshold_pp: 0.5,
      primary_metric: 'accuracy',
      supports_dataset_override: true,
      supports_hyperparameter_hints: true,
      baseline: {
        label: 'Improve intent classifier accuracy',
        source: 'runtime',
        runtime_name: 'category_classifier',
        model_path: 'models/mmbert32k-intent-classifier-merged',
        model_id: 'models/mmbert32k-intent-classifier-merged',
        request_model: 'MoM',
        description: 'Current MoM intent classifier baseline.',
      },
    },
    {
      key: 'pii',
      label: 'Improve PII classifier accuracy',
      goal_templates: ['improve_accuracy'],
      default_dataset: 'presidio',
      dataset_hint: 'Offline eval accepts a local JSON/CSV override. Training defaults to Presidio plus AI4Privacy unless advanced hints disable it.',
      default_success_threshold_pp: 0.5,
      primary_metric: 'accuracy',
      supports_dataset_override: true,
      supports_hyperparameter_hints: true,
      baseline: {
        label: 'Improve PII classifier accuracy',
        source: 'runtime',
        runtime_name: 'pii_classifier',
        model_path: 'models/mmbert32k-pii-detector-merged',
        model_id: 'models/mmbert32k-pii-detector-merged',
        request_model: 'MoM',
        description: 'Current MoM PII detector baseline.',
      },
    },
    {
      key: 'domain',
      label: 'Explore domain signal classifier',
      goal_templates: ['explore_signal'],
      default_dataset: 'mmlu-prox-en',
      dataset_hint: 'Signal eval dataset id from signal_eval.py.',
      default_success_threshold_pp: 0.5,
      primary_metric: 'accuracy',
      supports_dataset_override: true,
      supports_hyperparameter_hints: true,
      baseline: {
        label: 'Explore domain signal classifier',
        source: 'runtime',
        runtime_name: 'category_classifier',
        model_path: 'models/mmbert32k-intent-classifier-merged',
        model_id: 'models/mmbert32k-intent-classifier-merged',
        request_model: 'MoM',
        description: 'Current MoM domain classifier baseline.',
      },
    },
  ],
};

function buildCampaign(overrides: {
  id: string;
  name: string;
  apiBase: string;
  requestModel: string;
  datasetOverride: string;
}) {
  return {
    id: overrides.id,
    name: overrides.name,
    status: 'completed',
    goal_template: 'improve_accuracy',
    target: 'feedback',
    platform: 'amd',
    primary_metric: 'accuracy',
    success_threshold_pp: 0.5,
    budget: { max_trials: 2 },
    created_at: '2026-03-19T10:00:00Z',
    updated_at: '2026-03-19T10:05:00Z',
    completed_at: '2026-03-19T10:05:00Z',
    default_api_base: recipesResponse.default_api_base,
    api_base: overrides.apiBase,
    default_request_model: recipesResponse.default_request_model,
    request_model: overrides.requestModel,
    overrides: {
      api_base_override: overrides.apiBase,
      request_model_override: overrides.requestModel,
      dataset_override: overrides.datasetOverride,
      hyperparameter_hints: { epochs: 6 },
      allow_cpu_dry_run: true,
    },
    recipe: recipesResponse.recipes[0],
    baseline: recipesResponse.recipes[0].baseline,
    baseline_eval: {
      source: 'offline_eval',
      dataset: overrides.datasetOverride,
      accuracy: 0.81,
      f1: 0.8,
      precision: 0.79,
      recall: 0.8,
    },
    runtime_baseline: {
      source: 'runtime_signal_eval',
      dataset: 'feedback-en',
      accuracy: 0.78,
    },
    best_trial: {
      index: 2,
      name: 'trial-02',
      status: 'completed',
      started_at: '2026-03-19T10:01:00Z',
      completed_at: '2026-03-19T10:04:00Z',
      primary_metric: 'accuracy',
      model_path: '/tmp/model-research/trial-02',
      use_lora: true,
      eval: {
        source: 'offline_eval',
        dataset: overrides.datasetOverride,
        accuracy: 0.845,
        f1: 0.835,
        precision: 0.83,
        recall: 0.84,
        improvement_pp: 3.5,
      },
      artifacts: {
        training_dir: '/tmp/model-research/trial-02',
      },
    },
    trials: [
      {
        index: 1,
        name: 'trial-01',
        status: 'completed',
        started_at: '2026-03-19T10:00:30Z',
        completed_at: '2026-03-19T10:02:00Z',
        primary_metric: 'accuracy',
        model_path: '/tmp/model-research/trial-01',
        eval: {
          source: 'offline_eval',
          dataset: overrides.datasetOverride,
          accuracy: 0.825,
          improvement_pp: 1.5,
        },
      },
      {
        index: 2,
        name: 'trial-02',
        status: 'completed',
        started_at: '2026-03-19T10:02:10Z',
        completed_at: '2026-03-19T10:04:00Z',
        primary_metric: 'accuracy',
        model_path: '/tmp/model-research/trial-02',
        eval: {
          source: 'offline_eval',
          dataset: overrides.datasetOverride,
          accuracy: 0.845,
          improvement_pp: 3.5,
        },
      },
    ],
    events: [
      {
        timestamp: '2026-03-19T10:00:00Z',
        kind: 'status',
        level: 'info',
        message: 'Campaign created',
      },
      {
        timestamp: '2026-03-19T10:05:00Z',
        kind: 'status',
        level: 'info',
        message: 'Loop finished; best trial improved by 3.50pp',
        percent: 100,
      },
    ],
    artifact_dir: '/tmp/model-research',
    config_fragment_path: '/tmp/model-research/candidate-config.yaml',
    runtime_models: runtimeModels,
  };
}

test.describe('Auto Research page', () => {
  test('hides advanced overrides by default and creates a campaign with AMD-first defaults', async ({
    page,
  }) => {
    let campaigns: Array<ReturnType<typeof buildCampaign>> = [];
    let createPayload: Record<string, unknown> | null = null;

    await mockAuthenticatedAppShell(page, {
      settings: {
        platform: 'amd',
        envoyUrl: 'http://envoy.internal:8899',
      },
    });

    await page.route('**/api/model-research/recipes', async (route) => {
      await route.fulfill({
        status: 200,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(recipesResponse),
      });
    });

    await page.route('**/api/model-research/campaigns', async (route) => {
      if (route.request().method() === 'GET') {
        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(campaigns),
        });
        return;
      }

      createPayload = route.request().postDataJSON() as Record<string, unknown>;
      const created = buildCampaign({
        id: 'research-feedback-lab',
        name: 'feedback-lab',
        apiBase: 'http://shadow-router.internal:8080',
        requestModel: 'MoM-shadow',
        datasetOverride: 'custom-feedback-eval',
      });
      campaigns = [created];

      await route.fulfill({
        status: 201,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(created),
      });
    });

    await page.route('**/api/model-research/campaigns/*', async (route) => {
      const path = new URL(route.request().url()).pathname;
      if (path.endsWith('/events')) {
        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'text/event-stream' },
          body: '',
        });
        return;
      }
      if (path.endsWith('/stop')) {
        await route.fulfill({
          status: 200,
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status: 'stopping' }),
        });
        return;
      }

      const campaignId = path.split('/').pop();
      const campaign = campaigns.find((item) => item.id === campaignId);
      await route.fulfill({
        status: campaign ? 200 : 404,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(campaign ?? { error: 'not found' }),
      });
    });

    await page.goto('/dashboard');

    await page.getByRole('button', { name: 'System' }).click();
    await page.getByRole('menu', { name: 'System' }).getByRole('menuitem', { name: 'Auto Research' }).click();

    await expect(page.getByRole('heading', { name: 'Auto Research' })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Recent campaigns' })).toBeVisible();
    await expect(page.getByText('Default platform', { exact: true })).toBeVisible();
    await expect(page.getByText('Request model', { exact: true })).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Selected baseline' })).toBeVisible();
    await expect(page.getByText('Runtime key: feedback_detector')).toBeVisible();
    await expect(page.locator('select option[value="intent"]')).toHaveText('Improve intent classifier accuracy');
    await expect(page.locator('select option[value="pii"]')).toHaveText('Improve PII classifier accuracy');

    await expect(page.getByLabel('API base override')).toHaveCount(0);
    await expect(page.getByLabel('Request model override')).toHaveCount(0);

    await page.getByRole('button', { name: 'Show advanced controls' }).click();

    await expect(page.getByLabel('API base override')).toBeVisible();
    await expect(page.getByLabel('Request model override')).toBeVisible();

    await page.getByLabel('Campaign name').fill('feedback-lab');
    await page.getByLabel('API base override').fill('http://shadow-router.internal:8080');
    await page.getByLabel('Request model override').fill('MoM-shadow');
    await page.getByLabel('Dataset override').fill('custom-feedback-eval');
    await page.getByLabel('Partial hyperparameter hints (JSON)').fill('{"epochs": 6}');
    await page.getByRole('checkbox', { name: 'Allow CPU dry run when AMD is unavailable' }).check();

    await page.getByRole('button', { name: 'Start campaign' }).click();

    await expect.poll(() => createPayload).not.toBeNull();
    expect(createPayload).toMatchObject({
      name: 'feedback-lab',
      goal_template: 'improve_accuracy',
      target: 'feedback',
      success_threshold_pp: 0.5,
      overrides: {
        api_base_override: 'http://shadow-router.internal:8080',
        request_model_override: 'MoM-shadow',
        dataset_override: 'custom-feedback-eval',
        allow_cpu_dry_run: true,
        hyperparameter_hints: { epochs: 6 },
      },
    });

    await expect(page.getByRole('heading', { name: 'feedback-lab' })).toBeVisible();
    await expect(page.getByText('http://shadow-router.internal:8080')).toBeVisible();
    await expect(page.getByText('MoM-shadow')).toBeVisible();
    await expect(page.getByText('/tmp/model-research/candidate-config.yaml')).toBeVisible();
    await expect(page.getByRole('heading', { name: 'Progress trend' })).toBeVisible();
    await expect(page.getByText('Rounds recorded')).toBeVisible();
    await expect(page.getByText('Elapsed')).toBeVisible();
    await expect(page.getByRole('cell', { name: 'trial-02', exact: true })).toBeVisible();
    await expect(page.getByText('Loop finished; best trial improved by 3.50pp')).toBeVisible();
  });
});
