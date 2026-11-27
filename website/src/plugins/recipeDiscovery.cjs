const { existsSync, readdirSync, readFileSync } = require('node:fs')
const { basename, join, resolve, sep } = require('node:path')
const { load } = require('js-yaml')

const readText = path => readFileSync(path, 'utf8')
const readYaml = (path, text = readText(path)) => load(text, { filename: path })

function cardLimitations(markdown) {
  const sections = markdown.split(/^## /m)
  const limitations = sections.find(section => /^Limitations\s*\r?\n/.test(section))
  return (limitations?.replace(/^Limitations\s*\r?\n/, '') ?? '')
    .trim()
    .split(/\r?\n(?=- )|\r?\n\s*\r?\n/)
    .map(item => item.replace(/^- +/, '').replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
      .replace(/`/g, '').replace(/\s+/g, ' ').trim())
    .filter(Boolean)
}

function modelDeployment(catalog, name) {
  if (!name) return undefined
  const { artifact, provider, device } = catalog.deployments?.[name] ?? {}
  return { name, artifact, provider, device }
}

function readRecipe(directory, models) {
  const id = basename(directory)
  const metadataPath = join(directory, 'metadata.yaml')
  const metadata = readYaml(metadataPath)
  if (!/^[a-z0-9-]+$/.test(id) || metadata?.id !== id || !metadata.name || !metadata.description) {
    throw new Error(`${metadataPath} must describe the recipe directory ${id}`)
  }

  const configPath = join(directory, 'config.yaml')
  const yaml = readText(configPath)
  const config = readYaml(configPath, yaml)
  const modelCatalog = config.global?.model_catalog ?? {}
  const decision = modelDeployment(modelCatalog, modelCatalog.system?.decision_model?.deployment)
  const embedding = modelDeployment(modelCatalog, modelCatalog.embeddings?.default?.deployment)
  const source = `config/recipes/${id}`
  const repository = 'https://github.com/vllm-project/semantic-router'
  const card = metadata.links?.documentation ?? `${repository}/blob/main/${source}/README.md`
  const limitations = cardLimitations(readText(join(directory, 'README.md')))
  if (limitations.length === 0) {
    throw new Error(`${source}/README.md must document the recipe's limitations`)
  }

  const questions = (config.recipes ?? []).flatMap(recipe =>
    (recipe.signals?.systemone ?? [])
      .filter(signal => !signal.deployment || signal.deployment === decision?.name)
      .map(({ name, question }) => ({ name, type: question?.type ?? (question?.preset ? 'preset' : undefined) })),
  )
  const entrypoints = (config.entrypoints ?? []).flatMap(entrypoint => entrypoint.model_names)
  return {
    id,
    name: metadata.name,
    description: metadata.description,
    tags: metadata.tags ?? [],
    card,
    source: metadata.links?.source ?? `${repository}/tree/main/${source}`,
    configPath: `${source}/config.yaml`,
    servePlatform: [decision, embedding].some(model => /^rocm(?::|$)/.test(model?.device ?? ''))
      ? 'rocm'
      : undefined,
    yaml,
    entrypoints: entrypoints.length ? entrypoints : ['vllm-sr/auto'],
    providers: (config.providers?.models ?? []).map(({ name, provider_model_id, api_format }) => {
      const model = provider_model_id ?? name
      const known = models.get(model)
      return { name, model, protocol: api_format, lifecycle: known?.lifecycle }
    }),
    decision,
    embedding,
    questions,
    limitations,
  }
}

function groupDecisionModels(recipes) {
  const models = new Map()
  for (const { decision, name, card, questions } of recipes) {
    const artifact = decision?.artifact
    if (!artifact) continue
    if (!models.has(artifact)) {
      models.set(artifact, {
        artifact,
        card: /^vllm-sr\/[a-z0-9._-]+$/i.test(artifact) ? `https://huggingface.co/${artifact}` : undefined,
        questionTypes: new Set(),
        examples: [],
      })
    }
    const model = models.get(artifact)
    for (const question of questions) {
      if (question.type) model.questionTypes.add(question.type)
    }
    model.examples.push({ name, card, tasks: [...new Set(questions.map(question => question.name))] })
  }
  return [...models.values()].map(model => ({ ...model, questionTypes: [...model.questionTypes].sort() }))
}

module.exports = function recipeDiscoveryPlugin({ siteDir, baseUrl }) {
  const root = resolve(siteDir, '../config/recipes')
  const modelCatalog = join(siteDir, 'static/model-catalog/catalog.json')

  return {
    name: 'recipe-discovery',
    getPathsToWatch() {
      return [`${root.split(sep).join('/')}/*/{metadata.yaml,config.yaml,README.md}`, modelCatalog]
    },
    loadContent() {
      const models = new Map(JSON.parse(readText(modelCatalog)).models
        .filter(model => model.kind === 'physical')
        .map(model => [model.id, model]))
      const recipes = readdirSync(root, { withFileTypes: true })
        // The built-in catalog and syntax examples are not standalone recipes.
        .filter(entry => entry.isDirectory() && existsSync(join(root, entry.name, 'metadata.yaml')))
        .map(entry => entry.name)
        .sort()
        .map(name => readRecipe(join(root, name), models))
      if (!recipes.length) throw new Error(`No maintained recipes found in ${root}`)
      return { recipes, decisionModels: groupDecisionModels(recipes) }
    },
    async contentLoaded({ content, actions }) {
      // Keep the complete YAML in this route's data chunk, not global site data.
      const catalog = await actions.createData('recipes.json', JSON.stringify(content))
      actions.addRoute({
        path: `${baseUrl}recipes`,
        component: '@site/src/components/recipe-discovery/RecipeDiscoveryPage',
        exact: true,
        modules: { catalog },
      })
    },
  }
}
