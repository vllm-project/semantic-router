import Link from '@docusaurus/Link'
import { useLocation } from '@docusaurus/router'
import Translate, { translate } from '@docusaurus/Translate'
import BrowseLayout from '@site/src/components/site/BrowseLayout'
import Layout from '@theme/Layout'
import React, { useState } from 'react'

import { RecipeCard, type Recipe } from './RecipeCard'
import styles from './styles.module.css'

interface DecisionModel {
  artifact: string
  card?: string
  questionTypes: string[]
  examples: Array<{ name: string, card: string, tasks: string[] }>
}

interface RecipeCatalog {
  recipes: Recipe[]
  decisionModels: DecisionModel[]
}

function recipeUseCases() {
  return [
    { id: 'all', label: translate({ id: 'recipes.all', message: 'All recipes' }), tags: [] },
    { id: 'general', label: translate({ id: 'recipes.general', message: 'General purpose' }), tags: ['general-purpose', 'mixture-of-models'] },
    { id: 'agents', label: translate({ id: 'recipes.agents', message: 'Coding and agents' }), tags: ['agents'] },
    { id: 'privacy', label: translate({ id: 'recipes.privacy', message: 'Privacy and safety' }), tags: ['privacy', 'security', 'local-first'] },
    { id: 'reasoning', label: translate({ id: 'recipes.reasoning', message: 'Reasoning and verification' }), tags: ['accuracy', 'knowledge', 'orchestration', 'verification'] },
    { id: 'multimodal', label: translate({ id: 'recipes.multimodal', message: 'Vision and multimodal' }), tags: ['multimodal'] },
    { id: 'cost', label: translate({ id: 'recipes.cost', message: 'Cost and latency' }), tags: ['cost', 'latency'] },
    { id: 'feedback', label: translate({ id: 'recipes.feedback', message: 'Feedback and recovery' }), tags: ['feedback', 'recovery'] },
    { id: 'retrieval', label: translate({ id: 'recipes.retrieval', message: 'Retrieval' }), tags: ['retrieval'] },
  ]
}

function RecipeBrowser({ recipes, title }: { recipes: Recipe[], title: string }) {
  const [selected, setSelected] = useState('all')
  const categories = recipeUseCases().map(category => ({
    ...category,
    recipes: category.id === 'all'
      ? recipes
      : recipes.filter(recipe => category.tags.some(tag => recipe.tags.includes(tag))),
  })).filter(category => category.recipes.length > 0)
  const active = categories.find(category => category.id === selected) ?? categories[0]

  return (
    <section className={styles.section} aria-labelledby="by-use-case">
      <h2 id="by-use-case">{title}</h2>
      <p>
        <Translate id="recipes.examplesIntro">
          These maintained examples configure routing, not inference servers. Review each recipe’s requirements and limitations before adapting it.
        </Translate>
      </p>
      <div className={styles.filters} role="group" aria-label={title}>
        {categories.map(category => (
          <button
            key={category.id}
            type="button"
            aria-pressed={category.id === active.id}
            onClick={() => setSelected(category.id)}
          >
            {category.label}
          </button>
        ))}
      </div>
      <p role="status" className={styles.count}>
        <Translate id="recipes.resultCount" values={{ count: active.recipes.length }}>
          {'Matching recipes: {count}'}
        </Translate>
      </p>
      <div className={styles.grid}>
        {active.recipes.map(recipe => <RecipeCard key={recipe.id} recipe={recipe} />)}
      </div>
    </section>
  )
}

function DecisionModels({ models, title }: { models: DecisionModel[], title: string }) {
  return (
    <section className={styles.section} aria-labelledby="decision-models">
      <h2 id="decision-models">{title}</h2>
      <p>
        <Translate id="recipes.modelEvidence">
          These are the decision models and signal questions configured in the examples, not claims that every variant supports those tasks. Check System One for calibrated, zero-shot, and unsupported features, and Model Hub for provider-model lifecycle and benchmark evidence.
        </Translate>
      </p>
      <div className={styles.links}>
        <Link to="/docs/system-one/reference/support-matrix"><Translate id="recipes.supportMatrix">System One support and calibration</Translate></Link>
        <Link to="/docs/system-one/reference/compatibility-matrix"><Translate id="recipes.compatibility">Question compatibility</Translate></Link>
        <Link to="/models"><Translate id="recipes.modelHub">Model Hub</Translate></Link>
      </div>
      <div className={styles.grid}>
        {models.map(model => (
          <article key={model.artifact} className={styles.card}>
            <h3>{model.card ? <Link href={model.card}>{model.artifact}</Link> : model.artifact}</h3>
            {model.questionTypes.length > 0 && (
              <p>
                <Translate id="recipes.questionTypes">Question types:</Translate>
                {' '}
                <code>{model.questionTypes.join(', ')}</code>
              </p>
            )}
            <ul>
              {model.examples.map(example => (
                <li key={example.card}>
                  <Link href={`${example.card}#evaluation`}>{example.name}</Link>
                  {example.tasks.length > 0 && (
                    <>
                      {' · '}
                      {example.tasks.join(', ')}
                    </>
                  )}
                </li>
              ))}
            </ul>
          </article>
        ))}
      </div>
    </section>
  )
}

export default function RecipeDiscoveryPage({ catalog }: { catalog: RecipeCatalog }) {
  const { hash } = useLocation()
  const title = translate({ id: 'recipes.title', message: 'Recipes' })
  const description = translate({
    id: 'recipes.description',
    message: 'Start with the routing behavior you need, then validate a maintained configuration.',
  })
  const browse = translate({ id: 'recipes.byUseCase', message: 'Browse by use case' })
  const modelCatalog = translate({ id: 'recipes.decisionModels', message: 'Decision models' })

  return (
    <Layout title={title} description={description}>
      <BrowseLayout
        title={title}
        description={description}
        eyebrow={<Translate id="recipes.eyebrow">Routing catalog</Translate>}
        sidebarLabel={title}
        activeKey={hash === '#decision-models' ? 'decision-models' : 'by-use-case'}
        groups={[{
          key: 'explore',
          label: title,
          items: [
            { key: 'by-use-case', label: browse, to: '/recipes#by-use-case' },
            { key: 'decision-models', label: modelCatalog, to: '/recipes#decision-models' },
          ],
        }]}
      >
        <RecipeBrowser recipes={catalog.recipes} title={browse} />
        <DecisionModels models={catalog.decisionModels} title={modelCatalog} />
      </BrowseLayout>
    </Layout>
  )
}
