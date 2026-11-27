import Link from '@docusaurus/Link'
import Translate from '@docusaurus/Translate'
import CodeBlock from '@theme/CodeBlock'
import React, { useState } from 'react'

import styles from './styles.module.css'

interface ModelDeployment {
  name: string
  artifact?: string
  provider?: string
  device?: string
}

interface ProviderModel {
  name: string
  model: string
  protocol?: string
  lifecycle?: string
}

export interface Recipe {
  id: string
  name: string
  description: string
  tags: string[]
  card: string
  source: string
  configPath: string
  servePlatform?: string
  yaml: string
  entrypoints: string[]
  providers: ProviderModel[]
  decision?: ModelDeployment
  embedding?: ModelDeployment
  limitations: string[]
}

function Deployment({ model }: { model: ModelDeployment }) {
  return (
    <>
      <code>{model.artifact ?? model.name}</code>
      {model.provider && (
        <>
          {' · '}
          <code>{model.provider}</code>
        </>
      )}
      {model.device && (
        <>
          {' · '}
          <code>{model.device}</code>
        </>
      )}
    </>
  )
}

function Provider({ provider }: { provider: ProviderModel }) {
  const { name, model, protocol, lifecycle } = provider
  return (
    <>
      {name !== model && (
        <>
          <code>{name}</code>
          {' · '}
        </>
      )}
      {lifecycle
        ? <Link to={`/models?model=${encodeURIComponent(model)}#models`}><code>{model}</code></Link>
        : <code>{model}</code>}
      {protocol && (
        <>
          {' · '}
          <code>{protocol}</code>
        </>
      )}
      {lifecycle === 'experimental' && (
        <>
          {' · '}
          <Translate id="recipes.experimental">Experimental</Translate>
        </>
      )}
    </>
  )
}

function Configuration({ recipe }: { recipe: Recipe }) {
  const [open, setOpen] = useState(false)
  const platform = recipe.servePlatform ? ` --platform ${recipe.servePlatform}` : ''
  const commands = [
    `vllm-sr config validate --config ${recipe.configPath}`,
    `vllm-sr serve${platform} --config ${recipe.configPath}`,
  ].join('\n')

  return (
    <details onToggle={event => setOpen(event.currentTarget.open)}>
      <summary><Translate id="recipes.configuration">Configuration and commands</Translate></summary>
      {open && (
        <div className={styles.details}>
          <p>
            <Translate id="recipes.runFromRepo">
              Follow the Model Card’s backend and environment setup. Run these commands from the repository root.
            </Translate>
          </p>
          <p><Link href={`${recipe.card}#quick-start`}><Translate id="recipes.quickStart">Recipe quick start</Translate></Link></p>
          <CodeBlock language="bash">{commands}</CodeBlock>
          <div className={styles.yaml}>
            <CodeBlock language="yaml" title={recipe.configPath}>{recipe.yaml}</CodeBlock>
          </div>
        </div>
      )}
    </details>
  )
}

export function RecipeCard({ recipe }: { recipe: Recipe }) {
  return (
    <article className={styles.card} id={recipe.id}>
      <h3>{recipe.name}</h3>
      <p>{recipe.description}</p>
      <dl className={styles.facts}>
        <dt><Translate id="recipes.entrypoints">Entrypoints</Translate></dt>
        <dd><code>{recipe.entrypoints.join(', ')}</code></dd>
        {recipe.decision && (
          <>
            <dt><Translate id="recipes.decisionModel">Decision model</Translate></dt>
            <dd><Deployment model={recipe.decision} /></dd>
          </>
        )}
        {recipe.embedding && (
          <>
            <dt><Translate id="recipes.embeddingModel">Embedding model</Translate></dt>
            <dd><Deployment model={recipe.embedding} /></dd>
          </>
        )}
      </dl>
      <div className={styles.links}>
        <Link href={`${recipe.card}#routing-behavior`}><Translate id="recipes.behavior">Routing behavior</Translate></Link>
        <Link href={`${recipe.card}#evaluation`}><Translate id="recipes.evaluation">Evaluation</Translate></Link>
        <Link href={recipe.source}><Translate id="recipes.source">Source files</Translate></Link>
      </div>
      <details>
        <summary><Translate id="recipes.requirementsAndLimits">Requirements and limitations</Translate></summary>
        <div className={styles.details}>
          <p><Translate id="recipes.backends">Provider models in the example:</Translate></p>
          <ul>{recipe.providers.map(provider => <li key={provider.name}><Provider provider={provider} /></li>)}</ul>
          <ul>{recipe.limitations.map((limit, index) => <li key={index}>{limit}</li>)}</ul>
          <Link href={`${recipe.card}#requirements`}><Translate id="recipes.fullRequirements">Full requirements in the Model Card</Translate></Link>
        </div>
      </details>
      <Configuration recipe={recipe} />
    </article>
  )
}
