#!/usr/bin/env node

import { readFileSync, writeFileSync } from 'node:fs'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'
import {
  discoverConfigurationCatalog,
  renderConfigurationCatalog,
  replaceGeneratedCatalog,
} from './lib/configuration-catalog.mjs'

const args = process.argv.slice(2)
const allowedArgs = new Set(['--check'])
const unknownArgs = args.filter(argument => !allowedArgs.has(argument))
if (unknownArgs.length > 0 || args.length > 1) {
  console.error('Usage: node scripts/generate-configuration-catalog.mjs [--check]')
  process.exit(2)
}

const scriptDirectory = dirname(fileURLToPath(import.meta.url))
const repoRoot = resolve(scriptDirectory, '..', '..')
const configurationPage = resolve(
  repoRoot,
  'website',
  'docs',
  'installation',
  'configuration.md',
)
const current = readFileSync(configurationPage, 'utf8')
const catalog = discoverConfigurationCatalog(repoRoot)
const generated = renderConfigurationCatalog(catalog)
const next = replaceGeneratedCatalog(current, generated, configurationPage)

if (args.includes('--check')) {
  if (next !== current) {
    console.error(
      'Configuration catalog is out of date. Run `npm run config:generate` in website/.',
    )
    process.exit(1)
  }

  console.log('Configuration catalog is up to date.')
  process.exit(0)
}

if (next === current) {
  console.log('Configuration catalog is already up to date.')
}
else {
  writeFileSync(configurationPage, next)
  console.log('Updated website/docs/installation/configuration.md.')
}
