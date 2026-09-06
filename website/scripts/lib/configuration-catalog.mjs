import {
  existsSync,
  readFileSync,
  readdirSync,
} from 'node:fs'
import { basename, extname, join, relative, sep } from 'node:path'

export const GENERATED_CATALOG_START = '<!-- BEGIN GENERATED CONFIGURATION CATALOG -->'
export const GENERATED_CATALOG_END = '<!-- END GENERATED CONFIGURATION CATALOG -->'

const githubSourceRoot = 'https://github.com/vllm-project/semantic-router'
const sentenceSegmenter = new Intl.Segmenter('en', { granularity: 'sentence' })
const yamlExtensions = new Set(['.yaml', '.yml'])

function asPosix(path) {
  return path.split(sep).join('/')
}

function sortedEntries(directory) {
  return readdirSync(directory, { withFileTypes: true })
    .sort((left, right) => left.name.localeCompare(right.name))
}

function markdownFiles(directory) {
  if (!existsSync(directory)) {
    return []
  }

  return sortedEntries(directory).flatMap((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      return markdownFiles(path)
    }

    return entry.isFile() && extname(entry.name) === '.md' ? [path] : []
  })
}

function yamlFiles(directory) {
  if (!existsSync(directory)) {
    return []
  }

  return sortedEntries(directory).flatMap((entry) => {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      return yamlFiles(path)
    }

    return entry.isFile() && yamlExtensions.has(extname(entry.name))
      ? [path]
      : []
  })
}

function requireYamlFamily(directory, label) {
  if (yamlFiles(directory).length === 0) {
    throw new Error(`${label} has no YAML fragments: ${directory}`)
  }
}

function familyDirectories(fragmentRoot, label) {
  const entries = sortedEntries(fragmentRoot)
  const looseYaml = entries
    .filter(entry => entry.isFile() && yamlExtensions.has(extname(entry.name)))
    .map(entry => entry.name)
  if (looseYaml.length > 0) {
    throw new Error(
      `${label} fragments must be grouped into family directories; found ${looseYaml.join(', ')}`,
    )
  }

  return entries.filter(entry => entry.isDirectory())
}

export function findSingleTutorial(tutorialRoot, family) {
  const filename = `${family}.md`
  const matches = markdownFiles(tutorialRoot)
    .filter(path => basename(path) === filename)

  if (matches.length === 0) {
    throw new Error(`No tutorial named ${filename} under ${tutorialRoot}`)
  }

  if (matches.length > 1) {
    const paths = matches.map(path => `  - ${path}`).join('\n')
    throw new Error(`Ambiguous tutorial for ${family}:\n${paths}`)
  }

  return matches[0]
}

export function extractOverviewGoal(markdown, sourceLabel = 'tutorial') {
  const headings = [...markdown.matchAll(/^##[ \t]+Overview[ \t]*$/gm)]
  if (headings.length !== 1) {
    throw new Error(
      `${sourceLabel} must contain exactly one "## Overview" heading; found ${headings.length}`,
    )
  }

  const sectionStart = headings[0].index + headings[0][0].length
  const afterHeading = markdown.slice(sectionStart)
  const nextHeading = afterHeading.search(/^##[ \t]+/m)
  const overview = nextHeading === -1
    ? afterHeading
    : afterHeading.slice(0, nextHeading)
  const paragraph = overview.trim().split(/\n[ \t]*\n/, 1)[0]
    .replace(/[ \t]*\n[ \t]*/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()

  if (!paragraph || /^(?:#|[-*+] |```|:::)/.test(paragraph)) {
    throw new Error(`${sourceLabel} needs a prose paragraph immediately after "## Overview"`)
  }

  const firstSentence = [...sentenceSegmenter.segment(paragraph)][0]?.segment.trim()
  if (!firstSentence) {
    throw new Error(`${sourceLabel} has an empty Overview`)
  }

  return firstSentence
}

function tutorialGoal(path) {
  return extractOverviewGoal(readFileSync(path, 'utf8'), path)
}

function fragmentLink(path, kind) {
  const target = kind === 'directory' ? 'tree' : 'blob'
  return `${githubSourceRoot}/${target}/main/${path}`
}

function guideLink(repoRoot, path) {
  const docsRoot = join(repoRoot, 'website', 'docs')
  const docPath = asPosix(relative(docsRoot, path)).replace(/\.md$/, '')
  return `../${docPath}`
}

function signalEntries(repoRoot) {
  const fragmentRoot = join(repoRoot, 'config', 'fragments', 'signal')
  const tutorialRoot = join(repoRoot, 'website', 'docs', 'tutorials', 'signal')

  return familyDirectories(fragmentRoot, 'Signal')
    .map((entry) => {
      const family = entry.name
      const fragmentDirectory = join(fragmentRoot, family)
      requireYamlFamily(fragmentDirectory, `Signal family ${family}`)
      const tutorial = findSingleTutorial(tutorialRoot, family)
      const tutorialParts = asPosix(relative(tutorialRoot, tutorial)).split('/')
      const tutorialType = tutorialParts[0]
      if (tutorialParts.length !== 2 || !['heuristic', 'learned'].includes(tutorialType)) {
        throw new Error(
          `Signal tutorial ${tutorial} must live directly under signal/heuristic or signal/learned`,
        )
      }

      return {
        family,
        type: `${tutorialType} signal`,
        goal: tutorialGoal(tutorial),
        fragmentPath: `config/fragments/signal/${family}/`,
        fragmentKind: 'directory',
        guide: guideLink(repoRoot, tutorial),
      }
    })
}

function algorithmEntries(repoRoot, algorithmType) {
  const fragmentRoot = join(repoRoot, 'config', 'fragments', 'algorithm', algorithmType)
  const tutorialRoot = join(
    repoRoot,
    'website',
    'docs',
    'tutorials',
    'algorithm',
    algorithmType,
  )
  const familyFiles = yamlFiles(fragmentRoot)
  const nestedFiles = familyFiles.filter(path => asPosix(relative(fragmentRoot, path)).includes('/'))
  if (nestedFiles.length > 0) {
    throw new Error(
      `${algorithmType} algorithm fragments must be direct YAML files under ${fragmentRoot}: ${nestedFiles.join(', ')}`,
    )
  }

  const families = familyFiles.map(path => basename(path, extname(path)))
  if (new Set(families).size !== families.length) {
    throw new Error(`${algorithmType} algorithm fragment family names must be unique`)
  }

  return familyFiles.map((fragment) => {
    const filename = basename(fragment)
    const family = basename(filename, extname(filename))
    const tutorial = findSingleTutorial(tutorialRoot, family)

    return {
      family,
      type: `${algorithmType} algorithm`,
      goal: tutorialGoal(tutorial),
      fragmentPath: `config/fragments/algorithm/${algorithmType}/${filename}`,
      fragmentKind: 'file',
      guide: guideLink(repoRoot, tutorial),
    }
  })
}

function pluginEntries(repoRoot) {
  const fragmentRoot = join(repoRoot, 'config', 'fragments', 'plugin')
  const tutorialRoot = join(repoRoot, 'website', 'docs', 'tutorials', 'plugin')

  return familyDirectories(fragmentRoot, 'Plugin')
    .map((entry) => {
      const family = entry.name
      const fragmentDirectory = join(fragmentRoot, family)
      requireYamlFamily(fragmentDirectory, `Plugin family ${family}`)
      const tutorial = findSingleTutorial(tutorialRoot, family)

      return {
        family,
        type: family === 'content-safety' ? 'plugin bundle' : 'route plugin',
        goal: tutorialGoal(tutorial),
        fragmentPath: `config/fragments/plugin/${family}/`,
        fragmentKind: 'directory',
        guide: guideLink(repoRoot, tutorial),
      }
    })
}

export function discoverConfigurationCatalog(repoRoot) {
  return {
    signals: signalEntries(repoRoot),
    selectionAlgorithms: algorithmEntries(repoRoot, 'selection'),
    looperAlgorithms: algorithmEntries(repoRoot, 'looper'),
    plugins: pluginEntries(repoRoot),
  }
}

function escapeTableCell(value) {
  return value.replaceAll('|', '\\|').replaceAll('\n', ' ')
}

function catalogTable(entries) {
  const rows = entries.map((entry) => {
    const family = `\`${entry.family}\` — ${entry.type}`
    const source = `[\`${entry.fragmentPath}\`](${fragmentLink(entry.fragmentPath, entry.fragmentKind)})`
    const guide = `[Guide](${entry.guide})`
    return `| ${family} | ${escapeTableCell(entry.goal)} | ${source} | ${guide} |`
  })

  return [
    '| Family and type | Use it to | Reusable fragment | Guide |',
    '| --- | --- | --- | --- |',
    ...rows,
  ].join('\n')
}

export function renderConfigurationCatalog(catalog) {
  return [
    GENERATED_CATALOG_START,
    '<!-- Generated by website/scripts/generate-configuration-catalog.mjs. Do not edit this block by hand. -->',
    '',
    '### Signals',
    '',
    catalogTable(catalog.signals),
    '',
    '### Selection algorithms',
    '',
    catalogTable(catalog.selectionAlgorithms),
    '',
    '### Looper algorithms',
    '',
    catalogTable(catalog.looperAlgorithms),
    '',
    '### Plugins and bundles',
    '',
    catalogTable(catalog.plugins),
    '',
    GENERATED_CATALOG_END,
  ].join('\n')
}

export function replaceGeneratedCatalog(page, generatedCatalog, sourceLabel = 'configuration page') {
  const starts = page.split(GENERATED_CATALOG_START).length - 1
  const ends = page.split(GENERATED_CATALOG_END).length - 1
  if (starts !== 1 || ends !== 1) {
    throw new Error(
      `${sourceLabel} must contain one generated catalog marker pair; found ${starts} start and ${ends} end markers`,
    )
  }

  const start = page.indexOf(GENERATED_CATALOG_START)
  const end = page.indexOf(GENERATED_CATALOG_END, start)
  if (end < start) {
    throw new Error(`${sourceLabel} has generated catalog markers in the wrong order`)
  }

  const suffixStart = end + GENERATED_CATALOG_END.length
  return `${page.slice(0, start)}${generatedCatalog}${page.slice(suffixStart)}`
}
