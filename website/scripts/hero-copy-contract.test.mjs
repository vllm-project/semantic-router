import assert from 'node:assert/strict'
import { readFile } from 'node:fs/promises'
import test from 'node:test'

const heroSourceURL = new URL(
  '../src/components/site/SemanticTerrainHero/index.tsx',
  import.meta.url,
)
const chineseMessagesURL = new URL('../i18n/zh-Hans/code.json', import.meta.url)

function translatedDefault(source, id) {
  const opening = `<Translate id="${id}">`
  const start = source.indexOf(opening)
  assert.notEqual(start, -1, `missing source translation id ${id}`)
  const contentStart = start + opening.length
  const end = source.indexOf('</Translate>', contentStart)
  assert.notEqual(end, -1, `missing closing translation tag for ${id}`)
  return source.slice(contentStart, end).replace(/\s+/g, ' ').trim()
}

test('Chinese homepage positioning follows the current English message IDs', async () => {
  const [source, chineseMessages] = await Promise.all([
    readFile(heroSourceURL, 'utf8'),
    readFile(chineseMessagesURL, 'utf8').then(JSON.parse),
  ])
  const contract = {
    'homepage.hero.systemLabel': {
      english: 'The next-generation model architecture',
      chinese: '下一代模型架构',
    },
    'homepage.hero.description': {
      english: 'System-level intelligence for heterogeneous LLM inference',
      chinese: '面向异构 LLM 推理的系统级智能',
    },
  }

  for (const [id, expected] of Object.entries(contract)) {
    assert.equal(translatedDefault(source, id), expected.english)
    assert.equal(chineseMessages[id]?.message, expected.chinese)
  }
})
