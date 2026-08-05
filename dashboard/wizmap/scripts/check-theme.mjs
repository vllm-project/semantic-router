import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));

const read = relativePath => readFileSync(join(root, relativePath), 'utf8');
const withoutComments = source =>
  source.replace(/\/\*[\s\S]*?\*\//g, '').replace(/\/\/.*$/gm, '');

const globalTheme = read('public/global.css');
const componentSources = new Map([
  ['map view', read('src/components/mapview/MapView.scss')],
  ['search panel', read('src/components/search-panel/SearchPanel.scss')],
  ['embedding controls', read('src/components/embedding/Embedding.scss')],
  ['footer controls', read('src/components/footer/Footer.scss')]
]);
const failures = [];

const requireText = (source, expected, label) => {
  if (!source.includes(expected)) {
    failures.push(`${label} must include ${expected}`);
  }
};

for (const token of [
  '--wiz-canvas: #050505',
  '--wiz-surface: #101012',
  '--wiz-surface-raised: #18181b',
  '--wiz-text-primary: #f5f5f7',
  '--wiz-text-secondary: #b2b2b8',
  '--wiz-brand: #e31b23'
]) {
  requireText(globalTheme, token, 'global theme contract');
}

const requiredComponentTokens = new Map([
  ['map view', ['var(--wiz-canvas)', 'var(--wiz-surface-glass)']],
  [
    'search panel',
    [
      'var(--wiz-surface-glass)',
      'var(--wiz-text-primary)',
      'var(--wiz-brand-bright)'
    ]
  ],
  [
    'embedding controls',
    [
      'var(--wiz-surface-glass)',
      'var(--wiz-brand-soft)',
      'var(--wiz-border-strong)'
    ]
  ],
  ['footer controls', ['var(--wiz-surface-glass)', 'var(--wiz-text-secondary)']]
]);

const forbiddenChromePatterns = [
  [
    'literal white surface',
    /background(?:-color)?\s*:\s*(?:white|#fff(?:fff)?)\s*;/i
  ],
  ['legacy blue-black border', /rgba\(20\s*,\s*30\s*,\s*58/i],
  ['legacy blue-black shadow', /rgba\(15\s*,\s*23\s*,\s*42/i],
  ['legacy blue interaction border', /rgba\(30\s*,\s*136\s*,\s*229/i],
  ['debug red focus surface', /background(?:-color)?\s*:\s*red\s*;/i]
];

for (const [name, rawSource] of componentSources) {
  const source = withoutComments(rawSource);

  for (const token of requiredComponentTokens.get(name) ?? []) {
    requireText(source, token, name);
  }

  for (const [label, pattern] of forbiddenChromePatterns) {
    if (pattern.test(source)) {
      failures.push(`${name} contains ${label}`);
    }
  }

  if (name !== 'embedding controls' && /\$(?:blue|purple)-/.test(source)) {
    failures.push(
      `${name} still uses the legacy blue/purple palette for UI chrome`
    );
  }
}

const embeddingSource = withoutComments(
  componentSources.get('embedding controls')
);
const nonCategoricalPaletteLines = embeddingSource
  .split('\n')
  .filter(
    line =>
      /\$(?:blue|purple)-/.test(line) && !/accent-color:\s*\$blue-/.test(line)
  );

if (nonCategoricalPaletteLines.length > 0) {
  failures.push(
    'embedding controls use blue/purple outside categorical group accents'
  );
}

// The theme migration must not flatten the categorical colors used by map points and clusters.
const configSource = read('src/config/config.ts');
requireText(
  configSource,
  'defaultPointColorInt: [48, 65, 159]',
  'point color contract'
);
requireText(
  configSource,
  'secondPointColorInt: [194, 24, 92]',
  'point color contract'
);

const packingSource = read('src/components/packing/Packing.scss');
for (const clusterColor of ['$teal-200', '$teal-600', '$teal-900']) {
  requireText(packingSource, `fill: ${clusterColor}`, 'cluster color contract');
}

if (failures.length > 0) {
  throw new Error(`WizMap theme contract failed:\n- ${failures.join('\n- ')}`);
}

console.log('WizMap graphite theme contract passed.');
