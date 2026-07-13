import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { buildSearchTextSegments } from '../src/components/search-panel/searchSecurity.js';
import { resolveHostedDataURLs } from '../src/components/mapview/hostedDataPolicy.js';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const joinedText = segments => segments.map(segment => segment.text).join('');

test('hostile and ordinary markup remains plain search-result text', () => {
  for (const text of [
    '<img src=x onerror="globalThis.pwned=true">',
    '<b>ordinary tag</b>',
    '<ScRiPt>alert(1)</ScRiPt>'
  ]) {
    const segments = buildSearchTextSegments(text, 'tag');
    assert.equal(joinedText(segments), text);
  }

  const template = readFileSync(
    join(root, 'src/components/search-panel/SearchPanel.svelte'),
    'utf8'
  );
  assert.doesNotMatch(template, /\{@html\s+result\./);
  assert.match(template, /\{segment\.text\}/);
});

test('query metacharacters are treated as literal highlight text', () => {
  const text = 'prefix .*[x]( suffix';
  const segments = buildSearchTextSegments(text, '.*[x](');

  assert.equal(joinedText(segments), text);
  assert.deepEqual(
    segments.filter(segment => segment.highlighted),
    [{ text: '.*[x](', highlighted: true }]
  );
});

test('hosted data URLs accept only complete same-origin HTTP(S) pairs', () => {
  const origin = 'https://play.example.test';
  const accepted = resolveHostedDataURLs(
    new URLSearchParams({
      metadataURL: '/api/router/config/kbs/demo/map/metadata',
      dataURL:
        'https://play.example.test/api/router/config/kbs/demo/map/data.ndjson?limit=1'
    }),
    origin
  );
  assert.deepEqual(accepted, {
    requested: true,
    dataURLs: {
      metadata: '/api/router/config/kbs/demo/map/metadata',
      point: '/api/router/config/kbs/demo/map/data.ndjson?limit=1'
    }
  });

  for (const values of [
    {
      metadataURL: 'https://attacker.example/metadata',
      dataURL: '/api/router/config/kbs/demo/map/data.ndjson'
    },
    {
      metadataURL: '/api/router/config/kbs/demo/map/metadata',
      dataURL: '//attacker.example/data.ndjson'
    },
    {
      metadataURL: '/api/router/config/kbs/demo/map/metadata',
      dataURL: 'javascript:alert(1)'
    },
    {
      metadataURL: '/api/router/config/kbs/demo/map/metadata#fragment',
      dataURL: '/api/router/config/kbs/demo/map/data.ndjson'
    },
    {
      dataURL: '/api/router/config/kbs/demo/map/data.ndjson'
    }
  ]) {
    assert.deepEqual(
      resolveHostedDataURLs(new URLSearchParams(values), origin),
      { requested: true, dataURLs: null }
    );
  }
});

test('hosted data fetches fail closed on redirects', () => {
  const source = readFileSync(
    join(root, 'src/components/embedding/hostedKnowledgeMap.ts'),
    'utf8'
  );
  assert.equal(source.match(/redirect: 'error'/g)?.length, 2);
  assert.equal(source.match(/credentials: 'same-origin'/g)?.length, 2);
});
