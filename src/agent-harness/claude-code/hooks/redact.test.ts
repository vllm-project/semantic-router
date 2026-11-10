import { expect, test } from 'claude-code/testing'

import { hide, piiIn, secretsIn } from './lib/redact'

// Netlify's secret scan fails the website's deploy on a key-shaped literal anywhere in the repository,
// so the fake provider keys and tokens here are joined from their parts at run time.

const hidden = (text: string) => hide(text, secretsIn(text)).text

test('secret values under JSON and YAML keys are hidden, the keys kept', () => {
  const accessToken = ['ya29', 'a0AfH6SMBx3kQ9'].join('.')
  const clientSecret = ['GOCSPX', 'abcdef123456'].join('-')
  const refreshToken = ['1', '0gAbCdEfGhIjKlMn'].join('//')
  expect(hidden(`{"access_token": "${accessToken}"}`)).toBe('{"access_token": "[vsr:secret]"}')
  expect(hidden(`{"client_secret":"${clientSecret}"}`)).toBe('{"client_secret":"[vsr:secret]"}')
  expect(hidden("{'api_key': 'abcd1234efgh5678'}")).toBe("{'api_key': '[vsr:secret]'}")
  expect(hidden(`client_secret: "${clientSecret}"`)).toBe('client_secret: "[vsr:secret]"')
  expect(hidden(`  refresh_token: ${refreshToken}`)).toBe('  refresh_token: [vsr:secret]')
})

test('an Authorization header loses its credentials', () => {
  const jwt = ['eyJhbGciOiJIUzI1NiJ9', 'eyJzdWIiOiIxIn0', 'abc'].join('.')
  expect(hidden(`Authorization: Bearer ${jwt}`)).toBe('Authorization: Bearer [vsr:secret]')
  expect(hidden("curl -H 'authorization: token 0123456789abcdef' https://x")).toBe("curl -H 'authorization: token [vsr:secret]' https://x")
  expect(hidden('Authorization: Basic dXNlcjpwYXNzd29yZA==')).toBe('Authorization: Basic [vsr:secret]')
})

test('provider tokens are hidden wherever they stand', () => {
  for (const token of [
    ['sk', 'ant', 'api03', 'a1B2'.repeat(10)].join('-'),
    ['sk', 'proj', 'Zx9_'.repeat(10)].join('-'),
    ['github', 'pat', '11ABCDEFG0123456789', 'abcXYZ012'.repeat(6)].join('_'),
    ['xoxb', '1234567890', '1234567890123', 'aBcD'.repeat(6)].join('-'),
    ['xoxp', '1234567890', '1234567890', '1234567890123', 'abcd'.repeat(8)].join('-'),
  ]) {
    expect([token, hidden(`token is ${token} here`)]).toEqual([token, 'token is [vsr:secret] here'])
  }
})

test('a password in a connection URL or a DB_PASS assignment is hidden', () => {
  expect(hidden('DATABASE_URL=postgres://app:hunter2pass@db.local:5432/app')).toBe('DATABASE_URL=postgres://app:[vsr:secret]@db.local:5432/app')
  expect(hidden('redis://default:s3cr3t@cache:6379')).toBe('redis://default:[vsr:secret]@cache:6379')
  expect(hidden('DB_PASS=hunter2pass')).toBe('DB_PASS=[vsr:secret]')
  expect(hidden('db_pass=hunter2pass')).toBe('db_pass=[vsr:secret]')
  expect(hidden('SECRET_KEY=django-insecure-abc123')).toBe('SECRET_KEY=[vsr:secret]')
})

test('a lowercase assignment of a password or key is hidden', () => {
  expect(hidden('db_password=hunter2hunter2')).toBe('db_password=[vsr:secret]')
  expect(hidden('export github_token="abcdef0123456789"')).toBe('export github_token="[vsr:secret]"')
})

test('an INI-style assignment with spaces around = is hidden', () => {
  const apiKey = ['sk', 'test', 'FAKE1234567890abcdef'].join('-')
  expect(hidden(`api_key = ${apiKey}`)).toBe('api_key = [vsr:secret]')
  expect(hidden('[db]\npassword = hunter2FAKE99\nhost = db.local')).toBe('[db]\npassword = [vsr:secret]\nhost = db.local')
})

test('ordinary prose, code and Markdown keep their words', () => {
  for (const text of [
    'Enter your password in the box, then press the API key button.',
    'The token: a short-lived string the server hands out.',
    'password_hint = None',
    'def check(password):\n    return password == stored',
    'if (password.length < 8) return "too short"',
    '## Secret management\n\n- Use `api_key` from the vault, never a literal.',
    '| key | value |\n| --- | --- |\n| password | (see vault) |',
    'Authorization: see the docs',
    'const token = await getToken()',
    'sk-ant is the prefix of an Anthropic key',
    'secret_santa: true',
    'api_key = settings.api_key',
    'token = get_token()',
    'api_key = os.environ["API_KEY"]',
    'SECRET_KEY = os.environ["SECRET_KEY"]',
    'API_KEY = settings.API_KEY',
    'TOKEN = process.env.GITHUB_TOKEN',
    'export GITHUB_TOKEN=$GH_TOKEN',
    'PASSWORD = getpass.getpass()',
    'TOKEN = get_token()',
    'BYPASS_CACHE=truefalse1',
    'git clone git@github.com:org/repo.git',
    'https://example.com/path?a=b',
  ]) {
    expect(hidden(text)).toBe(text)
  }
})

// The model tags the 2.32 of a pinned version as an IP address; hiding it broke the requirements file Claude read.
const ipSpan = (text: string, value: string) => {
  const start = [...text].length - [...text.slice(text.indexOf(value))].length
  return { label: 'IP_ADDRESS', start, end: start + [...value].length, text: value, probability: 0.9 }
}

test('an IP span keeps its tag only when it is shaped like an address', () => {
  for (const [text, value, kept] of [
    ['requests==2.32.3', '2.32', false],
    ['pin numpy 1.26.4 here', '1.26.4', false],
    ['connect to 10.0.0.12 now', '10.0.0.12', true],
    ['listen on 192.168.1.20:8080', '192.168.1.20:8080', true],
    ['the host fe80::1 answered', 'fe80::1', true],
  ] as const) {
    expect(piiIn(text, [ipSpan(text, value)], 0.5).length).toBe(kept ? 1 : 0)
  }
  expect(hide('requests==2.32.3', piiIn('requests==2.32.3', [ipSpan('requests==2.32.3', '2.32')], 0.5)).text).toBe('requests==2.32.3')
})
