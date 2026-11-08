import { expect, test } from 'claude-code/testing'

import { hide, secretsIn } from './lib/redact'

const hidden = (text: string) => hide(text, secretsIn(text)).text

test('secret values under JSON and YAML keys are hidden, the keys kept', () => {
  expect(hidden('{"access_token": "ya29.a0AfH6SMBx3kQ9"}')).toBe('{"access_token": "[secret]"}')
  expect(hidden('{"client_secret":"GOCSPX-abcdef123456"}')).toBe('{"client_secret":"[secret]"}')
  expect(hidden("{'api_key': 'abcd1234efgh5678'}")).toBe("{'api_key': '[secret]'}")
  expect(hidden('client_secret: "GOCSPX-abcdef123456"')).toBe('client_secret: "[secret]"')
  expect(hidden('  refresh_token: 1//0gAbCdEfGhIjKlMn')).toBe('  refresh_token: [secret]')
})

test('an Authorization header loses its credentials', () => {
  expect(hidden('Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abc')).toBe('Authorization: Bearer [secret]')
  expect(hidden("curl -H 'authorization: token 0123456789abcdef' https://x")).toBe("curl -H 'authorization: token [secret]' https://x")
  expect(hidden('Authorization: Basic dXNlcjpwYXNzd29yZA==')).toBe('Authorization: Basic [secret]')
})

test('provider tokens are hidden wherever they stand', () => {
  for (const token of [
    `sk-ant-api03-${'a1B2'.repeat(10)}`,
    `sk-proj-${'Zx9_'.repeat(10)}`,
    `github_pat_11ABCDEFG0123456789_${'abcXYZ012'.repeat(6)}`,
    `xoxb-1234567890-1234567890123-${'aBcD'.repeat(6)}`,
    `xoxp-1234567890-1234567890-1234567890123-${'abcd'.repeat(8)}`,
  ]) {
    expect([token, hidden(`token is ${token} here`)]).toEqual([token, 'token is [secret] here'])
  }
})

test('a lowercase assignment of a password or key is hidden', () => {
  expect(hidden('db_password=hunter2hunter2')).toBe('db_password=[secret]')
  expect(hidden('export github_token="abcdef0123456789"')).toBe('export github_token="[secret]"')
})

test('an INI-style assignment with spaces around = is hidden', () => {
  expect(hidden('api_key = sk-test-FAKE1234567890abcdef')).toBe('api_key = [secret]')
  expect(hidden('[db]\npassword = hunter2FAKE99\nhost = db.local')).toBe('[db]\npassword = [secret]\nhost = db.local')
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
  ]) {
    expect(hidden(text)).toBe(text)
  }
})
