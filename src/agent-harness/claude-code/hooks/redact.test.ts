import { expect, test } from 'claude-code/testing'

import { hide, type Kind, kindOf, kindOfCommand, piiIn, secretsIn } from './lib/redact'

// Netlify's secret scan fails the website's deploy on a key-shaped literal anywhere in the repository,
// so the fake provider keys and tokens here are joined from their parts at run time.

const hidden = (text: string, kind?: Kind) => hide(text, secretsIn(text, kind)).text

// A fake value with no shape of its own, and what every hidden value reads as.
const VALUE = 'abcdef1234567890'
const TAG = '[vsr:secret]'

// Each text beside what Claude reads of it, so a failure names the text.
function expectEach(cases: readonly (readonly [string, string])[], kind?: Kind) {
  for (const [text, want] of cases) expect([text, hidden(text, kind)]).toEqual([text, want])
}

const expectKept = (texts: readonly string[], kind?: Kind) => expectEach(texts.map(text => [text, text] as const), kind)

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

test('a quoted secret is hidden whatever characters it holds', () => {
  expectEach([
    ['DB_PASSWORD="Sup3rS3cr3t$Pass"', `DB_PASSWORD="${TAG}"`],
    ["DB_PASSWORD='p&ssw0rd-long-one'", `DB_PASSWORD='${TAG}'`],
    ['DB_PASSWORD="Sup3r(S3cr3t)Pass"', `DB_PASSWORD="${TAG}"`],
    ['DB_PASSWORD="Sup3r,S3cr3t?Pass"', `DB_PASSWORD="${TAG}"`],
    ['ADMIN_PASSWORD="correct horse battery staple"', `ADMIN_PASSWORD="${TAG}"`],
    ["SECRET_KEY = 'django-insecure-9$k(2&x!q@w#e)r^t*y_u+i=o-p0a1s2d3f4g5h6j7'", `SECRET_KEY = '${TAG}'`],
    ['db_password = "Qx7$Lm9 Zt4w"', `db_password = "${TAG}"`],
    ['"password": "it\'s-a-long-secret-99"', `"password": "${TAG}"`],
    ['password: "correct horse battery"', `password: "${TAG}"`],
    ['PASSWORD="hunter2"', `PASSWORD="${TAG}"`],
  ])
})

test('a secret is hidden wherever its key stands on the line', () => {
  expectEach([
    [`      - DB_PASSWORD=${VALUE}`, `      - DB_PASSWORD=${TAG}`],
    [`ENV SECRET_KEY=${VALUE}`, `ENV SECRET_KEY=${TAG}`],
    [`ARG NPM_TOKEN=${VALUE}`, `ARG NPM_TOKEN=${TAG}`],
    [`docker run -e API_KEY=${VALUE} img`, `docker run -e API_KEY=${TAG} img`],
    [`DEBUG=1 API_KEY=${VALUE} ./run.sh`, `DEBUG=1 API_KEY=${TAG} ./run.sh`],
    [`"Env": ["DB_PASSWORD=${VALUE}", "PATH=/bin"]`, `"Env": ["DB_PASSWORD=${TAG}", "PATH=/bin"]`],
    [`Server=db;User Id=sa;Password=${VALUE};`, `Server=db;User Id=sa;Password=${TAG};`],
    [`connect(port=5432, password="${VALUE}")`, `connect(port=5432, password="${TAG}")`],
    [`{ user: 'root', password: '${VALUE}' }`, `{ user: 'root', password: '${TAG}' }`],
    [`kubectl create secret generic db --from-literal=password=${VALUE}`, `kubectl create secret generic db --from-literal=password=${TAG}`],
  ])
})

test('a secret is hidden in every way a language assigns one', () => {
  expectEach([
    [`password: str = "${VALUE}"`, `password: str = "${TAG}"`],
    [`token := "${VALUE}"`, `token := "${TAG}"`],
    [`'password' => '${VALUE}',`, `'password' => '${TAG}',`],
    [`:password => "${VALUE}"`, `:password => "${TAG}"`],
    [`$password = '${VALUE}';`, `$password = '${TAG}';`],
    [`const apiKey = "${VALUE}";`, `const apiKey = "${TAG}";`],
    [`"clientSecret": "${VALUE}"`, `"clientSecret": "${TAG}"`],
    [`define('DB_PASSWORD', '${VALUE}');`, `define('DB_PASSWORD', '${TAG}');`],
    [`os.environ["API_KEY"] = "${VALUE}"`, `os.environ["API_KEY"] = "${TAG}"`],
    [`headers.set("X-Api-Key", "${VALUE}")`, `headers.set("X-Api-Key", "${TAG}")`],
    [`PGPASSWORD=${VALUE} psql`, `PGPASSWORD=${TAG} psql`],
    [`DB_PWD=${VALUE}`, `DB_PWD=${TAG}`],
  ])
})

test('more tokens with a shape of their own are hidden wherever they stand', () => {
  const jwt = ['eyJhbGciOiJIUzI1NiJ9', 'eyJzdWIiOiIxMjM0NTY3ODkwIn0', 'SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV'].join('.')
  for (const token of [
    ['hf', 'AbCdEfGhIjKlMnOpQrStUvWxYzAbCdEfGh'].join('_'),
    ['glpat', 'abcdefghij1234567890'].join('-'),
    ['npm', 'abcdefghijklmnopqrstuvwxyz0123456789'].join('_'),
    ['pypi', 'AgEIcHlwaS5vcmcCJGFiY2RlZg'].join('-'),
    ['AIza', 'SyA1234567890abcdefghijklmnopqrstuv'].join(''),
    ['ASIA', 'IOSFODNN7EXAMPLE'].join(''),
    ['sk', 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV'].join('-'),
    ['dckr', 'pat', 'abcdefghijklmnopqrstuvwxyz0'].join('_'),
    ['SG', 'abcdefghijklmnopqrstuv', 'abcdefghijklmnopqrstuvwxyz0123456789abcdefg'].join('.'),
    jwt,
  ]) {
    expect([token, hidden(`token is ${token} here`)]).toEqual([token, `token is ${TAG} here`])
  }
  // The supply chain scan refuses a webhook's address written out, so it is joined here too.
  const services = `https://${['hooks', 'slack', 'com'].join('.')}/services`
  const webhook = ['T00000000', 'B00000000', 'abcdefghijklmnopqrstuvwx'].join('/')
  expect(hidden(`${services}/${webhook}`)).toBe(`${services}/${TAG}`)
  expect(hidden('-----BEGIN PGP PRIVATE KEY BLOCK-----\nlQOYBF\n-----END PGP PRIVATE KEY BLOCK-----')).toBe(TAG)
})

test('a password hash is hidden as shadow and htpasswd files keep it', () => {
  const sha512 = ['', '6', 'saltsalt', 'abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH'].join('$')
  const apr1 = ['', 'apr1', 'saltsalt', 'abcdefghijklmnopqrstuv'].join('$')
  expect(hidden(`root:${sha512}:19000:0:99999:7:::`)).toBe(`root:${TAG}:19000:0:99999:7:::`)
  expect(hidden(`admin:${apr1}`)).toBe(`admin:${TAG}`)
})

test('a password on a command line, in a URL query, in XML, a .netrc or a cookie is hidden', () => {
  expectEach([
    [`psql --password ${VALUE}`, `psql --password ${TAG}`],
    [`gh auth login --token=${VALUE}`, `gh auth login --token=${TAG}`],
    [`vault login --token ${VALUE}`, `vault login --token ${TAG}`],
    [`sshpass -p ${VALUE} ssh host`, `sshpass -p ${TAG} ssh host`],
    [`mysql -u root -p${VALUE} db`, `mysql -u root -p${TAG} db`],
    [`docker login -u me -p ${VALUE} registry.example`, `docker login -u me -p ${TAG} registry.example`],
    [`curl -u admin:${VALUE} https://x`, `curl -u admin:${TAG} https://x`],
    [`jdbc:mysql://db/app?user=root&password=${VALUE}`, `jdbc:mysql://db/app?user=root&password=${TAG}`],
    [`https://api.example.com/v1?api_key=${VALUE}&q=1`, `https://api.example.com/v1?api_key=${TAG}&q=1`],
    [`<password>${VALUE}</password>`, `<password>${TAG}</password>`],
    [`<add key="ApiKey" value="${VALUE}" />`, `<add key="ApiKey" value="${TAG}" />`],
    [`machine github.com login me password ${VALUE}`, `machine github.com login me password ${TAG}`],
    [`machine github.com\n  login me\n  password ${VALUE}`, `machine github.com\n  login me\n  password ${TAG}`],
    [`Cookie: session=${VALUE}; theme=dark`, `Cookie: ${TAG}`],
    [`X-Auth-Token: ${VALUE}abcdef`, `X-Auth-Token: ${TAG}`],
  ])
})

test('the credentials of .npmrc, a Docker config and a kubeconfig are hidden', () => {
  expectEach([
    [`//registry.npmjs.org/:_authToken=${VALUE}${VALUE}`, `//registry.npmjs.org/:_authToken=${TAG}`],
    ['_auth=dXNlcjpwYXNzd29yZDEyMzQ1', `_auth=${TAG}`],
    ['"auth": "dXNlcjpwYXNzd29yZDEyMzQ1"', `"auth": "${TAG}"`],
    ['    client-key-data: LS0tLS1CRUdJTiBSU0EgUFJJVkFURSBLRVktLS0tLQ==', `    client-key-data: ${TAG}`],
    ['OPENAI_KEY=a1b2c3d4e5f6g7h8i9j0k1l2', `OPENAI_KEY=${TAG}`],
    ['GH_PAT=abcdef1234567890abcdef', `GH_PAT=${TAG}`],
  ])
  expectKept(['    certificate-authority-data: LS0tLS1CRUdJTiBDRVJUSUZJQ0FURS0tLS0t', 'always-auth=true', 'auth = "oauth2"', 'CACHE_KEY=user:profile:v1abc123'])
})

test('in a config file an unquoted value under a credential key is hidden whole', () => {
  expectEach(
    [
      ['password: SuperSecretPassword', `password: ${TAG}`],
      ['password: p@ss(w0rd)long', `password: ${TAG}`],
      ['password: correct horse battery  # prod', `password: ${TAG}  # prod`],
      ['db_password = hunterhunter', `db_password = ${TAG}`],
      ['spring.datasource.password=my.pass.word1', `spring.datasource.password=${TAG}`],
      ['secret_id: 4b5c6d7e-aaaa-bbbb-cccc-1234567890ab', `secret_id: ${TAG}`],
    ],
    'config',
  )
  expectKept(
    [
      'token: string',
      'password: required',
      'auth: bearer',
      'existingSecret: my-app-secret',
      'secretName: my-app-tls',
      'token: ${{ secrets.GITHUB_TOKEN }}',
      'password: {{ .Values.db.password }}',
      'api_key: "${GEMINI_API_KEY}"',
      'api_key_env: OPENAI_API_KEY',
      'password: !vault |',
      'credential_source = Ec2InstanceMetadata',
      'PWD=/home/u/project',
      'max_tokens: 4096',
    ],
    'config',
  )
})

test('in source code an unquoted value is an expression, and stays', () => {
  expectKept(
    ['  token: Uint8Array', '  privateKey: Ed25519Key', 'access_token = oauth2_scheme', 'MAX_TOKENS = DEFAULT_MAX_TOKENS', 'TOKEN_COUNT = count_tokens(text, model)', '    api_key=config.openai_key', '    token=self.token'],
    'code',
  )
  expect(hidden(`const apiKey = "${VALUE}"`, 'code')).toBe(`const apiKey = "${TAG}"`)
})

test('a key that describes a credential is not one', () => {
  expectKept([
    'OAUTH_TOKEN_URL=https://auth.example.com/oauth/token',
    'API_KEY_FILE=/run/secrets/api_key',
    'TOKENIZER_PATH=/models/tokenizer.json',
    'TOKENIZERS_PARALLELISM=false_value',
    'MAX_TOKENS = DEFAULT_MAX_TOKENS',
    'TOKEN_BUDGET = 1_000_000',
    'TOKEN_EXPIRY = timedelta(days=7, hours=1)',
    'token_url: https://auth0.example.com/oauth/token',
    'password_file: /run/secrets/db1',
    '  --password-file=/etc/secrets/pw',
    '  --secret-name=my-app-tls',
    'secret_name: app-tls-v2',
    '"token_endpoint": "https://accounts.example.com/o/oauth2/token",',
    'token_type_hint="access_token"',
    'API_KEY_NAME = "X-API-Key"',
    'TOKEN_ARCHITECTURE = "ModernBertForTokenClassification"',
    '"cached_token_reporting": "positive",',
    '"eos_token": "<|im_end|>",',
    'credential_source = Ec2InstanceMetadata',
    'PWD=/home/u/project',
  ])
})

test('a value that names a variable, a key or a label is kept', () => {
  expectKept([
    'url = f"postgresql://{user}:{password}@{host}:{port}/{db}"',
    'DATABASE_URL: postgres://app:${POSTGRES_PASSWORD}@db:5432/app',
    'const url = `mongodb://${user}:${pass}@${host}/db`',
    'Use redis://<user>:<password>@host:6379',
    'TOKEN=$(kubectl get secret app -o jsonpath={.data.token})',
    'PASSWORD=$(openssl rand -base64 32)',
    'API_TOKEN=${API_TOKEN:-}',
    'export HF_TOKEN=${HF_TOKEN:?set HF_TOKEN}',
    'cmd --token $TOKEN',
    'cmd --password "$PASS"',
    '    api_key=config.openai_key',
    '    token=self.token',
    '    token=access_token',
    'access_token = oauth2_scheme',
    "fetch(url, { credentials: 'same-origin' })",
    '"password": "Password",',
    '"confirm_password": "Confirm your password",',
    'token: "Invalid or expired token",',
    'PASSWORD_FIELD = "password"',
    'XGoogAPIKey = "x-goog-api-key"',
    'TOKEN_WHEN: "WHEN", TOKEN_MODEL: "MODEL", TOKEN_ALGORITHM: "ALGORITHM",',
    "auth: { api_key_env: 'OPENAI_API_KEY' },",
    'fields = ["username", "password", "email_address"]',
    'const email, password = "grafana@example.com", "test-admin-password"',
    'docker run --secret id=mysecret,src=file img',
    'Cookie: a small file a site stores',
    'enter the password again',
    'GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}',
    // A model's output token, a syntax highlighter's, a class name, a workflow's file name, a result.
    '"token": "alternative", "logprob": -2.1,',
    "{ token: 'string.quote', next: '@pop' }",
    'TOKEN = "ModernBertForTokenClassification"',
    'which triggers `pypi-publish-vllm-sr-sim.yml`.',
    '"pass": "passed",',
    'sessionKey: "x-user-id"',
    'f"Cookie: vsr_session={session_token}"',
    '"#access_token=" + fragment + "&opaque=" + opaque,',
  ])
  expectKept(['token = request_handle:headers():get("api-key")', '#   gateway-model (auth.credential: delegated). The mocker'], 'config')
})

test('a file is read by its name, and a command by the files it names', () => {
  expect(['.env', '/repo/.env.local', 'config/app.yaml', '/home/u/.aws/credentials', '.npmrc'].map(kindOf)).toEqual(Array(5).fill('config'))
  expect(['src/a.ts', 'pkg/b.go', 'c.py'].map(kindOf)).toEqual(Array(3).fill('code'))
  expect(['run.sh', 'Dockerfile', 'README.md', 'data.json'].map(kindOf)).toEqual(Array(4).fill('unknown'))
  expect(kindOfCommand('cat .env')).toBe('config')
  expect(kindOfCommand('grep -n token src/a.ts src/b.py')).toBe('code')
  expect(kindOfCommand('cat .env src/a.ts')).toBe('unknown')
  expect(kindOfCommand('kubectl get cm -o yaml')).toBe('unknown')
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

test('an IP span the model cut short hides the whole address', () => {
  for (const [text, value, hidden] of [
    ['connect to 192.168.1.20 now', '192.168.1', 'connect to [vsr:ip] now'],
    ['the host is 10.0.0.12.', '10.0.0', 'the host is [vsr:ip].'],
    ['listen on 192.168.1.20:8080', '168.1.20', 'listen on [vsr:ip]:8080'],
  ] as const) {
    expect(hide(text, piiIn(text, [ipSpan(text, value)], 0.5)).text).toBe(hidden)
  }
})
