// Runs extension.ts with a stand-in for the vscode module, to see what the
// user is told when VS Code lists the models, which it does silently.
import { strict as assert } from 'node:assert';
import { createServer } from 'node:http';
import Module = require('node:module');
import type { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';
import type * as vscode from 'vscode';

const shown: { text: string; actions: string[] }[] = [];
const status = { text: '', tooltip: '', visible: false, show: () => void (status.visible = true), hide: () => void (status.visible = false) };
const disposable = { dispose() {} };
let provider: vscode.LanguageModelChatProvider | undefined;
const stub = {
  LanguageModelTextPart: class {},
  LanguageModelToolCallPart: class {},
  LanguageModelToolResultPart: class {},
  StatusBarAlignment: { Right: 2 },
  EventEmitter: class {
    event = () => disposable;
  },
  window: {
    createOutputChannel: () => ({ info() {}, warn() {}, error() {} }),
    createStatusBarItem: () => status,
    showErrorMessage: async (text: string, ...actions: string[]) => void shown.push({ text, actions }),
  },
  workspace: {
    getConfiguration: () => ({ get: (key: string, fallback: unknown) => (key === 'gatewayUrl' ? gatewayUrl : fallback) }),
    onDidChangeConfiguration: () => disposable,
  },
  lm: {
    registerLanguageModelChatProvider: (_vendor: string, registered: vscode.LanguageModelChatProvider) => {
      provider = registered;
      return disposable;
    },
  },
  commands: { registerCommand: () => disposable },
};
const loader = Module as unknown as { _load: (request: string, ...rest: unknown[]) => unknown };
const load = loader._load;
loader._load = (request, ...rest) => (request === 'vscode' ? stub : load.call(loader, request, ...rest));

let answer = 401;
let gatewayUrl = '';
const server = createServer((_request, response) =>
  answer === 200
    ? response.writeHead(200).end('{"data":[{"id":"vllm-sr/auto","routing":{"resolution":"virtual"}}]}')
    : response.writeHead(answer).end('{"error":{"message":"no"}}'),
);
before(async () => {
  await new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve));
  gatewayUrl = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const { activate } = require('../extension') as typeof import('../extension');
  activate({ subscriptions: [], secrets: { get: async () => undefined, onDidChange: () => disposable } } as unknown as vscode.ExtensionContext);
});
after(() => server.close());

const token = { isCancellationRequested: false, onCancellationRequested: () => disposable } as vscode.CancellationToken;
const list = async () => (await provider?.provideLanguageModelChatInformation({ silent: true }, token)) ?? [];

test('tells the user once why the model list is empty, though VS Code lists silently', async () => {
  assert.deepEqual(await list(), []);
  assert.deepEqual(await list(), []);
  assert.deepEqual(
    shown.map(({ text, actions }) => [text, actions]),
    [["vLLM Semantic Router could not list the gateway's models. The gateway answered 401: no. Run \"vLLM Semantic Router: Set Gateway API Key\".", ['Set API Key', 'Show Log']]],
  );
  assert.ok(status.visible);
  assert.equal(status.text, '$(warning) vLLM Semantic Router');
  assert.match(status.tooltip, /^Could not list the gateway's models: The gateway answered 401/);

  answer = 503;
  await list();
  assert.equal(shown.length, 2, 'a different error is shown too');

  answer = 200;
  assert.equal((await list()).length, 1);
  assert.ok(!status.visible, 'the warning goes once the models are listed');

  answer = 503;
  await list();
  assert.equal(shown.length, 3, 'an error that comes back after a listing worked is shown again');
});
