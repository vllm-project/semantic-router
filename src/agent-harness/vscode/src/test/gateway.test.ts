// Runs the gateway module against a fake gateway on a local port, with plain
// classes standing in for VS Code's parts, so no VS Code is needed.
import { strict as assert } from 'node:assert';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { AddressInfo } from 'node:net';
import { after, before, test } from 'node:test';
import type * as vscode from 'vscode';
import {
  chat,
  describeRoute,
  estimateTokens,
  listModels,
  serverSentData,
  toWireMessages,
  type ChatRequest,
  type Parts,
} from '../gateway';

class Text {
  constructor(public value: string) {}
}
class ToolCall {
  constructor(
    public callId: string,
    public name: string,
    public input: object,
  ) {}
}
class ToolResult {
  constructor(
    public callId: string,
    public content: unknown[],
  ) {}
}
const parts: Parts = { Text, ToolCall, ToolResult };
const bytes = (text: string) => new TextEncoder().encode(text);
const message = (role: number, ...content: unknown[]) =>
  ({ role, content, name: undefined }) as vscode.LanguageModelChatRequestMessage;

type Handler = (request: IncomingMessage, body: string, response: ServerResponse) => unknown;
let handle: Handler = (_request, _body, response) => response.end();
const server = createServer((request, response) => {
  let body = '';
  request.on('data', (chunk) => (body += chunk));
  request.on('end', () => void handle(request, body, response));
});
before(() => new Promise<void>((resolve) => server.listen(0, '127.0.0.1', resolve)));
after(() => {
  server.closeAllConnections();
  server.close();
});
const gateway = (suffix = '') => ({
  url: `http://127.0.0.1:${(server.address() as AddressInfo).port}${suffix}`,
  apiKey: 'secret-key',
  sessionId: 'vscode-test',
});
const pause = () => new Promise((resolve) => setTimeout(resolve, 5));

// Writes each piece as its own write, so a reader sees lines cut anywhere.
function stream(response: ServerResponse, headers: Record<string, string>, pieces: string[]): Promise<void> {
  response.writeHead(200, { 'content-type': 'text/event-stream', ...headers });
  return pieces.reduce(
    (previous, piece) => previous.then(async () => {
      response.write(piece);
      await pause();
    }),
    Promise.resolve(),
  ).then(() => void response.end());
}

const event = (chunk: object) => `data: ${JSON.stringify(chunk)}\n\n`;

async function collect(request: ChatRequest, signal?: AbortSignal) {
  const reported: unknown[] = [];
  const route = await chat(gateway(), request, parts, (part) => reported.push(part), signal);
  return { route, reported };
}

test('maps a tool loop to Chat Completions messages', () => {
  const wire = toWireMessages(
    [
      message(3, new Text('You are a coding agent.')),
      message(1, new Text('List the files.')),
      message(2, new Text('Listing.'), new ToolCall('call_1', 'list_dir', { path: '.' })),
      message(
        1,
        new ToolResult('call_1', [new Text('a.txt\n'), { mimeType: 'application/json', data: bytes('{"n":1}') }]),
        new Text('Now read it.'),
      ),
      message(2, new ToolCall('call_2', 'read_file', {})),
    ],
    parts,
  );
  assert.deepEqual(wire, [
    { role: 'system', content: 'You are a coding agent.' },
    { role: 'user', content: 'List the files.' },
    {
      role: 'assistant',
      content: 'Listing.',
      tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'list_dir', arguments: '{"path":"."}' } }],
    },
    { role: 'tool', tool_call_id: 'call_1', content: 'a.txt\n{"n":1}' },
    { role: 'user', content: 'Now read it.' },
    {
      role: 'assistant',
      content: null,
      tool_calls: [{ id: 'call_2', type: 'function', function: { name: 'read_file', arguments: '{}' } }],
    },
  ]);
});

test('leaves out parts Chat Completions cannot carry, and messages left empty', () => {
  const wire = toWireMessages(
    [
      message(1, new Text('Look at '), { mimeType: 'cache_control', data: bytes('ephemeral') }, new Text('this.')),
      message(1, { mimeType: 'image/png', data: new Uint8Array([137, 80]) }),
      message(2, { thinking: 'not part of the stable API' }),
      message(1, new ToolResult('call_3', [{ mimeType: 'image/png', data: new Uint8Array([1]) }])),
    ],
    parts,
  );
  assert.deepEqual(wire, [
    { role: 'user', content: 'Look at this.' },
    { role: 'tool', tool_call_id: 'call_3', content: '' },
  ]);
});

test('estimates about four characters a token', () => {
  assert.equal(estimateTokens('abcdefghi', parts), 3);
  assert.ok(estimateTokens(message(1, new Text('x'.repeat(400))), parts) >= 100);
});

const catalog = {
  object: 'list',
  data: [
    { id: 'vllm-sr/auto', description: 'Default recipe', routing: { api: 'chat', resolution: 'virtual', selectable: true, recipe: 'default' } },
    { id: 'backend-b', routing: { api: 'chat', resolution: 'passthrough', selectable: true } },
    { id: 'coding', routing: { api: 'chat', resolution: 'virtual', selectable: true, recipe: 'coding' } },
    { id: 'vllm-sr/review', routing: { api: 'chat', resolution: 'virtual', selectable: false, delegate: 'review', recipe: 'review' } },
  ],
};

test('lists the entrypoints by default, with the API key, under any base URL form', async () => {
  const seen: string[] = [];
  handle = (request, _body, response) => {
    seen.push(`${request.method} ${request.url} ${request.headers.authorization}`);
    response.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify(catalog));
  };
  for (const suffix of ['', '/', '/v1', '/v1/']) {
    const models = await listModels(gateway(suffix), []);
    assert.deepEqual(
      models.map((model) => model.id),
      ['vllm-sr/auto', 'coding'],
    );
  }
  assert.deepEqual(seen, Array(4).fill('GET /v1/models Bearer secret-key'));
});

test('lists the wanted models in the order asked, skipping ones the gateway lacks', async () => {
  handle = (_request, _body, response) => response.writeHead(200).end(JSON.stringify(catalog));
  const models = await listModels(gateway(), ['backend-b', 'missing', 'vllm-sr/auto']);
  assert.deepEqual(
    models.map((model) => model.id),
    ['backend-b', 'vllm-sr/auto'],
  );
});

test('passes on the gateway error message, with a hint on 401', async () => {
  handle = (_request, _body, response) =>
    response
      .writeHead(401, { 'content-type': 'application/json' })
      .end('{"error":{"message":"Unauthorized: missing or invalid API key","type":"authentication_error"}}');
  await assert.rejects(listModels(gateway(), []), /answered 401: Unauthorized: missing or invalid API key\. Run "vLLM Semantic Router: Set Gateway API Key"/);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), /answered 401/);
});

test('takes the API key and bearer tokens out of every error the gateway words', async () => {
  // Matches any part of the key, which a cut or a parse error could leave.
  const redacted = (expected: RegExp) => (error: Error) => {
    assert.doesNotMatch(`${error.message}\n${error.stack}`, /secre|other-token/);
    assert.match(error.message, expected);
    return true;
  };
  handle = (_request, _body, response) =>
    response
      .writeHead(401, { 'content-type': 'application/json' })
      .end('{"error":{"message":"Incorrect API key secret-key, sent as Authorization: Bearer other-token"}}');
  const echoed = redacted(/Incorrect API key \*\*\*, sent as Authorization: Bearer \*\*\*\./);
  await assert.rejects(listModels(gateway(), []), echoed);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), echoed);
  // A page longer than is kept, with the key where it is cut.
  handle = (_request, _body, response) => response.writeHead(502).end(`${'x'.repeat(495)}secret-key`);
  await assert.rejects(listModels(gateway(), []), redacted(/answered 502: x+\*\*\*\./));
  handle = (_request, _body, response) => stream(response, {}, [event({ error: { message: 'upstream refused secret-key' } })]);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), redacted(/failed the reply: upstream refused \*\*\*/));
  // A parse error quotes the text it could not read.
  handle = (_request, _body, response) => stream(response, {}, ['data: secret-key\n\n']);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), redacted(/not JSON/));
});

test('says the gateway refused the connection, with its address, and no key', async () => {
  const closed = createServer();
  await new Promise<void>((resolve) => closed.listen(0, '127.0.0.1', resolve));
  const { port } = closed.address() as AddressInfo;
  await new Promise((resolve) => closed.close(resolve));
  // A URL that holds the key, which the message quotes.
  const dead = { url: `http://127.0.0.1:${port}/secret-key`, apiKey: 'secret-key' };
  const refused = new RegExp(`The gateway at http://127\\.0\\.0\\.1:${port}/\\*\\*\\* refused the connection \\(connect ECONNREFUSED 127\\.0\\.0\\.1:${port}\\)\\.$`);
  await assert.rejects(listModels(dead, []), refused);
  await assert.rejects(chat(dead, { model: 'vllm-sr/auto', messages: [] }, parts, () => {}), refused);
});

test('gives up on a gateway that accepts the request and never answers', { timeout: 5000 }, async () => {
  handle = () => {};
  const impatient = { ...gateway(), timeoutMs: 100 };
  const timedOut = (error: Error) => {
    assert.equal(error.name, 'TimeoutError');
    assert.equal(error.message, `The gateway at ${impatient.url} did not answer within 0.1 s.`);
    return true;
  };
  for (const request of [() => listModels(impatient, []), () => chat(impatient, { model: 'vllm-sr/auto', messages: [] }, parts, () => {})]) {
    const started = Date.now();
    await assert.rejects(request(), timedOut);
    assert.ok(Date.now() - started >= 100);
  }
});

test('keeps reading a reply that started in time, however long it streams', async () => {
  handle = async (_request, _body, response) => {
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    for (const word of ['one ', 'two ', 'three ', 'four']) {
      response.write(event({ choices: [{ delta: { content: word } }] }));
      await new Promise((resolve) => setTimeout(resolve, 60));
    }
    response.end(event({ choices: [{ delta: {}, finish_reason: 'stop' }] }));
  };
  const impatient = { ...gateway(), timeoutMs: 100 };
  const reported: unknown[] = [];
  await chat(impatient, { model: 'vllm-sr/auto', messages: [] }, parts, (part) => reported.push(part));
  assert.deepEqual(reported, ['one ', 'two ', 'three ', 'four'].map((word) => new Text(word)));
});

test('streams text and a tool call cut across writes, and reads the routing headers', async () => {
  let request: { url?: string; headers: IncomingMessage['headers']; body: Record<string, unknown> } | undefined;
  handle = (incoming, body, response) => {
    request = { url: incoming.url, headers: incoming.headers, body: JSON.parse(body) };
    const text = event({ model: 'qwen3-8b', choices: [{ index: 0, delta: { role: 'assistant', content: 'Hel' } }] });
    return stream(
      response,
      {
        'x-vsr-selected-model': 'qwen3-8b',
        'x-vsr-selected-decision': 'coding_task',
        'x-vsr-selected-recipe': 'default',
        'x-vsr-response-path': 'upstream',
      },
      [
        text.slice(0, 17),
        text.slice(17),
        event({ choices: [{ index: 0, delta: { content: 'lo' } }] }),
        event({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, id: 'call_9', type: 'function', function: { name: 'read_file', arguments: '{"pa' } }] } }] }),
        event({ choices: [{ index: 0, delta: { tool_calls: [{ index: 0, function: { arguments: 'th":"a.txt"}' } }] } }] }),
        event({ choices: [{ index: 0, delta: {}, finish_reason: 'tool_calls' }] }),
        'data: [DONE]\n\n',
      ],
    );
  };
  const tool = { name: 'read_file', description: 'Read a file', inputSchema: { type: 'object', properties: { path: { type: 'string' } } } };
  const { route, reported } = await collect({
    model: 'vllm-sr/auto',
    messages: [{ role: 'user', content: 'Read a.txt' }],
    tools: [tool, { name: 'noop', description: 'No input' }],
    toolRequired: true,
  });

  assert.equal(request?.url, '/v1/chat/completions');
  assert.equal(request?.headers.authorization, 'Bearer secret-key');
  assert.equal(request?.headers['x-session-id'], 'vscode-test');
  assert.deepEqual(request?.body, {
    model: 'vllm-sr/auto',
    messages: [{ role: 'user', content: 'Read a.txt' }],
    stream: true,
    tools: [
      { type: 'function', function: { name: 'read_file', description: 'Read a file', parameters: tool.inputSchema } },
      { type: 'function', function: { name: 'noop', description: 'No input', parameters: { type: 'object', properties: {} } } },
    ],
    tool_choice: 'required',
  });
  assert.deepEqual(reported, [new Text('Hel'), new Text('lo'), new ToolCall('call_9', 'read_file', { path: 'a.txt' })]);
  assert.deepEqual(route, {
    requested: 'vllm-sr/auto',
    model: 'qwen3-8b',
    decision: 'coding_task',
    recipe: 'default',
    path: 'upstream',
    replyModel: 'qwen3-8b',
  });
});

test('sends no tools or tool choice when VS Code offers no tools', async () => {
  let body: Record<string, unknown> = {};
  handle = (_request, raw, response) => {
    body = JSON.parse(raw);
    return stream(response, {}, [event({ choices: [{ delta: { content: 'ok' }, finish_reason: 'stop' }] }), 'data: [DONE]\n\n']);
  };
  await collect({ model: 'vllm-sr/auto', messages: [], tools: [], toolRequired: true });
  assert.deepEqual(Object.keys(body), ['model', 'messages', 'stream']);
});

test('fails a stream that ends before the reply finished', async () => {
  handle = (_request, _body, response) => stream(response, {}, [event({ choices: [{ delta: { content: 'partial' } }] })]);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), /closed the stream before the reply finished/);
});

test('fails on an error event in the stream', async () => {
  handle = (_request, _body, response) =>
    stream(response, {}, [event({ choices: [{ delta: { content: 'a' } }] }), event({ error: { message: 'upstream reset' } })]);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), /failed the reply: upstream reset/);
});

test('fails a tool call whose arguments are not a JSON object', async () => {
  handle = (_request, _body, response) =>
    stream(response, {}, [
      event({ choices: [{ delta: { tool_calls: [{ index: 0, id: 'c', function: { name: 'run', arguments: '{"cmd": ' } }] }, finish_reason: 'tool_calls' }] }),
      'data: [DONE]\n\n',
    ]);
  await assert.rejects(collect({ model: 'vllm-sr/auto', messages: [] }), /called run with arguments that are not a JSON object/);
});

test('reports parallel tool calls in index order, with unique ids when the backend sent none', async () => {
  handle = (_request, _body, response) =>
    stream(response, {}, [
      event({ choices: [{ delta: { tool_calls: [{ index: 1, function: { name: 'second', arguments: '{}' } }] } }] }),
      event({ choices: [{ delta: { tool_calls: [{ index: 0, function: { name: 'first', arguments: '{"a":1}' } }] } }] }),
      event({ choices: [{ delta: {}, finish_reason: 'tool_calls' }] }),
      'data: [DONE]\n\n',
    ]);
  const { reported } = await collect({ model: 'vllm-sr/auto', messages: [] });
  const calls = reported as ToolCall[];
  assert.deepEqual(
    calls.map((call) => [call.name, call.input]),
    [
      ['first', { a: 1 }],
      ['second', {}],
    ],
  );
  assert.match(calls[0].callId, /^call_/);
  assert.notEqual(calls[0].callId, calls[1].callId);
});

test('reads a reply the gateway answered whole instead of streamed', async () => {
  handle = (_request, _body, response) =>
    response.writeHead(200, { 'content-type': 'application/json', 'x-vsr-response-path': 'cache' }).end(
      JSON.stringify({
        model: 'backend-b',
        choices: [
          {
            message: { role: 'assistant', content: 'Cached.', tool_calls: [{ id: 'call_1', type: 'function', function: { name: 'ls', arguments: '' } }] },
            finish_reason: 'tool_calls',
          },
        ],
      }),
    );
  const { route, reported } = await collect({ model: 'vllm-sr/auto', messages: [] });
  assert.deepEqual(reported, [new Text('Cached.'), new ToolCall('call_1', 'ls', {})]);
  assert.equal(route.replyModel, 'backend-b');
  assert.equal(route.model, undefined);
});

test('stops reading when VS Code cancels the request', async () => {
  handle = (_request, _body, response) => {
    response.writeHead(200, { 'content-type': 'text/event-stream' });
    response.write(event({ choices: [{ delta: { content: 'thinking' } }] }));
  };
  const controller = new AbortController();
  const pending = collect({ model: 'vllm-sr/auto', messages: [] }, controller.signal);
  setTimeout(() => controller.abort(), 20);
  await assert.rejects(pending, { name: 'AbortError' });
});

test('splits server-sent data cut mid-line and mid-character', async () => {
  const raw = bytes('data: {"a":"é"}\r\n\n: keep-alive\n\nevent: x\ndata: [DONE]');
  const cut = raw.indexOf(0xc3) + 1;
  async function* pieces() {
    yield raw.slice(0, 9);
    yield raw.slice(9, cut);
    yield raw.slice(cut);
  }
  const data: string[] = [];
  for await (const item of serverSentData(pieces())) {
    data.push(item);
  }
  assert.deepEqual(data, ['{"a":"é"}', '[DONE]']);
});

test('describes the route from the headers, or says where the model name came from', () => {
  assert.deepEqual(describeRoute({ requested: 'vllm-sr/auto', model: 'qwen3-8b', decision: 'coding_task', recipe: 'default', path: 'upstream' }), {
    model: 'qwen3-8b',
    lines: ['Entrypoint: vllm-sr/auto', 'Model: qwen3-8b', 'Decision: coding_task', 'Recipe: default'],
  });
  assert.deepEqual(describeRoute({ requested: 'vllm-sr/auto', replyModel: 'backend-b', path: 'cache' }), {
    model: 'backend-b',
    lines: [
      'Entrypoint: vllm-sr/auto',
      'Model: backend-b (named by the reply: the gateway sent no x-vsr-selected-model)',
      'Decision: none',
      'Response path: cache',
    ],
  });
  assert.equal(describeRoute({ requested: 'backend-b' }).lines[1], 'Model: backend-b (asked for: the gateway did not say which model answered)');
});
