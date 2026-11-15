// Puts the gateway's models in VS Code's chat model picker through the
// language model chat provider API, and shows in the status bar which model
// the gateway chose for the last reply. The gateway does all the routing.
import { randomUUID } from 'node:crypto';
import * as vscode from 'vscode';
import { chat, describeRoute, estimateTokens, listModels, toWireMessages, type Gateway, type Parts } from './gateway';

const API_KEY = 'vllm-sr.apiKey';
const parts: Parts = {
  Text: vscode.LanguageModelTextPart,
  ToolCall: vscode.LanguageModelToolCallPart,
  ToolResult: vscode.LanguageModelToolResultPart,
};

const settings = () => vscode.workspace.getConfiguration('vllm-sr');
const messageOf = (error: unknown) => (error instanceof Error ? error.message : String(error));

function abortOn(token: vscode.CancellationToken): AbortSignal {
  const controller = new AbortController();
  if (token.isCancellationRequested) {
    controller.abort();
  } else {
    token.onCancellationRequested(() => controller.abort());
  }
  return controller.signal;
}

export function activate(context: vscode.ExtensionContext): void {
  const log = vscode.window.createOutputChannel('vLLM Semantic Router', { log: true });
  const status = vscode.window.createStatusBarItem('vllm-sr.route', vscode.StatusBarAlignment.Right, 100);
  status.name = 'vLLM Semantic Router';
  status.command = 'vllm-sr.showLog';
  const changed = new vscode.EventEmitter<void>();
  // One session id for this window, so the gateway can hold each conversation
  // on its model; the gateway tells conversations apart by how they start.
  const sessionId = `vscode-${randomUUID()}`;
  const gateway = async (): Promise<Gateway> => ({
    url: settings().get<string>('gatewayUrl', 'http://localhost:8899'),
    apiKey: await context.secrets.get(API_KEY),
    sessionId,
  });
  // VS Code lists the models silently, at startup and after every key or
  // setting change, so a listing error goes to the status bar and is shown
  // once, until the listing works again.
  let listingError: string | undefined;

  const provider: vscode.LanguageModelChatProvider = {
    onDidChangeLanguageModelChatInformation: changed.event,

    async provideLanguageModelChatInformation(options, token) {
      const config = settings();
      const wanted = config.get<string[]>('models', []);
      try {
        const models = await listModels(await gateway(), wanted, abortOn(token));
        const missing = wanted.filter((id) => !models.some((model) => model.id === id));
        if (missing.length) {
          log.warn(`The gateway does not list ${missing.join(', ')}.`);
        }
        if (listingError) {
          listingError = undefined;
          status.hide();
        }
        return models.map((model) => ({
          id: model.id,
          name: model.id,
          family: 'vllm-sr',
          version: '1',
          tooltip: model.description,
          maxInputTokens: config.get<number>('maxInputTokens', 128000),
          maxOutputTokens: config.get<number>('maxOutputTokens', 16384),
          capabilities: { toolCalling: true, imageInput: false },
        }));
      } catch (error) {
        if (!token.isCancellationRequested) {
          const reason = messageOf(error);
          log.error(`Could not list the gateway's models: ${reason}`);
          status.text = '$(warning) vLLM Semantic Router';
          status.tooltip = `Could not list the gateway's models: ${reason}`;
          status.show();
          if (!options.silent || reason !== listingError) {
            listingError = reason;
            void vscode.window
              .showErrorMessage(`vLLM Semantic Router could not list the gateway's models. ${reason}`, 'Set API Key', 'Show Log')
              .then((action) => action && vscode.commands.executeCommand(action === 'Set API Key' ? 'vllm-sr.setApiKey' : 'vllm-sr.showLog'));
          }
        }
        return [];
      }
    },

    async provideLanguageModelChatResponse(model, messages, options, progress, token) {
      try {
        const route = await chat(
          await gateway(),
          {
            model: model.id,
            messages: toWireMessages(messages, parts),
            tools: options.tools,
            toolRequired: options.toolMode === vscode.LanguageModelChatToolMode.Required,
          },
          parts,
          (part) => progress.report(part),
          abortOn(token),
        );
        const { model: served, lines } = describeRoute(route);
        status.text = `$(arrow-swap) ${served}`;
        status.tooltip = ['vLLM Semantic Router, last reply', ...lines].join('\n');
        status.show();
        log.info(lines.join(', '));
      } catch (error) {
        if (token.isCancellationRequested) {
          return;
        }
        status.text = '$(warning) vLLM Semantic Router';
        status.tooltip = `The last request failed: ${messageOf(error)}`;
        status.show();
        log.error(`${model.id}: ${messageOf(error)}`);
        throw error;
      }
    },

    async provideTokenCount(_model, text) {
      return estimateTokens(text, parts);
    },
  };

  context.subscriptions.push(
    log,
    status,
    changed,
    vscode.lm.registerLanguageModelChatProvider('vllm-sr', provider),
    vscode.commands.registerCommand('vllm-sr.setApiKey', async () => {
      const key = await vscode.window.showInputBox({
        title: 'Gateway API key',
        prompt: "One of the listener's api_keys. VS Code keeps it in its secret storage.",
        password: true,
        ignoreFocusOut: true,
      });
      if (key?.trim()) {
        await context.secrets.store(API_KEY, key.trim());
      }
    }),
    vscode.commands.registerCommand('vllm-sr.clearApiKey', () => context.secrets.delete(API_KEY)),
    vscode.commands.registerCommand('vllm-sr.refreshModels', () => changed.fire()),
    vscode.commands.registerCommand('vllm-sr.showLog', () => log.show()),
    context.secrets.onDidChange((event) => {
      if (event.key === API_KEY) {
        changed.fire();
      }
    }),
    vscode.workspace.onDidChangeConfiguration((event) => {
      if (event.affectsConfiguration('vllm-sr')) {
        changed.fire();
      }
    }),
  );
}
