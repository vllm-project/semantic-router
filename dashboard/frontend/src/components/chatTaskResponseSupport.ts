import {
  consumeEventStream,
  isEventStreamContentType,
  parseChatCompletionPayload,
  type ParsedChatCompletion,
} from './chatResponseParsing'
import {
  PlaygroundRequestFailure,
  playgroundHTTPFailure,
  playgroundResponseFailure,
} from './playgroundErrorPresentation'

export class PlaygroundIncompleteResponseFailure extends PlaygroundRequestFailure {}

type ApplyParsedCompletion = (parsedCompletion: ParsedChatCompletion, streaming: boolean) => void

const invalidResponseFailure = () =>
  playgroundResponseFailure(
    'The model service returned an invalid response. Try again.',
    'The response body was not valid chat-completion JSON.',
  )

const responseErrorFailure = (technicalDetails: string) =>
  playgroundResponseFailure(
    'The model service could not complete this request. Review the model settings, then try again.',
    technicalDetails,
  )

const incompleteResponseFailure = (technicalDetails: string) =>
  playgroundResponseFailure(
    'The model service returned an incomplete response. Try again.',
    technicalDetails,
  )

const assertCompletionSucceeded = (parsedCompletion: ParsedChatCompletion): void => {
  if (parsedCompletion.errorMessage) {
    throw responseErrorFailure(parsedCompletion.errorMessage)
  }
}

const readNonStreamingCompletion = async (response: Response): Promise<ParsedChatCompletion> => {
  const parsedCompletion = parseChatCompletionPayload(await response.text())
  if (!parsedCompletion) {
    throw invalidResponseFailure()
  }

  assertCompletionSucceeded(parsedCompletion)
  if (parsedCompletion.choices.length === 0) {
    throw incompleteResponseFailure('The chat-completion response contained no choices.')
  }
  return parsedCompletion
}

const consumeStreamingCompletion = async (
  response: Response,
  applyParsedCompletion: ApplyParsedCompletion,
): Promise<void> => {
  if (!response.body) {
    throw incompleteResponseFailure('The streaming response did not contain a response body.')
  }

  await consumeEventStream(response.body, (data) => {
    const parsedCompletion = parseChatCompletionPayload(data)
    if (!parsedCompletion) return
    assertCompletionSucceeded(parsedCompletion)
    applyParsedCompletion(parsedCompletion, true)
  })
}

export const assertPlaygroundResponseSuccess = async (response: Response): Promise<void> => {
  if (response.ok) return
  throw playgroundHTTPFailure(response.status, response.statusText, await response.text())
}

export const consumePlaygroundResponseBody = async (
  response: Response,
  applyParsedCompletion: ApplyParsedCompletion,
): Promise<void> => {
  const choices = new Map<
    number,
    { hasOutput: boolean; hasReasoning: boolean; reachedLimit: boolean }
  >()
  const applyAndTrack: ApplyParsedCompletion = (parsedCompletion, streaming) => {
    for (const choice of parsedCompletion.choices) {
      const state = choices.get(choice.index) ?? {
        hasOutput: false,
        hasReasoning: false,
        reachedLimit: false,
      }
      state.hasOutput ||= Boolean(choice.content.trim()) || choice.toolCalls.length > 0
      state.hasReasoning ||= Boolean(choice.reasoningContent.trim())
      state.reachedLimit ||= choice.finishReason === 'length'
      choices.set(choice.index, state)
    }
    applyParsedCompletion(parsedCompletion, streaming)
  }

  if (isEventStreamContentType(response.headers.get('content-type'))) {
    await consumeStreamingCompletion(response, applyAndTrack)
  } else {
    applyAndTrack(await readNonStreamingCompletion(response), false)
  }

  if ([...choices.values()].some((choice) => choice.reachedLimit)) {
    throw new PlaygroundIncompleteResponseFailure(
      'The output budget was reached. Increase Output budget (including reasoning) and try again.',
      'The model returned finish_reason: length; the answer may be truncated or contain only reasoning.',
    )
  }
  const emptyChoice = [...choices.entries()].find(([, choice]) => !choice.hasOutput)
  if (choices.size === 0 || emptyChoice) {
    throw new PlaygroundIncompleteResponseFailure(
      emptyChoice?.[1].hasReasoning
        ? 'The model returned reasoning without a final answer. Increase Output budget and try again.'
        : 'The model returned no answer or tool calls. Try again.',
      emptyChoice
        ? `Completion choice ${emptyChoice[0]} contained no non-empty answer text or tool calls.`
        : 'The response stream contained no completion choices.',
    )
  }
}
