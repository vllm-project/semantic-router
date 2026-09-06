import {
  consumeEventStream,
  isEventStreamContentType,
  parseChatCompletionPayload,
  type ParsedChatCompletion,
} from './chatResponseParsing'
import { playgroundHTTPFailure, playgroundResponseFailure } from './playgroundErrorPresentation'

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
  if (isEventStreamContentType(response.headers.get('content-type'))) {
    await consumeStreamingCompletion(response, applyParsedCompletion)
    return
  }

  applyParsedCompletion(await readNonStreamingCompletion(response), false)
}
