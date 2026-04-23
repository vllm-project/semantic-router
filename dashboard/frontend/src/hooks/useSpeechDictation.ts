import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

/** Narrow surface used from the Web Speech API (Chromium; prefixed in WebKit). */
interface WebSpeechRecognitionResult {
  readonly isFinal: boolean
  readonly 0: { readonly transcript: string }
}

interface WebSpeechRecognitionResultList {
  readonly length: number
  readonly [index: number]: WebSpeechRecognitionResult
}

interface WebSpeechRecognitionEvent extends Event {
  readonly resultIndex: number
  readonly results: WebSpeechRecognitionResultList
}

interface WebSpeechRecognition extends EventTarget {
  continuous: boolean
  interimResults: boolean
  lang: string
  start(): void
  stop(): void
  abort(): void
  onresult: ((event: WebSpeechRecognitionEvent) => void) | null
  onerror: ((event: Event) => void) | null
  onend: ((event: Event) => void) | null
}

type WebSpeechRecognitionCtor = new () => WebSpeechRecognition

/**
 * Lazily resolved once `window` exists. Do not cache `null` from SSR or the ctor
 * would stay unavailable after hydration.
 */
let speechRecognitionCtor: WebSpeechRecognitionCtor | null | undefined

function resolveSpeechRecognitionCtor(): WebSpeechRecognitionCtor | null {
  if (typeof window === 'undefined') {
    return null
  }
  if (speechRecognitionCtor !== undefined) {
    return speechRecognitionCtor
  }
  const w = window as Window &
    typeof globalThis & {
      SpeechRecognition?: WebSpeechRecognitionCtor
      webkitSpeechRecognition?: WebSpeechRecognitionCtor
    }
  speechRecognitionCtor = w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
  return speechRecognitionCtor
}

function detachRecognitionHandlers(instance: WebSpeechRecognition) {
  instance.onresult = null
  instance.onerror = null
  instance.onend = null
}

function abortRecognitionSilently(instance: WebSpeechRecognition) {
  detachRecognitionHandlers(instance)
  try {
    instance.abort()
  } catch {
    try {
      instance.stop()
    } catch {
      // ignore
    }
  }
}

export function useSpeechDictation(onChangeInput: (value: string) => void) {
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef<WebSpeechRecognition | null>(null)
  const prefixRef = useRef('')
  /** Accumulated final transcripts (append avoids O(n) join each result). */
  const finalsAccRef = useRef('')
  const onChangeInputRef = useRef(onChangeInput)
  const lastEmittedRef = useRef('')

  onChangeInputRef.current = onChangeInput

  const isSupported = useMemo(() => Boolean(resolveSpeechRecognitionCtor()), [])

  const emitIfChanged = useCallback((full: string) => {
    if (full === lastEmittedRef.current) {
      return
    }
    lastEmittedRef.current = full
    onChangeInputRef.current(full)
  }, [])

  const stopListening = useCallback(() => {
    const instance = recognitionRef.current
    if (!instance) {
      setIsListening(false)
      return
    }
    recognitionRef.current = null
    abortRecognitionSilently(instance)
    setIsListening(false)
  }, [])

  const startListening = useCallback(() => {
    const Ctor = resolveSpeechRecognitionCtor()
    if (!Ctor) {
      return
    }

    const existing = recognitionRef.current
    if (existing) {
      recognitionRef.current = null
      abortRecognitionSilently(existing)
    }

    const recognition = new Ctor()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang =
      typeof navigator !== 'undefined' && navigator.language ? navigator.language : 'en-US'

    const finishActive = (instance: WebSpeechRecognition) => {
      if (recognitionRef.current !== instance) {
        return
      }
      recognitionRef.current = null
      detachRecognitionHandlers(instance)
      setIsListening(false)
    }

    recognition.onresult = (event: WebSpeechRecognitionEvent) => {
      let interim = ''
      const results = event.results
      for (let i = event.resultIndex; i < results.length; i += 1) {
        const result = results[i]
        const text = result[0]?.transcript ?? ''
        if (result.isFinal) {
          finalsAccRef.current += text
        } else {
          interim += text
        }
      }
      emitIfChanged(prefixRef.current + finalsAccRef.current + interim)
    }

    recognition.onerror = () => {
      finishActive(recognition)
    }

    recognition.onend = () => {
      finishActive(recognition)
    }

    try {
      recognition.start()
      recognitionRef.current = recognition
      setIsListening(true)
    } catch {
      detachRecognitionHandlers(recognition)
      recognitionRef.current = null
      setIsListening(false)
    }
  }, [emitIfChanged])

  const beginListeningFromInput = useCallback(
    (currentText: string) => {
      prefixRef.current = currentText.length === 0 ? '' : `${currentText.replace(/\s+$/, ' ')} `
      finalsAccRef.current = ''
      lastEmittedRef.current = ''
      startListening()
    },
    [startListening],
  )

  const toggleListening = useCallback(
    (currentText: string) => {
      if (isListening) {
        try {
          recognitionRef.current?.stop()
        } catch {
          // ignore the code
        }
        return
      }
      beginListeningFromInput(currentText)
    },
    [beginListeningFromInput, isListening],
  )

  useEffect(() => () => {
    stopListening()
  }, [stopListening])

  return {
    isSupported,
    isListening,
    toggleListening,
    stopListening,
  }
}
