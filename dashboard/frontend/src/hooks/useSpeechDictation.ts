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

function getSpeechRecognitionCtor(): WebSpeechRecognitionCtor | null {
  if (typeof window === 'undefined') {
    return null
  }
  const w = window as Window &
    typeof globalThis & {
      SpeechRecognition?: WebSpeechRecognitionCtor
      webkitSpeechRecognition?: WebSpeechRecognitionCtor
    }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

export function useSpeechDictation(onChangeInput: (value: string) => void) {
  const [isListening, setIsListening] = useState(false)
  const recognitionRef = useRef<WebSpeechRecognition | null>(null)
  const prefixRef = useRef('')
  const finalsRef = useRef<string[]>([])
  const onChangeInputRef = useRef(onChangeInput)
  onChangeInputRef.current = onChangeInput

  const isSupported = useMemo(() => Boolean(getSpeechRecognitionCtor()), [])

  const detachRecognition = useCallback((instance: WebSpeechRecognition) => {
    instance.onresult = null
    instance.onerror = null
    instance.onend = null
  }, [])

  const stopListening = useCallback(() => {
    const instance = recognitionRef.current
    if (!instance) {
      setIsListening(false)
      return
    }
    recognitionRef.current = null
    detachRecognition(instance)
    try {
      instance.abort()
    } catch {
      try {
        instance.stop()
      } catch {
        // ignore
      }
    }
    setIsListening(false)
  }, [detachRecognition])

  const startListening = useCallback(() => {
    const Ctor = getSpeechRecognitionCtor()
    if (!Ctor) {
      return
    }

    const existing = recognitionRef.current
    if (existing) {
      detachRecognition(existing)
      try {
        existing.abort()
      } catch {
        try {
          existing.stop()
        } catch {
          // ignore
        }
      }
      recognitionRef.current = null
    }

    const recognition = new Ctor()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = typeof navigator !== 'undefined' && navigator.language
      ? navigator.language
      : 'en-US'

    recognition.onresult = (event: WebSpeechRecognitionEvent) => {
      let interim = ''
      for (let i = event.resultIndex; i < event.results.length; i += 1) {
        const result = event.results[i]
        const text = result[0]?.transcript ?? ''
        if (result.isFinal) {
          finalsRef.current.push(text)
        } else {
          interim += text
        }
      }
      const finals = finalsRef.current.join('')
      onChangeInputRef.current(prefixRef.current + finals + interim)
    }

    recognition.onerror = () => {
      if (recognitionRef.current !== recognition) {
        return
      }
      recognitionRef.current = null
      detachRecognition(recognition)
      setIsListening(false)
    }

    recognition.onend = () => {
      if (recognitionRef.current !== recognition) {
        return
      }
      recognitionRef.current = null
      detachRecognition(recognition)
      setIsListening(false)
    }

    try {
      recognition.start()
      recognitionRef.current = recognition
      setIsListening(true)
    } catch {
      detachRecognition(recognition)
      recognitionRef.current = null
      setIsListening(false)
    }
  }, [detachRecognition])

  const beginListeningFromInput = useCallback(
    (currentText: string) => {
      prefixRef.current = currentText.length === 0 ? '' : `${currentText.replace(/\s+$/, ' ')} `
      finalsRef.current = []
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
          // ignore
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
