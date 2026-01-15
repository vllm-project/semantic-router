import { createContext, useContext, useState, useCallback, ReactNode, Dispatch, SetStateAction } from 'react'

export interface Message {
    id: string
    role: 'user' | 'assistant' | 'system'
    content: string
    timestamp: Date
    isStreaming?: boolean
    headers?: Record<string, string>
}

interface ChatContextValue {
    messages: Message[]
    setMessages: Dispatch<SetStateAction<Message[]>>
    clearMessages: () => void
}

const ChatContext = createContext<ChatContextValue | undefined>(undefined)

export const ChatProvider = ({ children }: { children: ReactNode }) => {
    const [messages, setMessages] = useState<Message[]>([])

    const clearMessages = useCallback(() => {
        setMessages([])
    }, [])

    return (
        <ChatContext.Provider value={{ messages, setMessages, clearMessages }}>
            {children}
        </ChatContext.Provider>
    )
}

export const useChatContext = () => {
    const context = useContext(ChatContext)
    if (!context) {
        throw new Error('useChatContext must be used within a ChatProvider')
    }
    return context
}
