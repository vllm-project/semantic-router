import styles from './PlaygroundPage.module.css'
import ChatComponent from '../components/ChatComponent'

const PlaygroundPage = () => {
  return (
    <div className={styles.container}>
      <ChatComponent
        endpoint="/api/router/v1/chat/completions"
        defaultModel="MoM"
        defaultSystemPrompt="You are a helpful assistant."
      />
    </div>
  )
}

export default PlaygroundPage
