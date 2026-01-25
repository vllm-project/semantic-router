import styles from './PlaygroundPage.module.css'
import ChatComponent from '../components/ChatComponent'
import AnimatedBackground from '../components/AnimatedBackground'

const PlaygroundPage = () => {
  return (
    <div className={styles.container}>
      <AnimatedBackground speed="slow" />
      <div className={styles.chatWrapper}>
        <ChatComponent
          endpoint="/api/router/v1/chat/completions"
          defaultModel="MoM"
          defaultSystemPrompt="You are a helpful assistant."
        />
      </div>
    </div>
  )
}

export default PlaygroundPage
