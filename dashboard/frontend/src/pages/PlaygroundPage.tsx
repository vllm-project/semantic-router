import styles from './PlaygroundPage.module.css'
import AnimatedBackground from '../components/AnimatedBackground'
import ChatComponent from '../components/ChatComponent'

const PlaygroundPage = () => {
  return (
    <div className={styles.container}>
      <AnimatedBackground speed="slow" />
      <div className={styles.chatWrapper}>
        <ChatComponent endpoint="/api/router/v1/chat/completions" />
      </div>
    </div>
  )
}

export default PlaygroundPage
