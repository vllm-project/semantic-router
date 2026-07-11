import styles from './PlaygroundPage.module.css'
import ChatComponent from '../components/ChatComponent'

const PlaygroundPage = () => {
  return (
    <div className={styles.container}>
      <div className={styles.chatWrapper}>
        <ChatComponent endpoint="/api/router/v1/chat/completions" />
      </div>
    </div>
  )
}

export default PlaygroundPage
