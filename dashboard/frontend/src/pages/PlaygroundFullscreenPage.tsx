import { useEffect } from 'react'
import styles from './PlaygroundFullscreenPage.module.css'
import ChatComponent from '../components/ChatComponent'

const PlaygroundFullscreenPage = () => {
  useEffect(() => {
    // Add fullscreen class to body on mount
    document.body.classList.add('playground-fullscreen')
    
    // Remove on unmount
    return () => {
      document.body.classList.remove('playground-fullscreen')
    }
  }, [])

  return (
    <div className={styles.container}>
      <ChatComponent
        endpoint="/api/router/v1/chat/completions"
        isFullscreenMode={true}
      />
    </div>
  )
}

export default PlaygroundFullscreenPage

