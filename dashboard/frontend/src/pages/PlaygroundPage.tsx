import styles from './PlaygroundPage.module.css'

const PlaygroundPage = () => {
  return (
    <div className={styles.container}>
      <div className={styles.iframeContainer}>
          <iframe
          src="/workspace"
            className={styles.iframe}
            title="Open WebUI Playground"
            allowFullScreen
          />
      </div>
    </div>
  )
}

export default PlaygroundPage
