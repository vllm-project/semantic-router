import styles from './PlaygroundPage.module.css'

const PlaygroundPage = () => {
  return (
    <div className={styles.container}>
      <div className={styles.iframeContainer}>
          <iframe
          src="/embedded/openwebui/"
            className={styles.iframe}
            title="Open WebUI Playground"
            allowFullScreen
          />
      </div>
    </div>
  )
}

export default PlaygroundPage
