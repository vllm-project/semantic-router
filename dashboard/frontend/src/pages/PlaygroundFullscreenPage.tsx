import { useEffect } from 'react'
import styles from './PlaygroundFullscreenPage.module.css'
import AgentPlayground from '../components/AgentPlayground'
import { useReadonly } from '../contexts/ReadonlyContext'
import { routerPublicEndpoint } from '../utils/routerPublicApi'

const PlaygroundFullscreenPage = () => {
  const { routerPublicUrl } = useReadonly()
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
      <AgentPlayground
        endpoint={routerPublicEndpoint(routerPublicUrl, '/v1/chat/completions')}
        fullscreen
      />
    </div>
  )
}

export default PlaygroundFullscreenPage
