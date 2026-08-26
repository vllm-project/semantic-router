import { useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import styles from './PlaygroundPage.module.css'
import AnimatedBackground from '../components/AnimatedBackground'
import AgentPlayground from '../components/AgentPlayground'
import { isPlaygroundInvocation } from '../types/playgroundInvocation'
import { useReadonly } from '../contexts/ReadonlyContext'
import { routerPublicEndpoint } from '../utils/routerPublicApi'

const PlaygroundPage = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const { routerPublicUrl } = useReadonly()
  const locationState =
    location.state && typeof location.state === 'object'
      ? (location.state as Record<string, unknown>)
      : null
  const invocation = isPlaygroundInvocation(locationState?.playgroundInvocation)
    ? locationState.playgroundInvocation
    : null
  const requestedModel = new URLSearchParams(location.search).get('model')
  const handleInvocationConsumed = useCallback(() => {
    navigate(`${location.pathname}${location.search}${location.hash}`, {
      replace: true,
      state: null,
    })
  }, [location.hash, location.pathname, location.search, navigate])
  const handleModelConsumed = useCallback(() => {
    const params = new URLSearchParams(location.search)
    params.delete('model')
    const search = params.toString()
    navigate(`${location.pathname}${search ? `?${search}` : ''}${location.hash}`, {
      replace: true,
      state: location.state,
    })
  }, [location.hash, location.pathname, location.search, location.state, navigate])

  return (
    <div className={styles.container}>
      <AnimatedBackground speed="slow" />
      <div className={styles.chatWrapper}>
        <AgentPlayground
          endpoint={routerPublicEndpoint(routerPublicUrl, '/v1/chat/completions')}
          invocation={invocation}
          initialModel={requestedModel}
          onInvocationConsumed={handleInvocationConsumed}
          onInitialModelConsumed={handleModelConsumed}
        />
      </div>
    </div>
  )
}

export default PlaygroundPage
