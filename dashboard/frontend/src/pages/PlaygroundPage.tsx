import { useCallback } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'

import styles from './PlaygroundPage.module.css'
import AnimatedBackground from '../components/AnimatedBackground'
import ChatComponent from '../components/ChatComponent'
import { isPlaygroundInvocation } from '../types/playgroundInvocation'

const PlaygroundPage = () => {
  const location = useLocation()
  const navigate = useNavigate()
  const locationState =
    location.state && typeof location.state === 'object'
      ? (location.state as Record<string, unknown>)
      : null
  const invocation = isPlaygroundInvocation(locationState?.playgroundInvocation)
    ? locationState.playgroundInvocation
    : null
  const handleInvocationConsumed = useCallback(() => {
    navigate(`${location.pathname}${location.search}${location.hash}`, {
      replace: true,
      state: null,
    })
  }, [location.hash, location.pathname, location.search, navigate])

  return (
    <div className={styles.container}>
      <AnimatedBackground speed="slow" />
      <div className={styles.chatWrapper}>
        <ChatComponent
          endpoint="/api/playground/v1/chat/completions"
          invocation={invocation}
          onInvocationConsumed={handleInvocationConsumed}
        />
      </div>
    </div>
  )
}

export default PlaygroundPage
