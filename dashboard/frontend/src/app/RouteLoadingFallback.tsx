import React from 'react'

const fallbackStyle: React.CSSProperties = {
  alignItems: 'center',
  color: 'var(--color-text-secondary)',
  display: 'flex',
  fontSize: '0.875rem',
  justifyContent: 'center',
  minHeight: '12rem',
  width: '100%',
}

const RouteLoadingFallback: React.FC = () => (
  <div aria-live="polite" role="status" style={fallbackStyle}>
    Loading view...
  </div>
)

export default RouteLoadingFallback
