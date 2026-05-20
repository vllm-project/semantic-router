import React from 'react'

export interface SetupStatusPageProps {
  title: string
  description: string
  actionLabel: string
  onAction: () => void
}

/** Full-screen status card used during auth/setup loading and errors. */
const SetupStatusPage: React.FC<SetupStatusPageProps> = ({
  title,
  description,
  actionLabel,
  onAction,
}) => (
  <div
    style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      minHeight: '100vh',
      padding: '2rem',
      background:
        'radial-gradient(circle at top, rgba(118, 185, 0, 0.12), transparent 30%), var(--color-bg)',
    }}
  >
    <div
      style={{
        width: '100%',
        maxWidth: '560px',
        padding: '2rem',
        borderRadius: '1rem',
        border: '1px solid var(--color-border)',
        background: 'var(--color-bg-secondary)',
        boxShadow: '0 20px 48px rgba(0, 0, 0, 0.28)',
      }}
    >
      <h1 style={{ fontSize: '1.5rem', marginBottom: '0.75rem' }}>{title}</h1>
      <p style={{ color: 'var(--color-text-secondary)', lineHeight: '1.6' }}>{description}</p>
      <button
        onClick={onAction}
        style={{
          marginTop: '1.25rem',
          padding: '0.75rem 1.15rem',
          borderRadius: '0.75rem',
          background: 'var(--color-primary)',
          color: '#081000',
          fontWeight: 700,
        }}
      >
        {actionLabel}
      </button>
    </div>
  </div>
)

export default SetupStatusPage
