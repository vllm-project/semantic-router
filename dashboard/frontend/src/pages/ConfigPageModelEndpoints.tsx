import type { NormalizedModel } from './configPageSupport'

interface ConfigPageModelEndpointsProps {
  model: NormalizedModel
  redactEndpoints: boolean
}

export default function ConfigPageModelEndpoints({
  model,
  redactEndpoints,
}: ConfigPageModelEndpointsProps) {
  if (!model.endpoints || model.endpoints.length === 0) {
    return (
      <div style={{ padding: '1rem', color: 'var(--color-text-secondary)', textAlign: 'center' }}>
        No endpoints configured for this model
      </div>
    )
  }

  return (
    <div style={{ padding: '1rem', background: 'rgba(0, 0, 0, 0.3)' }}>
      <h4
        style={{
          margin: '0 0 1rem 0',
          fontSize: '0.875rem',
          fontWeight: 600,
          color: 'var(--color-text-secondary)',
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
        }}
      >
        Endpoints for {model.name}
      </h4>
      <table style={{ width: '100%', borderCollapse: 'collapse' }}>
        <thead>
          <tr style={{ borderBottom: '1px solid var(--color-border)' }}>
            <th
              style={{
                padding: '0.5rem',
                textAlign: 'left',
                fontSize: '0.875rem',
                fontWeight: 600,
                color: 'var(--color-text-secondary)',
              }}
            >
              Name
            </th>
            <th
              style={{
                padding: '0.5rem',
                textAlign: 'left',
                fontSize: '0.875rem',
                fontWeight: 600,
                color: 'var(--color-text-secondary)',
              }}
            >
              Address
            </th>
            <th
              style={{
                padding: '0.5rem',
                textAlign: 'center',
                fontSize: '0.875rem',
                fontWeight: 600,
                color: 'var(--color-text-secondary)',
                width: '100px',
              }}
            >
              Protocol
            </th>
            <th
              style={{
                padding: '0.5rem',
                textAlign: 'center',
                fontSize: '0.875rem',
                fontWeight: 600,
                color: 'var(--color-text-secondary)',
                width: '100px',
              }}
            >
              Weight
            </th>
          </tr>
        </thead>
        <tbody>
          {model.endpoints.map((endpoint, index) => (
            <tr key={index} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
              <td style={{ padding: '0.75rem 0.5rem', fontSize: '0.875rem', fontWeight: 500 }}>
                {endpoint.name}
              </td>
              <td
                style={{
                  padding: '0.75rem 0.5rem',
                  fontSize: '0.875rem',
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--color-text-secondary)',
                }}
              >
                {redactEndpoints ? '************' : endpoint.endpoint || 'N/A'}
              </td>
              <td style={{ padding: '0.75rem 0.5rem', textAlign: 'center' }}>
                <span
                  style={{
                    padding: '0.25rem 0.5rem',
                    background:
                      endpoint.protocol === 'https'
                        ? 'rgba(34, 197, 94, 0.15)'
                        : 'rgba(234, 179, 8, 0.15)',
                    borderRadius: '4px',
                    fontSize: '0.75rem',
                    fontWeight: 600,
                    textTransform: 'uppercase',
                  }}
                >
                  {endpoint.protocol}
                </span>
              </td>
              <td
                style={{
                  padding: '0.75rem 0.5rem',
                  textAlign: 'center',
                  fontSize: '0.875rem',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {endpoint.weight}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
