import type { ReactNode, SVGProps } from 'react'

export type ProductIconName =
  | 'alert'
  | 'arrow-left'
  | 'arrow-right'
  | 'audit'
  | 'chart'
  | 'check'
  | 'chevron-down'
  | 'chevron-left'
  | 'chevron-right'
  | 'claw'
  | 'close'
  | 'code'
  | 'compute'
  | 'copy'
  | 'dashboard'
  | 'database'
  | 'decision'
  | 'edit'
  | 'evaluation'
  | 'eye'
  | 'eye-off'
  | 'insight'
  | 'label'
  | 'link'
  | 'list'
  | 'logs'
  | 'mixture'
  | 'model'
  | 'more'
  | 'play'
  | 'playground'
  | 'plus'
  | 'projection'
  | 'refresh'
  | 'search'
  | 'settings'
  | 'signal'
  | 'status'
  | 'tool'
  | 'topology'
  | 'trace'
  | 'trash'
  | 'user'

interface ProductIconProps extends Omit<SVGProps<SVGSVGElement>, 'name'> {
  name: ProductIconName
}

const paths: Record<ProductIconName, ReactNode> = {
  alert: (
    <>
      <path d="M10.3 3.7 2.6 17a2 2 0 0 0 1.7 3h15.4a2 2 0 0 0 1.7-3L13.7 3.7a2 2 0 0 0-3.4 0Z" />
      <path d="M12 9v4m0 3h.01" />
    </>
  ),
  audit: (
    <>
      <path d="M7 3.5h10v17H7z" />
      <path d="M9.5 8h5M9.5 12h5M9.5 16h3" />
    </>
  ),
  'arrow-left': <path d="m15 18-6-6 6-6M9 12h11" />,
  'arrow-right': <path d="m9 18 6-6-6-6M4 12h11" />,
  check: <path d="m5 12 4 4L19 6" />,
  'chevron-down': <path d="m6 9 6 6 6-6" />,
  'chevron-left': <path d="m14.5 6-6 6 6 6" />,
  'chevron-right': <path d="m9.5 6 6 6-6 6" />,
  claw: (
    <>
      <path d="M7.5 18.5c-2.2-1.4-3.5-3.7-3.5-6.3 0-3.7 2.6-6.8 6.1-7.5" />
      <path d="M16.5 18.5c2.2-1.4 3.5-3.7 3.5-6.3 0-3.7-2.6-6.8-6.1-7.5" />
      <path d="M9.5 9.5 12 12l2.5-2.5M8.5 21l3.5-5 3.5 5" />
    </>
  ),
  close: <path d="m7 7 10 10M17 7 7 17" />,
  code: <path d="m8.5 7-5 5 5 5M15.5 7l5 5-5 5M14 4l-4 16" />,
  compute: (
    <>
      <rect x="5" y="5" width="14" height="14" rx="2" />
      <rect x="9" y="9" width="6" height="6" rx="1" />
      <path d="M9 2v3m6-3v3M9 19v3m6-3v3M2 9h3m-3 6h3m14-6h3m-3 6h3" />
    </>
  ),
  copy: (
    <>
      <rect x="8" y="8" width="11" height="11" rx="2" />
      <path d="M16 8V6a2 2 0 0 0-2-2H6a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h2" />
    </>
  ),
  dashboard: (
    <>
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="4" rx="1.5" />
      <rect x="14" y="11" width="7" height="10" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
    </>
  ),
  database: (
    <>
      <ellipse cx="12" cy="5" rx="8" ry="3" />
      <path d="M4 5v7c0 1.7 3.6 3 8 3s8-1.3 8-3V5M4 12v7c0 1.7 3.6 3 8 3s8-1.3 8-3v-7" />
    </>
  ),
  decision: (
    <>
      <path d="M12 3v5m0 0-5 5m5-5 5 5" />
      <circle cx="12" cy="3" r="1.5" />
      <circle cx="7" cy="15" r="2" />
      <circle cx="17" cy="15" r="2" />
      <path d="M7 17v4m10-4v4" />
    </>
  ),
  edit: (
    <>
      <path d="M4 20h4l11-11-4-4L4 16v4Z" />
      <path d="m13.5 6.5 4 4" />
    </>
  ),
  eye: (
    <>
      <path d="M2.5 12s3.5-6 9.5-6 9.5 6 9.5 6-3.5 6-9.5 6-9.5-6-9.5-6Z" />
      <circle cx="12" cy="12" r="2.5" />
    </>
  ),
  'eye-off': (
    <>
      <path d="m3 3 18 18" />
      <path d="M10.6 6.2A9.6 9.6 0 0 1 12 6c6 0 9.5 6 9.5 6a15 15 0 0 1-2.1 2.8M6.6 6.6A15.4 15.4 0 0 0 2.5 12s3.5 6 9.5 6a9.7 9.7 0 0 0 3.4-.6" />
      <path d="M10.3 10.3a2.5 2.5 0 0 0 3.4 3.4" />
    </>
  ),
  evaluation: (
    <>
      <path d="M5 4h14v16H5zM8 8h5" />
      <path d="m8 14 2 2 5-5" />
    </>
  ),
  insight: (
    <>
      <path d="M9 18h6M10 22h4" />
      <path d="M8.2 15.5A7 7 0 1 1 15.8 15.5c-.8.7-1.2 1.4-1.3 2.5h-5c-.1-1.1-.5-1.8-1.3-2.5Z" />
    </>
  ),
  label: (
    <>
      <path d="M4 4h7l9 9-7 7-9-9V4Z" />
      <circle cx="8" cy="8" r="1.3" />
    </>
  ),
  link: (
    <>
      <path d="m10 13 4-4a3 3 0 1 1 4 4l-3 3a3 3 0 0 1-4.2 0" />
      <path d="m14 11-4 4a3 3 0 1 1-4-4l3-3a3 3 0 0 1 4.2 0" />
    </>
  ),
  list: (
    <>
      <path d="M9 6h11M9 12h11M9 18h11" />
      <path d="M4 6h.01M4 12h.01M4 18h.01" />
    </>
  ),
  logs: (
    <>
      <path d="M5 4h14v16H5z" />
      <path d="M8.5 8h7M8.5 12h7M8.5 16h4" />
    </>
  ),
  mixture: (
    <>
      <circle cx="6" cy="6" r="2.5" />
      <circle cx="18" cy="6" r="2.5" />
      <circle cx="12" cy="18" r="2.5" />
      <path d="m8 7.5 2.8 7.8m5.2-7.8-2.8 7.8M8.5 6h7" />
    </>
  ),
  model: (
    <>
      <path d="m12 3 8 4.5-8 4.5-8-4.5L12 3Z" />
      <path d="m4 12 8 4.5 8-4.5M4 16.5l8 4.5 8-4.5" />
    </>
  ),
  more: (
    <>
      <circle cx="5" cy="12" r="1" fill="currentColor" stroke="none" />
      <circle cx="12" cy="12" r="1" fill="currentColor" stroke="none" />
      <circle cx="19" cy="12" r="1" fill="currentColor" stroke="none" />
    </>
  ),
  play: <path d="m8 5 11 7-11 7V5Z" />,
  playground: (
    <>
      <path d="M8 4h8l1 5 3 6a3 3 0 0 1-5.2 3L13 16h-2l-1.8 2A3 3 0 0 1 4 15l3-6 1-5Z" />
      <path d="M8 10h4M10 8v4M16.5 9.5h.01M18.5 11.5h.01" />
    </>
  ),
  plus: <path d="M12 5v14M5 12h14" />,
  projection: (
    <>
      <circle cx="6" cy="12" r="3" />
      <circle cx="18" cy="6" r="2" />
      <circle cx="18" cy="18" r="2" />
      <path d="M9 11 16 7M9 13l7 4" />
    </>
  ),
  refresh: <path d="M20 7v5h-5M4 17v-5h5M6.1 8A7 7 0 0 1 18.5 6L20 8M4 16l1.5 2A7 7 0 0 0 18 16" />,
  search: (
    <>
      <circle cx="11" cy="11" r="7" />
      <path d="m16.5 16.5 4 4" />
    </>
  ),
  settings: (
    <>
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.7 1.7 0 0 0 .3 1.9l.1.1-2.8 2.8-.1-.1a1.7 1.7 0 0 0-1.9-.3 1.7 1.7 0 0 0-1 1.5V21h-4v-.1a1.7 1.7 0 0 0-1-1.5 1.7 1.7 0 0 0-1.9.3l-.1.1L4.2 17l.1-.1a1.7 1.7 0 0 0 .3-1.9 1.7 1.7 0 0 0-1.5-1H3v-4h.1a1.7 1.7 0 0 0 1.5-1A1.7 1.7 0 0 0 4.3 7l-.1-.1L7 4.2l.1.1A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-1.5V3h4v.1a1.7 1.7 0 0 0 1 1.5 1.7 1.7 0 0 0 1.9-.3l.1-.1L19.8 7l-.1.1a1.7 1.7 0 0 0-.3 1.9 1.7 1.7 0 0 0 1.5 1h.1v4h-.1a1.7 1.7 0 0 0-1.5 1Z" />
    </>
  ),
  signal: (
    <>
      <path d="M5 16a10 10 0 0 1 0-8M9 13a5 5 0 0 1 0-2m10 5a10 10 0 0 0 0-8m-4 5a5 5 0 0 0 0-2" />
      <circle cx="12" cy="12" r="2" />
    </>
  ),
  status: (
    <>
      <circle cx="12" cy="12" r="9" />
      <path d="m8 12 2.5 2.5L16.5 8" />
    </>
  ),
  tool: (
    <path d="M14.5 6.2a4.5 4.5 0 0 0-5.8 5.7L3.5 17a2.5 2.5 0 0 0 3.5 3.5l5.1-5.2a4.5 4.5 0 0 0 5.7-5.8l-2.7 2.7-3.3-.7-.7-3.3 2.7-2.7Z" />
  ),
  chart: (
    <>
      <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
    </>
  ),
  topology: (
    <>
      <circle cx="12" cy="5" r="2.5" />
      <circle cx="6" cy="18" r="2.5" />
      <circle cx="18" cy="18" r="2.5" />
      <path d="m10.8 7.2-3.6 8.4m6-8.4 3.6 8.4M8.5 18h7" />
    </>
  ),
  trace: (
    <>
      <circle cx="5" cy="6" r="2" />
      <circle cx="19" cy="6" r="2" />
      <circle cx="12" cy="18" r="2" />
      <path d="M7 6h10M6 8l5 8m7-8-5 8" />
    </>
  ),
  trash: (
    <>
      <path d="M4 7h16M9 3h6l1 4H8l1-4ZM7 7l1 14h8l1-14" />
      <path d="M10 11v6m4-6v6" />
    </>
  ),
  user: (
    <>
      <circle cx="12" cy="8" r="4" />
      <path d="M4.5 20c.6-4.2 3.1-6.5 7.5-6.5s6.9 2.3 7.5 6.5" />
    </>
  ),
}

export default function ProductIcon({ name, ...props }: ProductIconProps) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...props}
    >
      {paths[name]}
    </svg>
  )
}
