import React from "react";

export const SignalIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <path d="M2 12V8M5 12V6M8 12V4M11 12V7M14 12V2" strokeLinecap="round" />
  </svg>
);

export const ModelIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <path d="M8 2l5 2.5v7L8 14 3 11.5v-7L8 2Z" strokeLinejoin="round" />
    <path d="M3 4.5 8 7l5-2.5M8 7v7" strokeLinecap="round" />
  </svg>
);

export const RouteIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <path
      d="M2 8h4l2-4h6M8 8l2 4h4"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

export const PluginIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <rect x="3" y="1" width="10" height="14" rx="2" />
    <path d="M6 5h4M6 8h4M6 11h2" strokeLinecap="round" />
  </svg>
);

export const BackendIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <rect x="2" y="2" width="12" height="4" rx="1" />
    <rect x="2" y="10" width="12" height="4" rx="1" />
    <path d="M8 6v4" strokeLinecap="round" />
  </svg>
);

export const GlobalIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    className={className}
    viewBox="0 0 16 16"
    fill="none"
    stroke="currentColor"
    strokeWidth="1.5"
  >
    <circle cx="8" cy="8" r="6" />
    <path d="M2 8h12M8 2c-2 2-2 10 0 12M8 2c2 2 2 10 0 12" />
  </svg>
);
