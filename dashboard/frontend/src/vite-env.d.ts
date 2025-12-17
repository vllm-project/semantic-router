/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_OPENWEBUI_PORT?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare module "*.module.css" {
  const classes: { [key: string]: string };
  export default classes;
}
