export interface SetupState {
  setupMode: boolean;
  listenerPort: number;
  models: number;
  decisions: number;
  hasModels: boolean;
  hasDecisions: boolean;
  canActivate: boolean;
}

export interface SetupValidateResponse {
  valid: boolean;
  config?: Record<string, unknown>;
  models: number;
  decisions: number;
  signals: number;
  canActivate: boolean;
}

export interface SetupActivateResponse {
  status: string;
  setupMode: boolean;
  message?: string;
}

export interface SetupImportRemoteResponse {
  config: Record<string, unknown>;
  models: number;
  decisions: number;
  signals: number;
  canActivate: boolean;
  sourceUrl: string;
}
