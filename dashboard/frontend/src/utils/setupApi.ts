import {
  SetupActivateResponse,
  SetupImportRemoteResponse,
  SetupState,
  SetupValidateResponse,
} from "../types/setup";

async function readErrorMessage(response: Response): Promise<string> {
  const body = await response.text();
  if (!body) {
    return `HTTP ${response.status}: ${response.statusText}`;
  }

  try {
    const parsed = JSON.parse(body) as { error?: string; message?: string };
    return parsed.message || parsed.error || body;
  } catch {
    return body;
  }
}

async function postSetupConfig<T>(
  path: string,
  config: object,
): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ config }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}

export async function fetchSetupState(): Promise<SetupState> {
  const response = await fetch("/api/setup/state");
  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }
  return response.json();
}

export async function validateSetupConfig(
  config: object,
): Promise<SetupValidateResponse> {
  return postSetupConfig<SetupValidateResponse>("/api/setup/validate", config);
}

export async function activateSetupConfig(
  config: object,
): Promise<SetupActivateResponse> {
  return postSetupConfig<SetupActivateResponse>("/api/setup/activate", config);
}

export async function importRemoteSetupConfig(
  url: string,
): Promise<SetupImportRemoteResponse> {
  const response = await fetch("/api/setup/import-remote", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ url }),
  });

  if (!response.ok) {
    throw new Error(await readErrorMessage(response));
  }

  return response.json();
}
