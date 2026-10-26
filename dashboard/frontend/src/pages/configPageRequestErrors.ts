// A failed fetch carries the backend's reason in its body. Reading only the
// status line hides that reason behind a bare HTTP code, which is how the
// config load paths reported failures before the save path's body-parsing
// shape was extracted here for all three.
export async function responseErrorMessage(response: Response): Promise<string> {
  const errorText = await response.text()
  let errorMessage = `HTTP ${response.status}: ${response.statusText}`
  if (errorText) {
    try {
      const errorJson = JSON.parse(errorText)
      if (errorJson.error || errorJson.message) {
        errorMessage = errorJson.message || errorJson.error
      } else {
        errorMessage = errorText
      }
    } catch {
      // If not JSON, use the text as-is
      errorMessage = errorText
    }
  }
  return errorMessage
}
