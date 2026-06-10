export async function parseError(response: Response): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data.detail === 'string') return data.detail;
    return JSON.stringify(data);
  } catch {
    return response.statusText || '요청에 실패했어.';
  }
}
