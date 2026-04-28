const API_BASE = (import.meta.env.VITE_API_BASE || "").trim();

function toUrl(path: string): string {
  if (!API_BASE) {
    return path;
  }

  return `${API_BASE}${path}`;
}

export async function sendChatMessage(message: string): Promise<string> {
  const response = await fetch(toUrl("/chat"), {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ message }),
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.response || "Сталася помилка на сервері.");
  }

  if (!payload.response || typeof payload.response !== "string") {
    throw new Error("Сервер повернув некоректну відповідь.");
  }

  return payload.response;
}

export async function uploadPdf(file: File): Promise<string> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await fetch(toUrl("/upload"), {
    method: "POST",
    body: formData,
  });

  const payload = await response.json().catch(() => ({}));

  if (!response.ok) {
    throw new Error(payload.error || "Не вдалося завантажити PDF.");
  }

  return payload.message || "Файл успішно оброблено.";
}
