import { useMemo, useState } from "react";
import { sendChatMessage, uploadPdf } from "./api/client";
import { ChatWindow } from "./components/ChatWindow";
import { MessageComposer } from "./components/MessageComposer";
import { UploadPanel } from "./components/UploadPanel";
import type { ChatMessage } from "./types";

function makeId(): string {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: makeId(),
      role: "bot",
      text: "Привіт! Я готовий відповідати на питання. Завантаж PDF зверху або просто запитай щось про музику.",
    },
  ]);
  const [busy, setBusy] = useState(false);
  const [uploadStatus, setUploadStatus] = useState({
    type: "idle" as "idle" | "success" | "error" | "progress",
    text: "Підтримується PDF з OCR та звичайним текстом.",
  });

  const subtitle = useMemo(() => {
    return busy ? "Бот формує відповідь..." : "Music RAG Assistant";
  }, [busy]);

  async function handleSend(message: string) {
    const userMsg: ChatMessage = { id: makeId(), role: "user", text: message };
    const pendingId = makeId();

    setMessages((prev) => [
      ...prev,
      userMsg,
      {
        id: pendingId,
        role: "bot",
        text: "Думаю над відповіддю...",
        pending: true,
      },
    ]);
    setBusy(true);

    try {
      const response = await sendChatMessage(message);
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === pendingId
            ? { ...msg, text: response, pending: false }
            : msg,
        ),
      );
    } catch (error) {
      const errorText =
        error instanceof Error
          ? error.message
          : "Помилка зʼєднання з бекендом.";
      setMessages((prev) =>
        prev.map((msg) =>
          msg.id === pendingId
            ? { ...msg, text: errorText, pending: false }
            : msg,
        ),
      );
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload(file: File) {
    setUploadStatus({
      type: "progress",
      text: `Завантажую ${file.name} та індексую в базу...`,
    });
    setBusy(true);

    try {
      const result = await uploadPdf(file);
      setUploadStatus({ type: "success", text: `Готово: ${result}` });
    } catch (error) {
      const errorText =
        error instanceof Error
          ? error.message
          : "Невідома помилка при завантаженні.";
      setUploadStatus({ type: "error", text: `Помилка: ${errorText}` });
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <h1>Music AI Assistant</h1>
        <p>{subtitle}</p>
      </header>

      <UploadPanel
        disabled={busy}
        statusText={uploadStatus.text}
        statusType={uploadStatus.type}
        onUpload={handleUpload}
      />

      <ChatWindow messages={messages} />

      <MessageComposer disabled={busy} onSend={handleSend} />
    </main>
  );
}

export default App;
