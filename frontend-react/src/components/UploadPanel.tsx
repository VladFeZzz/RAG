import { useState } from "react";

type UploadPanelProps = {
  disabled?: boolean;
  statusText: string;
  statusType: "idle" | "success" | "error" | "progress";
  onUpload: (file: File) => Promise<void>;
};

export function UploadPanel({
  disabled = false,
  statusText,
  statusType,
  onUpload,
}: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);

  async function handleUpload() {
    if (!file || disabled) {
      return;
    }

    await onUpload(file);
  }

  return (
    <section className="upload-panel">
      <div className="upload-row">
        <label className="upload-label" htmlFor="pdfInput">
          Обери PDF-файл
        </label>
        <input
          id="pdfInput"
          type="file"
          accept=".pdf"
          onChange={(event) => setFile(event.target.files?.[0] || null)}
          disabled={disabled}
        />
        <span className="file-name">{file?.name || "Файл не обрано"}</span>
        <button
          className="upload-button"
          type="button"
          onClick={handleUpload}
          disabled={!file || disabled}
        >
          {disabled ? "Обробка..." : "Завантажити в базу"}
        </button>
      </div>
      <p className={`upload-status upload-${statusType}`}>{statusText}</p>
    </section>
  );
}
