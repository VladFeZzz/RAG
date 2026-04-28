import { useRef, useState } from "react";
import type { ChangeEvent, DragEvent, KeyboardEvent } from "react";

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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [dragActive, setDragActive] = useState(false);
  const [localError, setLocalError] = useState("");

  function formatFileSize(bytes: number): string {
    if (bytes < 1024) {
      return `${bytes} B`;
    }
    if (bytes < 1024 * 1024) {
      return `${(bytes / 1024).toFixed(1)} KB`;
    }
    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
  }

  function selectFile(nextFile: File | null) {
    if (!nextFile) {
      setFile(null);
      setLocalError("");
      return;
    }

    const isPdf =
      nextFile.type === "application/pdf" ||
      nextFile.name.toLowerCase().endsWith(".pdf");

    if (!isPdf) {
      setLocalError("Потрібно обрати PDF-файл.");
      return;
    }

    setLocalError("");
    setFile(nextFile);
  }

  function openFilePicker() {
    if (disabled) {
      return;
    }
    fileInputRef.current?.click();
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    selectFile(event.target.files?.[0] || null);
  }

  function handleDragOver(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (!disabled) {
      setDragActive(true);
    }
  }

  function handleDragLeave(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
  }

  function handleDrop(event: DragEvent<HTMLDivElement>) {
    event.preventDefault();
    setDragActive(false);
    if (disabled) {
      return;
    }
    selectFile(event.dataTransfer.files?.[0] || null);
  }

  function handleDropzoneKeyDown(event: KeyboardEvent<HTMLDivElement>) {
    if (event.key !== "Enter" && event.key !== " ") {
      return;
    }
    event.preventDefault();
    openFilePicker();
  }

  async function handleUpload() {
    if (!file || disabled) {
      return;
    }

    await onUpload(file);
  }

  function clearSelection() {
    setFile(null);
    setLocalError("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  }

  const statusClass = localError ? "upload-error" : `upload-${statusType}`;
  const statusMessage = localError || statusText;

  return (
    <section className="upload-panel">
      <div
        className={`upload-dropzone ${dragActive ? "drag-active" : ""} ${disabled ? "is-disabled" : ""}`}
        role="button"
        tabIndex={disabled ? -1 : 0}
        onClick={openFilePicker}
        onKeyDown={handleDropzoneKeyDown}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        aria-label="Зона завантаження PDF"
      >
        <p className="dropzone-title">
          Перетягни PDF сюди або натисни для вибору
        </p>
        <p className="dropzone-hint">Підтримується OCR та звичайний текст</p>
        <input
          ref={fileInputRef}
          id="pdfInput"
          className="upload-file-input"
          type="file"
          accept=".pdf"
          onChange={handleFileChange}
          disabled={disabled}
        />
      </div>

      <div className="upload-row">
        <span className="file-name">
          {file
            ? `${file.name} (${formatFileSize(file.size)})`
            : "Файл не обрано"}
        </span>
        <button
          className="upload-clear"
          type="button"
          onClick={clearSelection}
          disabled={!file || disabled}
        >
          Очистити
        </button>
        <button
          className="upload-button"
          type="button"
          onClick={handleUpload}
          disabled={!file || disabled}
        >
          {disabled ? "Обробка..." : "Завантажити PDF"}
        </button>
      </div>
      <p className={`upload-status ${statusClass}`}>{statusMessage}</p>
    </section>
  );
}
