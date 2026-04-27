import { useState } from "react";
import type { FormEvent } from "react";

type MessageComposerProps = {
  disabled?: boolean;
  onSend: (message: string) => Promise<void>;
};

export function MessageComposer({
  disabled = false,
  onSend,
}: MessageComposerProps) {
  const [value, setValue] = useState("");

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const message = value.trim();
    if (!message || disabled) {
      return;
    }

    setValue("");
    await onSend(message);
  }

  return (
    <form className="composer" onSubmit={handleSubmit}>
      <input
        className="composer-input"
        type="text"
        placeholder="Напиши своє питання..."
        value={value}
        onChange={(event) => setValue(event.target.value)}
        disabled={disabled}
      />
      <button
        className="composer-button"
        type="submit"
        disabled={disabled || !value.trim()}
      >
        {disabled ? "Зачекай..." : "Send"}
      </button>
    </form>
  );
}
