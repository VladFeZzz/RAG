import { useEffect, useRef } from "react";
import type { ChatMessage } from "../types";

type ChatWindowProps = {
  messages: ChatMessage[];
};

export function ChatWindow({ messages }: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!scrollRef.current) {
      return;
    }

    scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
  }, [messages]);

  return (
    <section className="chat-window" ref={scrollRef} aria-label="Chat messages">
      {messages.map((message) => (
        <article key={message.id} className={`message message-${message.role}`}>
          <div className={`bubble ${message.pending ? "pending" : ""}`}>
            {message.text}
          </div>
        </article>
      ))}
    </section>
  );
}
