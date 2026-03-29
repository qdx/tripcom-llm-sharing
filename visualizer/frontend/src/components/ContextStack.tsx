import { useEffect, useRef } from 'react';
import { ChatMessage, Mode } from '../types';
import { MessageBlock } from './MessageBlock';

interface ContextStackProps {
  messages: ChatMessage[];
  mode: Mode;
}

export function ContextStack({ messages, mode }: ContextStackProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages.length]);

  if (messages.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gray-600">
        <div className="text-center">
          <div className="text-4xl mb-4">{ }</div>
          <p className="text-lg">Waiting for messages...</p>
          <p className="text-sm mt-2">Run a demo script or send a request through the proxy</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-6 py-4 space-y-3">
      {messages.map((msg, i) => (
        <MessageBlock
          key={`${i}-${msg.role}-${msg.tokenCount}`}
          message={msg}
          mode={mode}
          animationDelay={i * 50}
        />
      ))}
      <div ref={bottomRef} />
    </div>
  );
}
