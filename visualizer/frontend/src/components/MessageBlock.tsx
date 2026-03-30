import { useState } from 'react';
import { ChatMessage, Mode } from '../types';

interface MessageBlockProps {
  message: ChatMessage;
  mode: Mode;
  animationDelay: number;
}

const ROLE_STYLES: Record<string, { bg: string; border: string; badge: string; badgeText: string }> = {
  system: {
    bg: 'bg-indigo-950/40',
    border: 'border-indigo-500/50',
    badge: 'bg-indigo-600',
    badgeText: 'SYSTEM',
  },
  user: {
    bg: 'bg-emerald-950/40',
    border: 'border-emerald-500/50',
    badge: 'bg-emerald-600',
    badgeText: 'USER',
  },
  assistant: {
    bg: 'bg-amber-950/40',
    border: 'border-amber-500/50',
    badge: 'bg-amber-600',
    badgeText: 'ASSISTANT',
  },
  tool: {
    bg: 'bg-orange-950/40',
    border: 'border-orange-500/50',
    badge: 'bg-orange-600',
    badgeText: 'TOOL RESULT',
  },
  subagent: {
    bg: 'bg-violet-950/40',
    border: 'border-violet-500/50',
    badge: 'bg-violet-600',
    badgeText: 'SUB-AGENT',
  },
};

/** Category-based overrides for harness mode */
const CATEGORY_STYLES: Record<string, { bg: string; border: string; badge: string; badgeText: string }> = {
  core: {
    bg: 'bg-indigo-950/40',
    border: 'border-indigo-500/50',
    badge: 'bg-indigo-600',
    badgeText: 'SYSTEM',
  },
  memory: {
    bg: 'bg-purple-950/40',
    border: 'border-purple-500/50',
    badge: 'bg-purple-600',
    badgeText: 'MEMORY',
  },
  skills: {
    bg: 'bg-cyan-950/40',
    border: 'border-cyan-500/50',
    badge: 'bg-cyan-600',
    badgeText: 'SKILL',
  },
  tools: {
    bg: 'bg-pink-950/40',
    border: 'border-pink-500/50',
    badge: 'bg-pink-600',
    badgeText: 'TOOL DEF',
  },
  runtime: {
    bg: 'bg-yellow-950/40',
    border: 'border-yellow-500/50',
    badge: 'bg-yellow-600',
    badgeText: 'RUNTIME',
  },
  subagent: {
    bg: 'bg-violet-950/40',
    border: 'border-violet-500/50',
    badge: 'bg-violet-600',
    badgeText: 'SUB-AGENT',
  },
};

function getToolCallStyle() {
  return {
    bg: 'bg-rose-950/40',
    border: 'border-rose-500/50',
    badge: 'bg-rose-600',
    badgeText: 'TOOL CALL',
  };
}

function blockHeight(tokenCount: number): number {
  // Proportional height: min 60px, max 500px, scales with tokens
  const base = 60;
  const scale = Math.log2(Math.max(tokenCount, 1) + 1) * 25;
  return Math.min(Math.max(base, base + scale), 500);
}

function truncate(text: string | null, maxLen: number): string {
  if (!text) return '';
  if (text.length <= maxLen) return text;
  return text.slice(0, maxLen) + '...';
}

export function MessageBlock({ message, mode, animationDelay }: MessageBlockProps) {
  const [expanded, setExpanded] = useState(false);

  // In harness mode, use category-based styles if available
  const style =
    mode === 'harness' && message.category && CATEGORY_STYLES[message.category]
      ? CATEGORY_STYLES[message.category]
      : ROLE_STYLES[message.role] || ROLE_STYLES.user;

  const height = expanded ? undefined : blockHeight(message.tokenCount);
  const isExpandable = (mode === 'harness' && message.role === 'system') || (message.content && message.content.length > 300);
  const content = message.content || '';

  return (
    <div
      className="message-block-enter"
      style={{ animationDelay: `${animationDelay}ms` }}
    >
      <div
        className={`${style.bg} ${style.border} border rounded-lg p-4 transition-all duration-300 overflow-hidden`}
        style={height != null ? { minHeight: `${height}px` } : undefined}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <span className={`${style.badge} text-white text-xs font-bold px-2 py-0.5 rounded`}>
              {message.label || style.badgeText}
            </span>
            {message.name && (
              <span className="text-xs text-gray-400 font-mono">{message.name}</span>
            )}
            {message.section && (
              <span className="text-xs text-gray-500 italic ml-1">— {message.section}</span>
            )}
          </div>
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono text-gray-500">
              {message.tokenCount.toLocaleString()} tokens
            </span>
            {isExpandable && (
              <button
                onClick={() => setExpanded(!expanded)}
                className="text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
              >
                {expanded ? '▲ Collapse' : '▼ Expand'}
              </button>
            )}
          </div>
        </div>

        {/* Content */}
        <pre className="text-sm text-gray-300 whitespace-pre-wrap font-mono leading-relaxed">
          {expanded ? content : truncate(content, 300)}
        </pre>

        {/* Tool calls within assistant messages */}
        {message.tool_calls && message.tool_calls.length > 0 && (
          <div className="mt-3 space-y-2">
            {message.tool_calls.map((tc, i) => {
              const tcStyle = getToolCallStyle();
              return (
                <div
                  key={tc.id || i}
                  className={`${tcStyle.bg} ${tcStyle.border} border rounded p-3`}
                >
                  <div className="flex items-center gap-2 mb-1">
                    <span className={`${tcStyle.badge} text-white text-xs font-bold px-2 py-0.5 rounded`}>
                      {tcStyle.badgeText}
                    </span>
                    <span className="text-xs text-gray-300 font-mono">
                      {tc.function.name}
                    </span>
                  </div>
                  <pre className="text-xs text-gray-400 whitespace-pre-wrap font-mono">
                    {truncate(tc.function.arguments, 200)}
                  </pre>
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
