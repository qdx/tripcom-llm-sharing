import { ChatMessage, TokenBreakdown, Mode } from '../types';

interface TokenCounterProps {
  tokenCount: number;
  maxTokens: number;
  messages?: ChatMessage[];
  mode?: Mode;
}

const CATEGORY_COLORS: Record<string, { bar: string; text: string; label: string }> = {
  core: { bar: 'bg-indigo-600', text: 'text-indigo-400', label: 'System Prompt' },
  memory: { bar: 'bg-purple-600', text: 'text-purple-400', label: 'Memory Docs' },
  skills: { bar: 'bg-cyan-600', text: 'text-cyan-400', label: 'Skills' },
  tools: { bar: 'bg-pink-600', text: 'text-pink-400', label: 'Tool Defs' },
  runtime: { bar: 'bg-yellow-600', text: 'text-yellow-400', label: 'Runtime' },
  subagent: { bar: 'bg-violet-600', text: 'text-violet-400', label: 'Sub-agents' },
  conversation: { bar: 'bg-emerald-600', text: 'text-emerald-400', label: 'Conversation' },
};

function computeBreakdown(messages: ChatMessage[]): TokenBreakdown {
  const bd: TokenBreakdown = { core: 0, memory: 0, skills: 0, tools: 0, runtime: 0, subagent: 0, conversation: 0 };
  for (const m of messages) {
    const cat = m.category || 'conversation';
    if (cat in bd) {
      bd[cat as keyof TokenBreakdown] += m.tokenCount;
    } else {
      bd.conversation += m.tokenCount;
    }
  }
  return bd;
}

export function TokenCounter({ tokenCount, maxTokens, messages = [], mode }: TokenCounterProps) {
  const pct = Math.min((tokenCount / maxTokens) * 100, 100);
  const showBreakdown = mode === 'harness' && messages.length > 0;
  const breakdown = showBreakdown ? computeBreakdown(messages) : null;
  const totalTokens = tokenCount || 0;

  if (!showBreakdown) {
    const color = pct > 80 ? 'bg-red-500' : pct > 50 ? 'bg-amber-500' : 'bg-emerald-500';
    return (
      <div className="px-6 py-4 border-b border-gray-800">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-400">Context Window</span>
          <span className="text-sm font-mono text-gray-300">
            {tokenCount.toLocaleString()} / {maxTokens.toLocaleString()} tokens
          </span>
        </div>
        <div className="h-3 bg-gray-800 rounded-full overflow-hidden">
          <div
            className={`h-full ${color} rounded-full transition-all duration-500 ease-out`}
            style={{ width: `${pct}%` }}
          />
        </div>
        <div className="mt-1 text-right text-xs text-gray-500">
          {pct.toFixed(1)}% used
        </div>
      </div>
    );
  }

  // Segmented bar for harness mode
  const segments = Object.entries(breakdown!).filter(([, v]) => v > 0);

  return (
    <div className="px-6 py-4 border-b border-gray-800">
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-gray-400">Context Window — Harness Breakdown</span>
        <span className="text-sm font-mono text-gray-300">
          {tokenCount.toLocaleString()} / {maxTokens.toLocaleString()} tokens
          <span className="text-gray-500 ml-2">({pct.toFixed(1)}%)</span>
        </span>
      </div>
      {/* Segmented bar */}
      <div className="h-4 bg-gray-800 rounded-full overflow-hidden flex">
        {segments.map(([cat, tokens]) => {
          const segPct = (tokens / maxTokens) * 100;
          const colors = CATEGORY_COLORS[cat] || CATEGORY_COLORS.conversation;
          return (
            <div
              key={cat}
              className={`${colors.bar} h-full transition-all duration-700 ease-out first:rounded-l-full last:rounded-r-full`}
              style={{ width: `${segPct}%` }}
              title={`${colors.label}: ${tokens.toLocaleString()} tokens`}
            />
          );
        })}
      </div>
      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1">
        {segments.map(([cat, tokens]) => {
          const colors = CATEGORY_COLORS[cat] || CATEGORY_COLORS.conversation;
          const segPct = totalTokens > 0 ? ((tokens / totalTokens) * 100).toFixed(1) : '0';
          return (
            <div key={cat} className="flex items-center gap-1.5">
              <div className={`w-2.5 h-2.5 rounded-sm ${colors.bar}`} />
              <span className={`text-xs ${colors.text}`}>
                {colors.label}
              </span>
              <span className="text-xs text-gray-500 font-mono">
                {tokens.toLocaleString()} ({segPct}%)
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
