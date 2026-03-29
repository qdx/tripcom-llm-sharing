interface TokenCounterProps {
  tokenCount: number;
  maxTokens: number;
}

export function TokenCounter({ tokenCount, maxTokens }: TokenCounterProps) {
  const pct = Math.min((tokenCount / maxTokens) * 100, 100);
  const color =
    pct > 80 ? 'bg-red-500' : pct > 50 ? 'bg-amber-500' : 'bg-emerald-500';

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
