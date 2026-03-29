import { useWebSocket } from './hooks/useWebSocket';
import { ContextStack } from './components/ContextStack';
import { TokenCounter } from './components/TokenCounter';
import { ModeSelector } from './components/ModeSelector';
import { Mode } from './types';

function App() {
  const { state, connected } = useWebSocket();

  const handleModeChange = async (mode: Mode) => {
    try {
      await fetch('http://localhost:8080/mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode }),
      });
    } catch {
      // proxy not available
    }
  };

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="px-6 py-4 border-b border-gray-800 flex items-center justify-between">
        <div>
          <h1 className="text-xl font-bold text-white">Context Window Visualizer</h1>
          <p className="text-sm text-gray-500">Real-time LLM context window visualization</p>
        </div>
        <div className="flex items-center gap-3">
          <div
            className={`w-2.5 h-2.5 rounded-full ${
              connected ? 'bg-emerald-500' : 'bg-red-500'
            }`}
            title={connected ? 'Connected' : 'Disconnected'}
          />
          <span className="text-xs text-gray-500">
            {connected ? 'Live' : 'Reconnecting...'}
          </span>
          <button
            onClick={async () => {
              try {
                await fetch('http://localhost:8080/reset', { method: 'POST' });
              } catch { /* ignore */ }
            }}
            className="px-3 py-1.5 bg-gray-800 hover:bg-gray-700 text-gray-300 text-sm rounded-lg transition-colors"
          >
            Reset
          </button>
        </div>
      </header>

      <ModeSelector mode={state.mode} onModeChange={handleModeChange} />
      <TokenCounter tokenCount={state.tokenCount} maxTokens={state.maxTokens} />
      <ContextStack messages={state.messages} mode={state.mode} />
    </div>
  );
}

export default App;
