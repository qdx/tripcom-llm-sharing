import { useState } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import { ContextStack } from './components/ContextStack';
import { TokenCounter } from './components/TokenCounter';
import { ModeSelector } from './components/ModeSelector';
import { PCAVisualization } from './components/PCAVisualization';
import { Mode } from './types';

function App() {
  const { state, connected, pcaData } = useWebSocket();
  const [pcaLayer, setPcaLayer] = useState<number>(-1);

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

  const isSplitScreen = state.mode === 'simple' && pcaData !== null;

  return (
    <div className="h-screen flex flex-col bg-gray-950 text-gray-100">
      {/* Header */}
      <header className="px-6 py-4 border-b border-gray-800 flex items-center justify-between flex-shrink-0">
        <div>
          <h1 className="text-xl font-bold text-white">
            {isSplitScreen ? 'Demo 1: Context Window + Hidden States' : 'Context Window Visualizer'}
          </h1>
          <p className="text-sm text-gray-500">
            {isSplitScreen
              ? 'Split view — GPT-2 prompt context window & PCA trajectory of hidden states'
              : 'Real-time LLM context window visualization'
            }
          </p>
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

      {isSplitScreen ? (
        /* ===== SPLIT SCREEN LAYOUT (Demo 1) ===== */
        <>
          <TokenCounter tokenCount={state.tokenCount} maxTokens={state.maxTokens} />
          <div className="flex-1 flex min-h-0">
            {/* Left panel: Context Window */}
            <div className="w-1/2 flex flex-col border-r border-gray-800 min-h-0">
              <div className="px-4 py-2 border-b border-gray-800/50 flex-shrink-0">
                <h2 className="text-sm font-semibold text-gray-400 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500"></span>
                  Context Window
                </h2>
                <p className="text-xs text-gray-600 mt-0.5">Messages in the LLM's working memory</p>
              </div>
              <div className="flex-1 overflow-hidden">
                <ContextStack messages={state.messages} mode={state.mode} />
              </div>
            </div>

            {/* Right panel: PCA Visualization */}
            <div className="w-1/2 flex flex-col min-h-0">
              <div className="px-4 py-2 border-b border-gray-800/50 flex-shrink-0">
                <h2 className="text-sm font-semibold text-gray-400 flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-purple-500"></span>
                  Vector Space (PCA)
                </h2>
                <p className="text-xs text-gray-600 mt-0.5">Hidden state trajectories through transformer layers</p>
              </div>
              <div className="flex-1 overflow-hidden">
                <PCAVisualization
                  data={pcaData}
                  activeLayer={pcaLayer}
                  onLayerChange={setPcaLayer}
                />
              </div>
            </div>
          </div>
        </>
      ) : (
        /* ===== STANDARD LAYOUT ===== */
        <>
          <TokenCounter tokenCount={state.tokenCount} maxTokens={state.maxTokens} />
          <ContextStack messages={state.messages} mode={state.mode} />
        </>
      )}
    </div>
  );
}

export default App;
