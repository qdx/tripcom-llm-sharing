import { Mode } from '../types';

interface ModeSelectorProps {
  mode: Mode;
  onModeChange: (mode: Mode) => void;
}

const MODES: { value: Mode; label: string; description: string }[] = [
  { value: 'simple', label: 'Simple', description: 'Basic conversation' },
  { value: 'tool', label: 'Tool Calling', description: 'ReAct agent loop' },
  { value: 'harness', label: 'Harness', description: 'Full harness engineering' },
];

export function ModeSelector({ mode, onModeChange }: ModeSelectorProps) {
  return (
    <div className="flex gap-2 px-6 py-3 border-b border-gray-800">
      {MODES.map((m) => (
        <button
          key={m.value}
          onClick={() => onModeChange(m.value)}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
            mode === m.value
              ? 'bg-indigo-600 text-white'
              : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-200'
          }`}
          title={m.description}
        >
          {m.label}
        </button>
      ))}
    </div>
  );
}
