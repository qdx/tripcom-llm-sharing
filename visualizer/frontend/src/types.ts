export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  tokenCount: number;
  label?: string;
  section?: string;
}

export type Mode = 'simple' | 'tool' | 'harness';

export interface VisualizerState {
  type: string;
  messages: ChatMessage[];
  tokenCount: number;
  maxTokens: number;
  mode: Mode;
  metadata?: {
    model: string;
    timestamp: string;
    latency_ms?: number;
  };
}
