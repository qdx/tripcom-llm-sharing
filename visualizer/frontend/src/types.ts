export interface ToolCall {
  id: string;
  type: 'function';
  function: {
    name: string;
    arguments: string;
  };
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool' | 'subagent';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
  name?: string;
  tokenCount: number;
  label?: string;
  section?: string;
  /** Harness category for segmented visualization */
  category?: 'core' | 'memory' | 'skills' | 'tools' | 'runtime' | 'subagent' | 'conversation';
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

// PCA Visualization types
export interface PCAPoint {
  layer: number;
  x: number;
  y: number;
}

export interface PCATrajectory {
  token: string;
  tokenRaw: string;
  index: number;
  points: PCAPoint[];
}

export interface PCAComparisonTrajectory {
  prompt: string;
  tokenCount: number;
  pathLength: number;
  points: PCAPoint[];
}

export interface PCAData {
  model: string;
  nLayers: number;
  dModel: number;
  tokenTrajectories: {
    prompt: string;
    tokens: string[];
    trajectories: PCATrajectory[];
    pcaVariance: [number, number];
  };
  comparison: {
    direct: PCAComparisonTrajectory;
    cot: PCAComparisonTrajectory;
    pathRatio: number;
    pcaVariance: [number, number];
  };
}

export type PCAView = 'tokens' | 'comparison';

export interface PCAVisualizerState {
  data: PCAData | null;
  view: PCAView;
  activeLayer: number;          // -1 = all layers
  highlightedToken: number;     // -1 = none
  animating: boolean;
}

/** Token usage breakdown by category */
export interface TokenBreakdown {
  core: number;
  memory: number;
  skills: number;
  tools: number;
  runtime: number;
  subagent: number;
  conversation: number;
}
