import { useEffect, useRef, useState, useCallback } from 'react';
import { VisualizerState, PCAData } from '../types';

const WS_URL = 'ws://localhost:8080/ws';
const PCA_DATA_URL = 'http://localhost:8080/pca-data';
const RECONNECT_INTERVAL = 2000;

export function useWebSocket() {
  const [state, setState] = useState<VisualizerState>({
    type: 'state_update',
    messages: [],
    tokenCount: 0,
    maxTokens: 128_000,
    mode: 'simple',
  });
  const [pcaData, setPcaData] = useState<PCAData | null>(null);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>();

  // Fetch PCA data once on mount
  useEffect(() => {
    fetch(PCA_DATA_URL)
      .then(res => res.ok ? res.json() : null)
      .then(data => { if (data && !data.error) setPcaData(data); })
      .catch(() => { /* PCA data not available */ });
  }, []);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);

    ws.onopen = () => {
      setConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data: VisualizerState = JSON.parse(event.data);
        setState(data);
      } catch {
        // ignore parse errors
      }
    };

    ws.onclose = () => {
      setConnected(false);
      reconnectTimer.current = setTimeout(connect, RECONNECT_INTERVAL);
    };

    ws.onerror = () => {
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { state, connected, pcaData };
}
