import { useState, useMemo, useEffect, useCallback } from 'react';
import { PCAData, PCAPoint, PCAView } from '../types';

interface PCAVisualizationProps {
  data: PCAData | null;
  activeLayer: number;
  onLayerChange?: (layer: number) => void;
}

// Color palette for token trajectories
const TOKEN_COLORS = [
  '#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24', '#6ab04c',
  '#e056fd', '#f0932b', '#7ed6df', '#badc58', '#e17055',
];

const COMPARISON_COLORS = {
  direct: '#ff6b6b',
  cot: '#4ecdc4',
};

const PADDING = 50;

function scalePoints(points: PCAPoint[], width: number, height: number, allPoints: PCAPoint[]) {
  const xs = allPoints.map(p => p.x);
  const ys = allPoints.map(p => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const rangeX = maxX - minX || 1;
  const rangeY = maxY - minY || 1;

  return points.map(p => ({
    ...p,
    sx: PADDING + ((p.x - minX) / rangeX) * (width - 2 * PADDING),
    sy: PADDING + ((p.y - minY) / rangeY) * (height - 2 * PADDING),
  }));
}

export function PCAVisualization({ data, activeLayer, onLayerChange }: PCAVisualizationProps) {
  const [view, setView] = useState<PCAView>('tokens');
  const [hoveredToken, setHoveredToken] = useState<number>(-1);
  const [animLayer, setAnimLayer] = useState<number>(-1);
  const [isAnimating, setIsAnimating] = useState(false);

  const effectiveLayer = isAnimating ? animLayer : activeLayer;

  const startAnimation = useCallback(() => {
    if (!data) return;
    setIsAnimating(true);
    setAnimLayer(0);
  }, [data]);

  useEffect(() => {
    if (!isAnimating || !data) return;
    if (animLayer > data.nLayers) {
      setIsAnimating(false);
      setAnimLayer(-1);
      return;
    }
    const timer = setTimeout(() => {
      setAnimLayer(prev => prev + 1);
    }, 400);
    return () => clearTimeout(timer);
  }, [isAnimating, animLayer, data]);

  const svgWidth = 560;
  const svgHeight = 420;

  const tokenView = useMemo(() => {
    if (!data) return null;
    const { trajectories } = data.tokenTrajectories;
    const allPoints = trajectories.flatMap(t => t.points);
    return trajectories.map(t => ({
      ...t,
      scaledPoints: scalePoints(t.points, svgWidth, svgHeight, allPoints),
    }));
  }, [data]);

  const comparisonView = useMemo(() => {
    if (!data) return null;
    const { direct, cot } = data.comparison;
    const allPoints = [...direct.points, ...cot.points];
    return {
      direct: { ...direct, scaledPoints: scalePoints(direct.points, svgWidth, svgHeight, allPoints) },
      cot: { ...cot, scaledPoints: scalePoints(cot.points, svgWidth, svgHeight, allPoints) },
    };
  }, [data]);

  if (!data) {
    return (
      <div className="flex-1 flex items-center justify-center bg-gray-900/50 rounded-lg border border-gray-800">
        <div className="text-center text-gray-500">
          <div className="text-3xl mb-3">🧠</div>
          <p className="text-sm">PCA trajectory data not loaded</p>
          <p className="text-xs mt-1 text-gray-600">Run generate_pca_data.py first</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Controls */}
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-800">
        <div className="flex gap-2">
          <button
            onClick={() => setView('tokens')}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              view === 'tokens' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Token Trajectories
          </button>
          <button
            onClick={() => setView('comparison')}
            className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
              view === 'comparison' ? 'bg-purple-600 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            Direct vs CoT
          </button>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={startAnimation}
            disabled={isAnimating}
            className="px-3 py-1.5 text-xs font-medium rounded-lg bg-gray-800 text-gray-400 hover:bg-gray-700 disabled:opacity-50 transition-colors"
          >
            {isAnimating ? `Layer ${animLayer}/${data.nLayers}` : '▶ Animate'}
          </button>
        </div>
      </div>

      {/* Model info */}
      <div className="px-4 py-2 text-xs text-gray-500 flex items-center gap-4 border-b border-gray-800/50">
        <span>Model: <span className="text-gray-400 font-mono">{data.model}</span></span>
        <span>{data.nLayers} layers</span>
        <span>{data.dModel}-dim</span>
        {view === 'tokens' && (
          <span className="text-purple-400">
            Variance: {(data.tokenTrajectories.pcaVariance[0] * 100).toFixed(1)}% + {(data.tokenTrajectories.pcaVariance[1] * 100).toFixed(1)}%
          </span>
        )}
        {view === 'comparison' && (
          <span className="text-purple-400">
            Path ratio: {data.comparison.pathRatio}×
          </span>
        )}
      </div>

      {/* SVG Canvas */}
      <div className="flex-1 flex items-center justify-center p-4">
        <svg
          viewBox={`0 0 ${svgWidth} ${svgHeight}`}
          className="w-full h-full max-h-[420px]"
          style={{ background: 'rgba(0,0,0,0.3)', borderRadius: '8px' }}
        >
          {/* Grid lines */}
          <defs>
            <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
              <path d="M 40 0 L 0 0 0 40" fill="none" stroke="rgba(255,255,255,0.03)" strokeWidth="0.5" />
            </pattern>
          </defs>
          <rect width={svgWidth} height={svgHeight} fill="url(#grid)" />

          {/* Axis labels */}
          <text x={svgWidth / 2} y={svgHeight - 8} textAnchor="middle" fill="#666" fontSize="10" fontFamily="monospace">
            PC1 {view === 'tokens' 
              ? `(${(data.tokenTrajectories.pcaVariance[0] * 100).toFixed(1)}%)`
              : `(${(data.comparison.pcaVariance[0] * 100).toFixed(1)}%)`
            }
          </text>
          <text x={12} y={svgHeight / 2} textAnchor="middle" fill="#666" fontSize="10" fontFamily="monospace"
            transform={`rotate(-90, 12, ${svgHeight / 2})`}
          >
            PC2
          </text>

          {view === 'tokens' && tokenView && tokenView.map((traj, tIdx) => {
            const color = TOKEN_COLORS[tIdx % TOKEN_COLORS.length];
            const isHighlighted = hoveredToken === -1 || hoveredToken === tIdx;
            const opacity = isHighlighted ? 1 : 0.15;
            const visiblePoints = effectiveLayer === -1
              ? traj.scaledPoints
              : traj.scaledPoints.filter(p => p.layer <= effectiveLayer);
            
            if (visiblePoints.length === 0) return null;
            
            const pathD = visiblePoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.sx} ${p.sy}`).join(' ');
            
            return (
              <g key={tIdx} opacity={opacity} style={{ transition: 'opacity 0.3s' }}>
                {/* Trajectory line */}
                <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
                
                {/* Layer points */}
                {visiblePoints.map((p, i) => (
                  <circle
                    key={i}
                    cx={p.sx}
                    cy={p.sy}
                    r={i === 0 ? 5 : i === visiblePoints.length - 1 ? 6 : 3}
                    fill={i === 0 ? 'transparent' : color}
                    stroke={color}
                    strokeWidth={i === 0 || i === visiblePoints.length - 1 ? 2 : 1}
                  />
                ))}
                
                {/* Token label at last visible point */}
                {visiblePoints.length > 0 && (
                  <text
                    x={visiblePoints[visiblePoints.length - 1].sx + 8}
                    y={visiblePoints[visiblePoints.length - 1].sy - 6}
                    fill={color}
                    fontSize="11"
                    fontFamily="monospace"
                    fontWeight="bold"
                  >
                    {traj.token}
                  </text>
                )}
              </g>
            );
          })}

          {view === 'comparison' && comparisonView && (() => {
            const { direct, cot } = comparisonView;
            
            const renderTrajectory = (
              traj: typeof direct,
              color: string,
              label: string,
            ) => {
              const visiblePoints = effectiveLayer === -1
                ? traj.scaledPoints
                : traj.scaledPoints.filter(p => p.layer <= effectiveLayer);
              
              if (visiblePoints.length === 0) return null;
              
              const pathD = visiblePoints.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.sx} ${p.sy}`).join(' ');
              
              return (
                <g>
                  <path d={pathD} fill="none" stroke={color} strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" />
                  {visiblePoints.map((p, i) => (
                    <g key={i}>
                      <circle
                        cx={p.sx}
                        cy={p.sy}
                        r={i === 0 ? 7 : i === visiblePoints.length - 1 ? 8 : 4}
                        fill={i === 0 ? 'transparent' : color}
                        stroke={i === 0 || i === visiblePoints.length - 1 ? 'white' : color}
                        strokeWidth={i === 0 || i === visiblePoints.length - 1 ? 2 : 1}
                      />
                      {i === 0 && (
                        <text x={p.sx + 10} y={p.sy + 4} fill={color} fontSize="10" fontFamily="monospace">
                          {label} (L0)
                        </text>
                      )}
                      {i === visiblePoints.length - 1 && effectiveLayer === -1 && (
                        <text x={p.sx + 10} y={p.sy + 4} fill={color} fontSize="10" fontFamily="monospace">
                          L{data.nLayers} ★
                        </text>
                      )}
                    </g>
                  ))}
                </g>
              );
            };
            
            return (
              <>
                {renderTrajectory(direct, COMPARISON_COLORS.direct, 'Direct')}
                {renderTrajectory(cot, COMPARISON_COLORS.cot, 'CoT')}
              </>
            );
          })()}

          {/* Legend */}
          {view === 'comparison' && (
            <g transform={`translate(${svgWidth - 160}, 20)`}>
              <rect x="-10" y="-5" width="155" height="55" rx="6" fill="rgba(0,0,0,0.6)" stroke="rgba(255,255,255,0.1)" />
              <circle cx="8" cy="10" r="5" fill={COMPARISON_COLORS.direct} />
              <text x="20" y="14" fill="#ccc" fontSize="11" fontFamily="monospace">
                Direct (path={data.comparison.direct.pathLength})
              </text>
              <circle cx="8" cy="30" r="5" fill={COMPARISON_COLORS.cot} />
              <text x="20" y="34" fill="#ccc" fontSize="11" fontFamily="monospace">
                CoT (path={data.comparison.cot.pathLength})
              </text>
            </g>
          )}
        </svg>
      </div>

      {/* Token legend (for token view) */}
      {view === 'tokens' && tokenView && (
        <div className="px-4 py-3 border-t border-gray-800 flex flex-wrap gap-2">
          {tokenView.map((traj, tIdx) => (
            <button
              key={tIdx}
              onMouseEnter={() => setHoveredToken(tIdx)}
              onMouseLeave={() => setHoveredToken(-1)}
              className="px-2 py-1 rounded text-xs font-mono transition-all"
              style={{
                backgroundColor: `${TOKEN_COLORS[tIdx % TOKEN_COLORS.length]}20`,
                color: TOKEN_COLORS[tIdx % TOKEN_COLORS.length],
                border: `1px solid ${TOKEN_COLORS[tIdx % TOKEN_COLORS.length]}40`,
                opacity: hoveredToken === -1 || hoveredToken === tIdx ? 1 : 0.3,
              }}
            >
              {traj.token}
            </button>
          ))}
        </div>
      )}

      {/* Layer slider */}
      <div className="px-4 py-3 border-t border-gray-800">
        <div className="flex items-center gap-3">
          <span className="text-xs text-gray-500 w-16">
            {effectiveLayer === -1 ? 'All' : `Layer ${effectiveLayer}`}
          </span>
          <input
            type="range"
            min="-1"
            max={data.nLayers}
            value={effectiveLayer}
            onChange={(e) => {
              setIsAnimating(false);
              onLayerChange?.(parseInt(e.target.value));
            }}
            className="flex-1 h-1.5 bg-gray-700 rounded-lg appearance-none cursor-pointer accent-purple-500"
          />
          <span className="text-xs text-gray-500 font-mono">
            embed → L{data.nLayers}
          </span>
        </div>
      </div>
    </div>
  );
}
