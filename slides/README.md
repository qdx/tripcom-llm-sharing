# Slide Deck — From Prompt Engineering to Harness Engineering

## Quick Start

```bash
# Option 1: Open directly in browser
open slides/index.html

# Option 2: Serve locally (for proper CDN loading)
cd slides && python3 -m http.server 8888
# Then open http://localhost:8888
```

## Structure (1 hour)

| Time  | Section | Duration |
|-------|---------|----------|
| 0:00  | Opening — The Three Eras | 5 min |
| 0:05  | Timeline — GPT-2 to Harness Engineering | 10 min |
| 0:15  | **Demo 1** — Context Window Visualizer (Simple Mode) | 10 min |
| 0:25  | **Demo 2** — ReAct Agent (Tool Calling Mode) | 12 min |
| 0:37  | **Demo 3** — Full Agent Harness (Harness Mode) | 13 min |
| 0:50  | Key Takeaways for Engineers | 5 min |
| 0:55  | Q&A | 5 min |

## Technology

- **reveal.js 5.1.0** via CDN (no build step needed)
- Dark theme matching context window visualizer aesthetic
- Color palette: indigo (system), green (user), amber (assistant), rose (tool_call), orange (tool)

## Keyboard Shortcuts

- `→` / `Space` — next slide
- `←` — previous slide
- `Esc` — overview mode
- `S` — speaker notes view (open in separate window)
- `B` — black screen (pause)
- `F` — fullscreen

## Design Decisions

1. **HTML + reveal.js** — portable, no dependencies, works offline after first CDN load
2. **Context window visual language** — consistent block-and-color system from visualizer
3. **Progressive disclosure** — fragments reveal content on click for pacing
4. **Dark theme** — projector-friendly, matches all demo tools
5. **Bilingual hints** — Chinese subtitle on title slide for audience comfort

## Demo Integration

During demos, switch to the Context Window Visualizer (`localhost:3000`).
The slides include transition slides before/after each demo to provide context.

## Customization

Edit `index.html` directly. Key CSS variables are at the top in `:root`.
All content is in standard reveal.js `<section>` blocks.
