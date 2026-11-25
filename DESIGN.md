---
# DESIGN.md for the entity resolver review report. Tokens above the line are the only values UI code may use; prose below the line is the intent the tokens cannot carry.
name: "entity-resolver"
version: 1
base_unit: 4
color:
  primary:
    100: "#dbe8fb"
    300: "#8db6f0"
    500: "#1f6fd6"
    700: "#1553a6"
    900: "#0c2f61"
  neutral:
    50: "#f3f5f8"
    100: "#fbfcfd"
    300: "#d5dbe3"
    500: "#8a94a3"
    700: "#556070"
    900: "#161b22"
  accent:
    500: "#0a7d6e"
  mark:
    agree: "#fff1b8"
  dark:
    background: "#12161c"
    surface: "#1a1f27"
    border: "#2f3743"
    text-muted: "#9aa5b4"
    text: "#e8ecf1"
    action: "#5a9cf0"
    action-soft: "#0c2f61"
    agree: "#5c4a00"
    focus: "#3fbfae"
    success: "#4cc26a"
    warning: "#e0a530"
    danger: "#f0706b"
  semantic:
    background: neutral.50
    surface: neutral.100
    text: neutral.900
    text-muted: neutral.700
    border: neutral.300
    action: primary.500
    action-hover: primary.700
    action-text: neutral.50
    focus: accent.500
    success: "#1a7f37"
    warning: "#9a6700"
    danger: "#c9302c"
type:
  display:
    family: "system-ui, -apple-system, 'Segoe UI', sans-serif"
    weights: [650]
  body:
    family: "system-ui, -apple-system, 'Segoe UI', sans-serif"
    weights: [400, 600]
  scale:
    xs: 12
    sm: 14
    md: 16
    lg: 20
    xl: 24
    2xl: 30
    3xl: 36
    4xl: 48
  line_length_max_ch: 75
  line_height:
    body: 1.5
    heading: 1.15
spacing:
  scale: [4, 8, 12, 16, 24, 32, 48, 64, 96]
radius:
  sm: 4
  md: 8
  lg: 12
  full: 9999
shadow:
  sm: "none"
  md: "none"
motion:
  hover_ms: 120
  enter_ms: 240
  exit_ms: 160
  easing: "cubic-bezier(0.2, 0, 0, 1)"
  reduced_motion: required
breakpoints:
  phone: 390
  laptop: 1280
  wide: 1920
icons: "none; the report uses no icons"
---

# Entity resolver review report design

One screen: `resolver/report_template.html`, filled by `resolver/report.py` and written to `out/report.html` by every run. Two readers: a Fundable engineer judging the case study, and a reviewer working the queue of uncertain decisions.

## The one bold element

The comparison: the incoming record and the database record set side by side, with the words they share highlighted in `mark.agree`. Agreement is seen before anything is read. The verdict sentence ("New entity", "Same as Accel", "New, repeat of Ola Inc") with its confidence figure sits above it at heading size and is the only other large element. The highlight color is used nowhere else.

## What this product must never look like

- A dashboard that opens with a row of big-number tiles. The run summary is one line of text.
- A wall of identical cards. Rows are a list; one row at a time gets the full comparison.
- A database record hidden behind a disclosure. The two records are always visible together.
- A decision hidden in a small badge while the entity name takes the heading weight.
- A page that needs the network. The file is opened from disk, so no web fonts, no CDN scripts, no remote images.
- Color as the only signal of state. Review flags and wrong marks carry words.
- A confidence shown as a bar with no number, or a number with no scale.

## References

- `F:\fundable\.scratch\problem.html` (the problem map made earlier in this project): taken, the neutral surfaces and direct labels on every mark; rejected, its stat-tile hero.
- https://openrefine.org/docs/manual/cellediting (Cluster and edit): taken, every suggested merge needs the reviewer's approval, one decision per group, and the result can be exported; rejected, a modal dialog as the whole tool.
- https://moj-analytical-services.github.io/splink/charts/waterfall_chart.html (Splink waterfall chart): taken, each piece of evidence shown with the two compared values; rejected, the additive waterfall, because this pipeline's signals are not additive weights.
- https://primer.style/product/getting-started/foundations/color-usage/ (GitHub Primer color usage): taken, one role per color (action for selection, warning for review, danger for wrong); rejected, the full eight-role set.

## Hierarchy

Three levels only, by size, weight and color together: the verdict line at `xl` and the two record names at `lg` in the display weight; body text and list names at `sm` in `text`; ids, scores, labels and hints at `xs` in `text-muted`. Groups are separated by spacing from the scale; hairlines separate list rows and the judgment bar only.

## Layout

A header line (title, one summary sentence, theme and export buttons) over two panes that each scroll: a 400 px queue (search, decision tabs, type tabs, progress, then rows sorted least certain first) and a detail pane whose content is capped at 1120 px (verdict, reason, the two records, other candidates, links, the judgment bar). Left aligned throughout. Below 860 px the queue is the page and a row opens the detail full width with a back button; the two records stack, incoming first; nothing scrolls sideways.

## States

Empty filter result says no rows match and names the search and tab as the cause. A file with no data says so and names the command that fills it. A row with no candidate says no database record came close enough to compare. A new entity with no link says why links are left blank. Long names and URLs wrap. Each row shows one of three review states in words: needs review, confirmed, marked wrong. There is no loading state: the data is embedded in the file.

## Copy voice

Plain and literal, addressed to the reviewer. Verbs: match, call new, confirm ("Decision is right"), mark wrong ("Decision is wrong"), export. Sentence case everywhere. Banned: any word that sells, and machine codes shown raw (`insufficient_evidence` is shown as "insufficient evidence").

## Decisions

- 2026-11-25: Direction changed to Plan A, side by side (shape file `.scratch/design-shape.md`), approved by Alex on the rendered prototype ("much better"). The bold element moves from the verdict line to the record comparison.
- 2026-11-25: Card grid dropped for a queue list plus a detail pane. This overrides the standing card-grid preference for this screen only: the job is comparing two records, and a grid cannot keep both visible.
- 2026-11-25: Light is the default theme whatever the system setting, with a toggle to dark (Alex asked for light). `color.dark` gains `action-soft` and `agree`.
- 2026-11-25: `color.mark.agree` added for the shared-word highlight. It is the only use of yellow.
- 2026-11-25: Motion: color transitions at `hover_ms` on rows and buttons; one entrance at `enter_ms` when the selected row changes (the pane rises 8 px, the confidence bar fills, the highlights light up last). Nothing animates on scroll or on marking a decision. All motion is off under reduced motion.
- 2026-11-25: Confirming a decision moves to the next row; marking wrong stays so the correct answer can be chosen from the candidates.
- 2026-11-25: System font stack for display and body. The report is one self-contained file opened from disk and the case study forbids fetching from anything but the entity's own site, so a web font cannot be loaded.
- 2026-11-25: No shadows. Surfaces are separated by a hairline border and a tinted background.
- 2026-11-25: `color.dark` added as a second set of semantic values, selected for the dark surface instead of derived by inversion.
- 2026-11-25: File created by the design skill from the harness template. The palette is PROPOSED, confirm; the direction and the references were settled later the same day (entries above).
