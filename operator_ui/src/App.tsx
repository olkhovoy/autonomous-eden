import { useCallback, useEffect, useMemo, useState } from "react";
import type {
  BroomLine,
  CandidateRow,
  CombinationScenarioPayload,
  DashboardFeed,
  FarmBroomLine,
  FarmDashboardFeed,
  FarmScenarioRow,
  ScenarioPayload,
} from "./types";

const CANDIDATE_FEED_URL = "/data/dashboard-feed.json";
const FARM_FEED_URL = "/data/farm-dashboard-feed.json";

const CLUSTER_COLORS = [
  "#0f8a8d",
  "#de6b48",
  "#5f0f40",
  "#4c6a92",
  "#9d4edd",
  "#52796f",
  "#f4a261",
  "#bc4749",
];

function colorForCluster(clusterId: string | null): string {
  if (!clusterId) {
    return "#58717a";
  }
  const digits = Number.parseInt(clusterId.replace(/\D+/g, ""), 10);
  const index = Number.isNaN(digits) ? 0 : digits % CLUSTER_COLORS.length;
  return CLUSTER_COLORS[index];
}

function colorForFarmLine(line: FarmBroomLine): string {
  if (line.status === "running") {
    return "#0f8a8d";
  }
  if (line.status === "failed") {
    return "#bc4749";
  }
  if (line.gate_pass) {
    return "#137f48";
  }
  if (line.status === "planned") {
    return "#6c757d";
  }
  return "#de6b48";
}

function formatMoney(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return new Intl.NumberFormat("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
    signDisplay: "exceptZero",
  }).format(value);
}

function formatPct(value: number | null | undefined, digits = 1): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return `${value.toFixed(digits)}%`;
}

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return "n/a";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return value;
  }
  return new Intl.DateTimeFormat("en-GB", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    timeZone: "UTC",
  }).format(date);
}

function compact(value: number | null | undefined, digits = 2): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function formatDurationSeconds(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) {
    return "n/a";
  }
  const seconds = Math.max(0, Math.round(value));
  if (seconds < 60) {
    return `${seconds}s`;
  }
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) {
    return `${minutes}m`;
  }
  const hours = Math.floor(minutes / 60);
  const remMinutes = minutes % 60;
  if (hours < 48) {
    return `${hours}h ${remMinutes}m`;
  }
  const days = Math.floor(hours / 24);
  const remHours = hours % 24;
  return `${days}d ${remHours}h`;
}

function secondsSinceTimestamp(value: string | null | undefined, nowMs: number): number | null {
  if (!value) {
    return null;
  }
  const timestampMs = new Date(value).getTime();
  if (Number.isNaN(timestampMs)) {
    return null;
  }
  return Math.max(0, (nowMs - timestampMs) / 1000);
}

function deriveHeartbeatState(secondsSinceLastEvent: number | null, medianGapSeconds: number | null): string {
  if (secondsSinceLastEvent == null) {
    return "unknown";
  }
  if (medianGapSeconds == null || medianGapSeconds <= 0) {
    if (secondsSinceLastEvent <= 120) {
      return "fresh";
    }
    if (secondsSinceLastEvent <= 900) {
      return "watch";
    }
    return "stale";
  }
  const ratio = secondsSinceLastEvent / medianGapSeconds;
  if (ratio <= 1.5) {
    return "fresh";
  }
  if (ratio <= 4.0) {
    return "watch";
  }
  return "stale";
}

function deriveStagnationState(input: {
  secondsSinceLastCompletion: number | null;
  secondsSinceLastGatePass: number | null;
  medianCompletionGapSeconds: number | null;
  completionCount: number;
  gatePassCompletionCount: number;
}): string {
  if (input.completionCount === 0) {
    return "pre-first-completion";
  }
  if (input.gatePassCompletionCount === 0) {
    if (input.secondsSinceLastCompletion != null && input.secondsSinceLastCompletion <= 1800) {
      return "searching";
    }
    return "stagnating";
  }
  const threshold = Math.max(3600, (input.medianCompletionGapSeconds ?? 0) * 4);
  if (input.secondsSinceLastGatePass != null && input.secondsSinceLastGatePass > threshold) {
    return "stagnating";
  }
  return "healthy";
}

function useNowTick(periodMs: number): number {
  const [nowMs, setNowMs] = useState(() => Date.now());

  useEffect(() => {
    const intervalId = window.setInterval(() => {
      setNowMs(Date.now());
    }, periodMs);
    return () => window.clearInterval(intervalId);
  }, [periodMs]);

  return nowMs;
}

function candidateSortValue(candidate: CandidateRow, sortKey: string): number {
  if (sortKey === "oos") {
    return candidate.periods.oos?.pnl ?? Number.NEGATIVE_INFINITY;
  }
  if (sortKey === "robustness") {
    return candidate.resampling?.pessimistic_net_profit ?? Number.NEGATIVE_INFINITY;
  }
  if (sortKey === "drawdown") {
    return -(candidate.periods.oos?.max_drawdown_pct ?? Number.POSITIVE_INFINITY);
  }
  return candidate.shortlist.base_score ?? Number.NEGATIVE_INFINITY;
}

function farmSortValue(scenario: FarmScenarioRow, sortKey: string): number {
  if (sortKey === "drawdown") {
    return -(scenario.max_drawdown_pct ?? Number.POSITIVE_INFINITY);
  }
  if (sortKey === "cycles") {
    return scenario.evaluated_cycle_count ?? Number.NEGATIVE_INFINITY;
  }
  if (sortKey === "updated") {
    const date = scenario.updated_at_utc ? new Date(scenario.updated_at_utc) : null;
    return date && !Number.isNaN(date.getTime()) ? date.getTime() : Number.NEGATIVE_INFINITY;
  }
  if (sortKey === "gate") {
    return scenario.gate_pass ? 1 : 0;
  }
  return scenario.total_pnl ?? Number.NEGATIVE_INFINITY;
}

function useFeed(feedUrl: string): {
  feed: unknown;
  loading: boolean;
  refreshing: boolean;
  error: string | null;
  lastLoadedAt: string | null;
  reload: () => void;
} {
  const [feed, setFeed] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastLoadedAt, setLastLoadedAt] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const refreshMs = useMemo(() => {
    const url = new URL(window.location.href);
    const param = url.searchParams.get("refresh");
    if (!param) {
      return new URLSearchParams(window.location.search).get("view") === "farm" ? 5000 : 0;
    }
    const seconds = Number.parseFloat(param);
    if (!Number.isFinite(seconds) || seconds <= 0) {
      return 0;
    }
    return Math.round(seconds * 1000);
  }, []);

  const load = useCallback(async (cancelledRef?: { cancelled: boolean }) => {
    const initialLoad = feed === null;
    if (initialLoad) {
      setLoading(true);
    } else {
      setRefreshing(true);
    }
    setError(null);
    try {
      const cacheBust = `ts=${Date.now()}`;
      const separator = feedUrl.includes("?") ? "&" : "?";
      const response = await fetch(`${feedUrl}${separator}${cacheBust}`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Failed to load ${feedUrl}: ${response.status}`);
      }
      const payload = (await response.json()) as unknown;
      if (!cancelledRef?.cancelled) {
        setFeed(payload);
        setLastLoadedAt(new Date().toISOString());
      }
    } catch (err) {
      if (!cancelledRef?.cancelled) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (!cancelledRef?.cancelled) {
        setLoading(false);
        setRefreshing(false);
      }
    }
  }, [feed, feedUrl]);

  useEffect(() => {
    const cancelledRef = { cancelled: false };
    void load(cancelledRef);
    return () => {
      cancelledRef.cancelled = true;
    };
  }, [feedUrl, reloadToken, load]);

  useEffect(() => {
    if (refreshMs <= 0) {
      return;
    }
    const intervalId = window.setInterval(() => {
      void load();
    }, refreshMs);
    return () => window.clearInterval(intervalId);
  }, [load, refreshMs]);

  const reload = useCallback(() => {
    setReloadToken((value) => value + 1);
  }, []);

  return { feed, loading, refreshing, error, lastLoadedAt, reload };
}

function SummaryCard(props: {
  label: string;
  value: string | number;
  accent?: "teal" | "ember" | "ink";
  hint?: string;
}) {
  return (
    <article className={`summary-card accent-${props.accent ?? "ink"}`}>
      <div className="summary-label">{props.label}</div>
      <div className="summary-value">{props.value}</div>
      {props.hint ? <div className="summary-hint">{props.hint}</div> : null}
    </article>
  );
}

function FilterChip(props: {
  active: boolean;
  label: string;
  onClick: () => void;
}) {
  return (
    <button className={`filter-chip ${props.active ? "active" : ""}`} onClick={props.onClick}>
      {props.label}
    </button>
  );
}

function CandidateTable(props: {
  candidates: CandidateRow[];
  selectedId: string | null;
  onSelect: (candidateId: string) => void;
}) {
  return (
    <div className="panel table-panel">
      <div className="panel-header">
        <div>
          <h2>Candidate Grid</h2>
          <p>Shortlist, OOS, robustness, cluster pressure, manual overrides.</p>
        </div>
        <div className="panel-caption">{props.candidates.length} visible</div>
      </div>
      <div className="table-wrap">
        <table className="candidate-table">
          <thead>
            <tr>
              <th>Name</th>
              <th>Rank</th>
              <th>OOS</th>
              <th>Train p05</th>
              <th>DD</th>
              <th>Cluster</th>
              <th>Flags</th>
            </tr>
          </thead>
          <tbody>
            {props.candidates.map((candidate) => {
              const isSelected = props.selectedId === candidate.candidate_id;
              const flagParts = [
                candidate.shortlist.selected ? "selected" : null,
                candidate.overrides.pin ? "pin" : null,
                candidate.shortlist.exception_flags.length > 0 ? "exception" : null,
              ].filter(Boolean);
              return (
                <tr
                  key={candidate.candidate_id}
                  className={isSelected ? "selected" : ""}
                  onClick={() => props.onSelect(candidate.candidate_id)}
                >
                  <td>
                    <div className="table-name">{candidate.display_name}</div>
                    <div className="table-subtitle">{candidate.representation_name}</div>
                  </td>
                  <td>{candidate.shortlist.selected_rank ?? "—"}</td>
                  <td className={candidate.periods.oos?.pnl && candidate.periods.oos.pnl > 0 ? "positive" : "negative"}>
                    {formatMoney(candidate.periods.oos?.pnl)}
                  </td>
                  <td>{formatMoney(candidate.resampling?.p05_net_profit)}</td>
                  <td>{formatPct(candidate.periods.oos?.max_drawdown_pct)}</td>
                  <td>
                    <span
                      className="cluster-dot"
                      style={{ backgroundColor: colorForCluster(candidate.cluster_id) }}
                    />
                    {candidate.cluster_id ?? "none"}
                  </td>
                  <td>{flagParts.length > 0 ? flagParts.join(" · ") : "—"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function BroomView(props: {
  lines: BroomLine[];
  selectedId: string | null;
  onSelect: (candidateId: string) => void;
}) {
  const width = 1180;
  const height = 360;

  const extents = useMemo(() => {
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    for (const line of props.lines) {
      for (const point of line.normalized_balance_history) {
        minValue = Math.min(minValue, point);
        maxValue = Math.max(maxValue, point);
      }
    }
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
      minValue = 0.95;
      maxValue = 1.05;
    }
    const padding = (maxValue - minValue || 0.05) * 0.12;
    return {
      minY: minValue - padding,
      maxY: maxValue + padding,
      maxX: Math.max(...props.lines.flatMap((line) => line.sample_indices), 1),
    };
  }, [props.lines]);

  return (
    <div className="panel broom-panel">
      <div className="panel-header">
        <div>
          <h2>Broom View</h2>
          <p>Normalized capital curves on one common window. Brightness follows shortlist pressure.</p>
        </div>
        <div className="panel-caption">{props.lines.length} curves</div>
      </div>
      <svg className="broom-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <line
          x1={0}
          x2={width}
          y1={((extents.maxY - 1) / (extents.maxY - extents.minY)) * height}
          y2={((extents.maxY - 1) / (extents.maxY - extents.minY)) * height}
          className="baseline-line"
        />
        {props.lines.map((line) => {
          const color = colorForCluster(line.cluster_id);
          const opacity = props.selectedId && props.selectedId !== line.candidate_id ? 0.1 : Math.max(0.12, line.brightness_hint ?? 0.35);
          const strokeWidth = props.selectedId === line.candidate_id ? 4.4 : line.selected ? 2.8 : 1.4;
          const points = line.sample_indices
            .map((index, pointIndex) => {
              const x = (index / extents.maxX) * width;
              const value = line.normalized_balance_history[pointIndex] ?? 1;
              const y = ((extents.maxY - value) / (extents.maxY - extents.minY)) * height;
              return `${x},${y}`;
            })
            .join(" ");
          return (
            <polyline
              key={line.candidate_id}
              points={points}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              opacity={opacity}
              onClick={() => props.onSelect(line.candidate_id)}
              className="broom-line"
            />
          );
        })}
      </svg>
      <div className="legend-strip">
        {Array.from(new Set(props.lines.map((line) => line.cluster_id ?? "none"))).map((clusterId) => (
          <div className="legend-item" key={clusterId}>
            <span className="cluster-dot" style={{ backgroundColor: colorForCluster(clusterId === "none" ? null : clusterId) }} />
            {clusterId}
          </div>
        ))}
      </div>
    </div>
  );
}

function CandidateDetail(props: { candidate: CandidateRow | null }) {
  if (!props.candidate) {
    return (
      <div className="panel detail-panel">
        <div className="panel-header">
          <div>
            <h2>Candidate Detail</h2>
            <p>Select a curve or a row to inspect full metrics.</p>
          </div>
        </div>
      </div>
    );
  }

  const candidate = props.candidate;
  return (
    <div className="panel detail-panel">
      <div className="panel-header">
        <div>
          <h2>{candidate.display_name}</h2>
          <p>{candidate.representation_name}</p>
        </div>
        <div className="detail-status-row">
          <span className={`status-pill status-${candidate.status}`}>{candidate.status}</span>
          {candidate.overrides.pin ? <span className="status-pill status-pin">pinned</span> : null}
        </div>
      </div>
      <div className="detail-grid">
        {["train", "oos"].map((periodName) => {
          const period = candidate.periods[periodName];
          if (!period) {
            return null;
          }
          return (
            <section className="detail-card" key={periodName}>
              <h3>{periodName.toUpperCase()}</h3>
              <div className="detail-stat-row">
                <span>PnL</span>
                <strong>{formatMoney(period.pnl)}</strong>
              </div>
              <div className="detail-stat-row">
                <span>Drawdown</span>
                <strong>{formatPct(period.max_drawdown_pct)}</strong>
              </div>
              <div className="detail-stat-row">
                <span>Trades</span>
                <strong>{period.trades ?? "n/a"}</strong>
              </div>
              <div className="detail-stat-row">
                <span>Win Rate</span>
                <strong>{formatPct(period.win_rate_pct)}</strong>
              </div>
            </section>
          );
        })}
        <section className="detail-card">
          <h3>Robustness</h3>
          <div className="detail-stat-row">
            <span>Train p05</span>
            <strong>{formatMoney(candidate.resampling?.p05_net_profit)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>p95 DD</span>
            <strong>{formatPct(candidate.resampling?.p95_max_drawdown_pct)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Loss Rate</span>
            <strong>{formatPct((candidate.resampling?.loss_rate ?? 0) * 100)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Profitable Rate</span>
            <strong>{formatPct((candidate.resampling?.profitable_rate ?? 0) * 100)}</strong>
          </div>
        </section>
        <section className="detail-card">
          <h3>Operator Context</h3>
          <div className="detail-stat-row">
            <span>Cluster</span>
            <strong>{candidate.cluster_id ?? "none"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Base Score</span>
            <strong>{compact(candidate.shortlist.base_score)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Exception Flags</span>
            <strong>{candidate.shortlist.exception_flags.join(", ") || "none"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Cluster Cap</span>
            <strong>{candidate.overrides.cluster_max_cap_fraction == null ? "none" : formatPct(candidate.overrides.cluster_max_cap_fraction * 100)}</strong>
          </div>
        </section>
      </div>
    </div>
  );
}

function ScenarioPanel(props: {
  title: string;
  subtitle: string;
  scenarios: ScenarioPayload[] | CombinationScenarioPayload[];
  chosenName: string | null;
  mode: "allocator" | "combination";
}) {
  return (
    <div className="panel scenario-panel">
      <div className="panel-header">
        <div>
          <h2>{props.title}</h2>
          <p>{props.subtitle}</p>
        </div>
      </div>
      <div className="scenario-stack">
        {props.scenarios.map((scenario) => {
          const active = scenario.name === props.chosenName;
          return (
            <article className={`scenario-card ${active ? "active" : ""}`} key={scenario.name}>
              <div className="scenario-header">
                <strong>{scenario.name}</strong>
                <span>{compact(scenario.objective_score, 3)}</span>
              </div>
              <div className="scenario-meta">
                <span>risk {compact(scenario.allocated_risk_fraction, 2)}</span>
                <span>p05 {formatMoney(scenario.resampling.p05_net_profit)}</span>
                <span>dd {formatPct(scenario.resampling.p95_max_drawdown_pct)}</span>
              </div>
              {props.mode === "combination" && "subset_display_names" in scenario ? (
                <div className="scenario-subset">{scenario.subset_display_names.join(" + ")}</div>
              ) : null}
              <div className="weight-stack">
                {scenario.weights.map((weight) => (
                  <div className="weight-row" key={`${scenario.name}-${weight.candidate_id}`}>
                    <div className="weight-label">
                      <span
                        className="cluster-dot"
                        style={{ backgroundColor: colorForCluster(weight.cluster_id) }}
                      />
                      {weight.display_name}
                    </div>
                    <div className="weight-bar">
                      <div
                        className={`weight-fill ${weight.capped ? "capped" : ""}`}
                        style={{ width: `${Math.max(2, weight.capital_fraction * 100)}%` }}
                      />
                    </div>
                    <div className="weight-value">{compact(weight.capital_fraction, 3)}</div>
                  </div>
                ))}
              </div>
            </article>
          );
        })}
      </div>
    </div>
  );
}

function ClusterPanel(props: { clusters: DashboardFeed["clusters"] }) {
  return (
    <div className="panel cluster-panel">
      <div className="panel-header">
        <div>
          <h2>Cluster Pressure</h2>
          <p>Similarity buckets for concentration review.</p>
        </div>
      </div>
      <div className="cluster-list">
        {props.clusters.map((cluster) => (
          <article className="cluster-card" key={cluster.cluster_id}>
            <div className="cluster-card-head">
              <div className="cluster-title">
                <span
                  className="cluster-dot large"
                  style={{ backgroundColor: colorForCluster(cluster.cluster_id) }}
                />
                {cluster.cluster_id}
              </div>
              <div className="cluster-size">{cluster.cluster_size} systems</div>
            </div>
            <div className="cluster-metrics">
              <span>selected {cluster.selected_count}</span>
              <span>pinned {cluster.pinned_count}</span>
              <span>sim {compact(cluster.mean_similarity_score, 2)}</span>
              <span>cap {cluster.max_cap_fraction == null ? "none" : compact(cluster.max_cap_fraction, 2)}</span>
            </div>
            <div className="cluster-members">{cluster.display_names.join(", ")}</div>
          </article>
        ))}
      </div>
    </div>
  );
}

function OverridesPanel(props: { overrides: DashboardFeed["overrides"] }) {
  if (!props.overrides) {
    return null;
  }
  return (
    <div className="panel override-panel">
      <div className="panel-header">
        <div>
          <h2>Operator Overrides</h2>
          <p>Manual intent stays on the same reproducible path as search and allocation.</p>
        </div>
      </div>
      <div className="override-summary">
        <span>{props.overrides.candidate_override_count} candidate overrides</span>
        <span>{props.overrides.cluster_override_count} cluster overrides</span>
        <span>updated {formatDateTime(props.overrides.updated_at_utc)}</span>
      </div>
      <div className="audit-stack">
        {props.overrides.recent_audit_entries.map((entry) => (
          <article className="audit-card" key={`${entry.created_at_utc}-${entry.target_id}`}>
            <div className="audit-title">
              <strong>{entry.action}</strong>
              <span>{entry.actor}</span>
            </div>
            <div className="audit-target">
              {entry.target_type}:{entry.target_id}
            </div>
            <div className="audit-note">{entry.note ?? "no note"}</div>
          </article>
        ))}
      </div>
    </div>
  );
}

function CandidateDashboardApp(props: { feed: DashboardFeed }) {
  const feed = props.feed;
  const [query, setQuery] = useState("");
  const [selectedOnly, setSelectedOnly] = useState(false);
  const [exceptionsOnly, setExceptionsOnly] = useState(false);
  const [pinnedOnly, setPinnedOnly] = useState(false);
  const [sortKey, setSortKey] = useState<"score" | "oos" | "robustness" | "drawdown">("score");
  const [selectedCandidateId, setSelectedCandidateId] = useState<string | null>(null);

  useEffect(() => {
    if (selectedCandidateId) {
      return;
    }
    setSelectedCandidateId(feed.candidates[0]?.candidate_id ?? null);
  }, [feed, selectedCandidateId]);

  const filteredCandidates = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const result = feed.candidates.filter((candidate) => {
      if (selectedOnly && !candidate.shortlist.selected) {
        return false;
      }
      if (exceptionsOnly && candidate.shortlist.exception_flags.length === 0) {
        return false;
      }
      if (pinnedOnly && !candidate.overrides.pin) {
        return false;
      }
      if (normalizedQuery) {
        const haystack = [
          candidate.display_name,
          candidate.candidate_id,
          candidate.cluster_id ?? "",
          candidate.representation_name ?? "",
        ]
          .join(" ")
          .toLowerCase();
        if (!haystack.includes(normalizedQuery)) {
          return false;
        }
      }
      return true;
    });
    result.sort((left, right) => {
      const delta = candidateSortValue(right, sortKey) - candidateSortValue(left, sortKey);
      if (delta !== 0) {
        return delta;
      }
      return left.display_name.localeCompare(right.display_name);
    });
    return result;
  }, [exceptionsOnly, feed, pinnedOnly, query, selectedOnly, sortKey]);

  const visibleBroomLines = useMemo(() => {
    if (!feed.broom) {
      return [];
    }
    const allowedIds = new Set(filteredCandidates.map((candidate) => candidate.candidate_id));
    return feed.broom.lines.filter((line) => allowedIds.has(line.candidate_id));
  }, [feed, filteredCandidates]);

  const selectedCandidate = useMemo(
    () =>
      filteredCandidates.find((candidate) => candidate.candidate_id === selectedCandidateId) ??
      feed.candidates.find((candidate) => candidate.candidate_id === selectedCandidateId) ??
      null,
    [feed, filteredCandidates, selectedCandidateId],
  );

  return (
    <main className="app-shell">
      <div className="app-backdrop" />
      <header className="hero">
        <div className="hero-copy">
          <div className="eyebrow">Operator Console</div>
          <h1>Search Monitoring And Portfolio Review</h1>
          <p>
            A thin client over the snapshot feed. The operator sees shortlist pressure,
            cluster concentration, overrides, and portfolio scenarios on one surface.
          </p>
        </div>
        <div className="hero-meta">
          <div className="hero-meta-row">
            <span>feed</span>
            <strong>{feed.name}</strong>
          </div>
          <div className="hero-meta-row">
            <span>updated</span>
            <strong>{formatDateTime(feed.created_at_utc)}</strong>
          </div>
          <div className="hero-meta-row">
            <span>allocator</span>
            <strong>{feed.allocator?.chosen_scenario_name ?? "n/a"}</strong>
          </div>
          <div className="hero-meta-row">
            <span>best subset</span>
            <strong>{feed.combinations?.best_scenario_name ?? "n/a"}</strong>
          </div>
        </div>
      </header>

      <section className="summary-strip">
        <SummaryCard label="Candidate Pool" value={feed.summary.total_candidates} accent="ink" hint="registry scope" />
        <SummaryCard label="Selected" value={feed.summary.selected_candidate_count} accent="teal" hint="current shortlist" />
        <SummaryCard label="Exceptions" value={feed.summary.exception_candidate_count} accent="ember" hint="near misses and outliers" />
        <SummaryCard label="Pinned" value={feed.summary.pinned_candidate_count} accent="ink" hint="manual operator hold" />
        <SummaryCard label="Clusters" value={feed.summary.cluster_count} accent="teal" hint="concentration buckets" />
        <SummaryCard label="OOS Positive" value={feed.summary.oos_positive_count} accent="ember" hint="adjacent holdout winners" />
      </section>

      <section className="toolbar">
        <div className="toolbar-left">
          <input
            className="search-input"
            placeholder="Search candidate, cluster, representation"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <FilterChip active={selectedOnly} label="Selected Only" onClick={() => setSelectedOnly((value) => !value)} />
          <FilterChip active={exceptionsOnly} label="Exceptions" onClick={() => setExceptionsOnly((value) => !value)} />
          <FilterChip active={pinnedOnly} label="Pinned" onClick={() => setPinnedOnly((value) => !value)} />
        </div>
        <div className="toolbar-right">
          <span>Sort</span>
          <select className="sort-select" value={sortKey} onChange={(event) => setSortKey(event.target.value as typeof sortKey)}>
            <option value="score">Shortlist Score</option>
            <option value="oos">OOS PnL</option>
            <option value="robustness">Train p05</option>
            <option value="drawdown">OOS Drawdown</option>
          </select>
        </div>
      </section>

      <section className="dashboard-grid">
        <BroomView lines={visibleBroomLines} selectedId={selectedCandidateId} onSelect={setSelectedCandidateId} />
        <CandidateDetail candidate={selectedCandidate} />
        <CandidateTable candidates={filteredCandidates} selectedId={selectedCandidateId} onSelect={setSelectedCandidateId} />
        <div className="right-column">
          {feed.allocator ? (
            <ScenarioPanel
              title="Allocator Workbench"
              subtitle="Global risk dial and current sizing proposals."
              scenarios={feed.allocator.scenarios}
              chosenName={feed.allocator.chosen_scenario_name}
              mode="allocator"
            />
          ) : null}
          {feed.combinations ? (
            <ScenarioPanel
              title="Combination Search"
              subtitle="Best shortlist subsets under the same portfolio objective."
              scenarios={feed.combinations.scenarios}
              chosenName={feed.combinations.best_scenario_name}
              mode="combination"
            />
          ) : null}
          <ClusterPanel clusters={feed.clusters} />
          <OverridesPanel overrides={feed.overrides} />
        </div>
      </section>
    </main>
  );
}

function FarmBroomView(props: {
  lines: FarmBroomLine[];
  selectedScenarioName: string | null;
  onSelect: (scenarioName: string) => void;
}) {
  const width = 1180;
  const height = 360;
  const extents = useMemo(() => {
    let minValue = Number.POSITIVE_INFINITY;
    let maxValue = Number.NEGATIVE_INFINITY;
    for (const line of props.lines) {
      for (const point of line.normalized_balance_history) {
        minValue = Math.min(minValue, point);
        maxValue = Math.max(maxValue, point);
      }
    }
    if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
      minValue = 0.95;
      maxValue = 1.05;
    }
    const padding = (maxValue - minValue || 0.05) * 0.12;
    return {
      minY: minValue - padding,
      maxY: maxValue + padding,
      maxX: Math.max(...props.lines.flatMap((line) => line.sample_indices), 1),
    };
  }, [props.lines]);

  return (
    <div className="panel broom-panel">
      <div className="panel-header">
        <div>
          <h2>Farm Broom</h2>
          <p>Scenario-level normalized conveyor curves from a shared origin. Brightness follows current rank.</p>
        </div>
        <div className="panel-caption">{props.lines.length} scenarios</div>
      </div>
      <svg className="broom-svg" viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <line
          x1={0}
          x2={width}
          y1={((extents.maxY - 1) / (extents.maxY - extents.minY)) * height}
          y2={((extents.maxY - 1) / (extents.maxY - extents.minY)) * height}
          className="baseline-line"
        />
        {props.lines.map((line) => {
          const color = colorForFarmLine(line);
          const opacity =
            props.selectedScenarioName && props.selectedScenarioName !== line.scenario_name
              ? 0.1
              : Math.max(0.14, line.brightness_hint ?? 0.4);
          const strokeWidth = props.selectedScenarioName === line.scenario_name ? 4.2 : line.gate_pass ? 2.8 : 1.6;
          const points = line.sample_indices
            .map((index, pointIndex) => {
              const x = (index / extents.maxX) * width;
              const value = line.normalized_balance_history[pointIndex] ?? 1;
              const y = ((extents.maxY - value) / (extents.maxY - extents.minY)) * height;
              return `${x},${y}`;
            })
            .join(" ");
          return (
            <polyline
              key={line.scenario_name}
              points={points}
              fill="none"
              stroke={color}
              strokeWidth={strokeWidth}
              opacity={opacity}
              onClick={() => props.onSelect(line.scenario_name)}
              className="broom-line"
            />
          );
        })}
      </svg>
      <div className="legend-strip">
        <div className="legend-item">
          <span className="cluster-dot" style={{ backgroundColor: "#137f48" }} />
          gate pass
        </div>
        <div className="legend-item">
          <span className="cluster-dot" style={{ backgroundColor: "#de6b48" }} />
          completed laggard
        </div>
        <div className="legend-item">
          <span className="cluster-dot" style={{ backgroundColor: "#0f8a8d" }} />
          running
        </div>
        <div className="legend-item">
          <span className="cluster-dot" style={{ backgroundColor: "#bc4749" }} />
          failed
        </div>
      </div>
    </div>
  );
}

function FarmScenarioTable(props: {
  scenarios: FarmScenarioRow[];
  selectedScenarioName: string | null;
  onSelect: (scenarioName: string) => void;
}) {
  return (
    <div className="panel table-panel">
      <div className="panel-header">
        <div>
          <h2>Farm Scenario Grid</h2>
          <p>Gate verdicts, conveyor returns, cycle counts, progress stages.</p>
        </div>
        <div className="panel-caption">{props.scenarios.length} visible</div>
      </div>
      <div className="table-wrap">
        <table className="candidate-table">
          <thead>
            <tr>
              <th>Scenario</th>
              <th>Status</th>
              <th>PnL</th>
              <th>DD</th>
              <th>Cycles</th>
              <th>Pool</th>
              <th>Gate</th>
            </tr>
          </thead>
          <tbody>
            {props.scenarios.map((scenario) => (
              <tr
                key={scenario.scenario_name}
                className={props.selectedScenarioName === scenario.scenario_name ? "selected" : ""}
                onClick={() => props.onSelect(scenario.scenario_name)}
              >
                <td>
                  <div className="table-name">{scenario.scenario_name}</div>
                  <div className="table-subtitle">{scenario.progress_stage ?? "no-stage"}</div>
                </td>
                <td>
                  <span className={`status-pill status-${scenario.status}`}>{scenario.status}</span>
                </td>
                <td className={scenario.total_pnl != null && scenario.total_pnl > 0 ? "positive" : "negative"}>
                  {formatMoney(scenario.total_pnl)}
                </td>
                <td>{formatPct(scenario.max_drawdown_pct)}</td>
                <td>{scenario.evaluated_cycle_count ?? "—"}</td>
                <td>{scenario.selected_candidate_count}/{scenario.candidate_pool_count}</td>
                <td>{scenario.gate_pass == null ? "—" : scenario.gate_pass ? "pass" : "fail"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function FarmScenarioDetail(props: { scenario: FarmScenarioRow | null }) {
  if (!props.scenario) {
    return (
      <div className="panel detail-panel">
        <div className="panel-header">
          <div>
            <h2>Scenario Detail</h2>
            <p>Select a farm curve or row to inspect one scenario.</p>
          </div>
        </div>
      </div>
    );
  }
  const scenario = props.scenario;
  return (
    <div className="panel detail-panel">
      <div className="panel-header">
        <div>
          <h2>{scenario.scenario_name}</h2>
          <p>{scenario.mode} · {scenario.progress_stage ?? "no-stage"}</p>
        </div>
        <div className="detail-status-row">
          <span className={`status-pill status-${scenario.status}`}>{scenario.status}</span>
          {scenario.gate_pass != null ? (
            <span className={`status-pill ${scenario.gate_pass ? "status-gate-pass" : "status-gate-fail"}`}>
              {scenario.gate_pass ? "gate pass" : "gate fail"}
            </span>
          ) : null}
        </div>
      </div>
      <div className="detail-grid">
        <section className="detail-card">
          <h3>Conveyor Result</h3>
          <div className="detail-stat-row">
            <span>PnL</span>
            <strong>{formatMoney(scenario.total_pnl)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Return</span>
            <strong>{formatPct(scenario.total_return_pct)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Drawdown</span>
            <strong>{formatPct(scenario.max_drawdown_pct)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Positive Cycles</span>
            <strong>{scenario.positive_cycle_count ?? "n/a"}</strong>
          </div>
        </section>
        <section className="detail-card">
          <h3>Coverage</h3>
          <div className="detail-stat-row">
            <span>Candidate Pool</span>
            <strong>{scenario.candidate_pool_count}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Selected</span>
            <strong>{scenario.selected_candidate_count}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Cycles</span>
            <strong>{scenario.evaluated_cycle_count ?? "n/a"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Selection Start</span>
            <strong>{formatDateTime(scenario.selection_start_utc)}</strong>
          </div>
        </section>
        <section className="detail-card">
          <h3>Gate Context</h3>
          <div className="detail-stat-row">
            <span>Beaten</span>
            <strong>{scenario.beaten_baselines.join(", ") || "none"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Failed Required</span>
            <strong>{scenario.failed_required_baselines.join(", ") || "none"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Status Counts</span>
            <strong>{Object.entries(scenario.final_status_counts).map(([key, value]) => `${key}:${value}`).join(" ") || "n/a"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Error</span>
            <strong>{scenario.error_message ?? "none"}</strong>
          </div>
        </section>
        <section className="detail-card">
          <h3>Artifacts</h3>
          <div className="detail-stat-row">
            <span>Rolling</span>
            <strong>{scenario.rolling_report_name ?? "n/a"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Portfolio</span>
            <strong>{scenario.portfolio_ledger_report_name ?? "n/a"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Baselines</span>
            <strong>{scenario.portfolio_baselines_report_name ?? "n/a"}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Output</span>
            <strong className="mono-path">{scenario.output_dir}</strong>
          </div>
        </section>
      </div>
    </div>
  );
}

function FarmMonitoringPanel(props: { feed: FarmDashboardFeed }) {
  return (
    <div className="panel scenario-panel">
      <div className="panel-header">
        <div>
          <h2>Farm Monitoring</h2>
          <p>Heartbeat, throughput, stagnation, recent events, and stage distribution.</p>
        </div>
      </div>
      <div className="monitor-grid">
        <div className="detail-card">
          <h3>Heartbeat</h3>
          <div className="detail-stat-row">
            <span>State</span>
            <strong>{props.feed.monitoring.heartbeat_state}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Last Event</span>
            <strong>{formatDurationSeconds(props.feed.monitoring.seconds_since_last_event)} ago</strong>
          </div>
          <div className="detail-stat-row">
            <span>Median Gap</span>
            <strong>{formatDurationSeconds(props.feed.monitoring.median_event_gap_seconds)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Events / hour</span>
            <strong>{compact(props.feed.monitoring.events_per_hour, 2)}</strong>
          </div>
        </div>
        <div className="detail-card">
          <h3>Stagnation</h3>
          <div className="detail-stat-row">
            <span>State</span>
            <strong>{props.feed.monitoring.stagnation_state}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Since Completion</span>
            <strong>{formatDurationSeconds(props.feed.monitoring.seconds_since_last_completion)} ago</strong>
          </div>
          <div className="detail-stat-row">
            <span>Since Gate Pass</span>
            <strong>{formatDurationSeconds(props.feed.monitoring.seconds_since_last_gate_pass)} ago</strong>
          </div>
          <div className="detail-stat-row">
            <span>Gate Pass / hour</span>
            <strong>{compact(props.feed.monitoring.gate_pass_events_per_hour, 2)}</strong>
          </div>
        </div>
        <div className="detail-card">
          <h3>Throughput</h3>
          <div className="detail-stat-row">
            <span>Recent 15m</span>
            <strong>{props.feed.monitoring.events_last_15m} events</strong>
          </div>
          <div className="detail-stat-row">
            <span>Completions 15m</span>
            <strong>{props.feed.monitoring.completion_events_last_15m}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Completed / hour</span>
            <strong>{compact(props.feed.monitoring.completion_events_per_hour, 2)}</strong>
          </div>
          <div className="detail-stat-row">
            <span>Gate Pass 15m</span>
            <strong>{props.feed.monitoring.gate_pass_events_last_15m}</strong>
          </div>
        </div>
        <div className="detail-card">
          <h3>Running</h3>
          <div className="monitor-list">
            {props.feed.monitoring.running_scenarios.length > 0 ? props.feed.monitoring.running_scenarios.join(", ") : "none"}
          </div>
        </div>
        <div className="detail-card">
          <h3>Recent Gate Pass</h3>
          <div className="monitor-list">
            {props.feed.monitoring.recent_gate_pass_scenarios.length > 0 ? props.feed.monitoring.recent_gate_pass_scenarios.join(", ") : "none"}
          </div>
        </div>
        <div className="detail-card">
          <h3>Recent Failures</h3>
          <div className="monitor-list">
            {props.feed.monitoring.recent_failed_scenarios.length > 0 ? props.feed.monitoring.recent_failed_scenarios.join(", ") : "none"}
          </div>
        </div>
        <div className="detail-card">
          <h3>Stages</h3>
          <div className="monitor-list">
            {Object.entries(props.feed.monitoring.progress_stage_counts)
              .map(([key, value]) => `${key}:${value}`)
              .join(" · ") || "none"}
          </div>
        </div>
        <div className="detail-card">
          <h3>Recent Events</h3>
          <div className="monitor-list">
            {props.feed.monitoring.recent_events.length > 0
              ? props.feed.monitoring.recent_events
                  .slice()
                  .reverse()
                  .map((event) => `${event.scenario_name} ${event.event_kind} ${event.progress_stage ?? event.status}`)
                  .join(" · ")
              : "none"}
          </div>
        </div>
      </div>
    </div>
  );
}

function FarmDashboardApp(props: {
  feed: FarmDashboardFeed;
  refreshing: boolean;
  lastLoadedAt: string | null;
  onReload: () => void;
}) {
  const feed = props.feed;
  const nowMs = useNowTick(1000);
  const [query, setQuery] = useState("");
  const [gateOnly, setGateOnly] = useState(false);
  const [runningOnly, setRunningOnly] = useState(false);
  const [completedOnly, setCompletedOnly] = useState(false);
  const [sortKey, setSortKey] = useState<"pnl" | "gate" | "drawdown" | "cycles" | "updated">("pnl");
  const [selectedScenarioName, setSelectedScenarioName] = useState<string | null>(null);

  const liveMonitoring = useMemo(() => {
    const base = feed.monitoring;
    const secondsSinceLastEvent =
      secondsSinceTimestamp(base.last_event_at ?? base.latest_updated_at, nowMs) ?? base.seconds_since_last_event;
    const secondsSinceLastCompletion =
      secondsSinceTimestamp(base.last_completion_at, nowMs) ?? base.seconds_since_last_completion;
    const secondsSinceLastGatePass =
      secondsSinceTimestamp(base.last_gate_pass_at, nowMs) ?? base.seconds_since_last_gate_pass;
    return {
      ...base,
      seconds_since_last_event: secondsSinceLastEvent,
      seconds_since_last_completion: secondsSinceLastCompletion,
      seconds_since_last_gate_pass: secondsSinceLastGatePass,
      heartbeat_state: deriveHeartbeatState(secondsSinceLastEvent, base.median_event_gap_seconds),
      stagnation_state: deriveStagnationState({
        secondsSinceLastCompletion,
        secondsSinceLastGatePass,
        medianCompletionGapSeconds: base.median_completion_gap_seconds,
        completionCount: base.completion_event_count,
        gatePassCompletionCount: base.gate_pass_completion_count,
      }),
    };
  }, [feed.monitoring, nowMs]);

  useEffect(() => {
    if (selectedScenarioName) {
      return;
    }
    setSelectedScenarioName(feed.scenarios[0]?.scenario_name ?? null);
  }, [feed, selectedScenarioName]);

  const filteredScenarios = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const result = feed.scenarios.filter((scenario) => {
      if (gateOnly && scenario.gate_pass !== true) {
        return false;
      }
      if (runningOnly && scenario.status !== "running") {
        return false;
      }
      if (completedOnly && !["completed", "reused"].includes(scenario.status)) {
        return false;
      }
      if (normalizedQuery) {
        const haystack = [
          scenario.scenario_name,
          scenario.status,
          scenario.progress_stage ?? "",
          scenario.mode,
        ]
          .join(" ")
          .toLowerCase();
        if (!haystack.includes(normalizedQuery)) {
          return false;
        }
      }
      return true;
    });
    result.sort((left, right) => {
      const delta = farmSortValue(right, sortKey) - farmSortValue(left, sortKey);
      if (delta !== 0) {
        return delta;
      }
      return left.scenario_name.localeCompare(right.scenario_name);
    });
    return result;
  }, [completedOnly, feed, gateOnly, query, runningOnly, sortKey]);

  const visibleBroomLines = useMemo(() => {
    if (!feed.broom) {
      return [];
    }
    const allowed = new Set(filteredScenarios.map((scenario) => scenario.scenario_name));
    return feed.broom.lines.filter((line) => allowed.has(line.scenario_name));
  }, [feed, filteredScenarios]);

  const selectedScenario = useMemo(
    () =>
      filteredScenarios.find((scenario) => scenario.scenario_name === selectedScenarioName) ??
      feed.scenarios.find((scenario) => scenario.scenario_name === selectedScenarioName) ??
      null,
    [feed, filteredScenarios, selectedScenarioName],
  );

  return (
    <main className="app-shell">
      <div className="app-backdrop" />
      <header className="hero">
        <div className="hero-copy">
          <div className="eyebrow">Farm Console</div>
          <h1>Long-Run Candidate Farm Overview</h1>
          <p>
            Scenario-level monitoring over the same gate contract. This view is built to
            surface throughput, gate passes, conveyor curves, and search stagnation without
            inventing a second state model.
          </p>
        </div>
        <div className="hero-meta">
          <div className="hero-meta-row">
            <span>feed</span>
            <strong>{feed.name}</strong>
          </div>
          <div className="hero-meta-row">
            <span>updated</span>
            <strong>{formatDateTime(feed.created_at_utc)}</strong>
          </div>
          <div className="hero-meta-row">
            <span>best gate pass</span>
            <strong>{feed.summary.best_gate_scenario_by_pnl ?? "n/a"}</strong>
          </div>
          <div className="hero-meta-row">
            <span>latest activity</span>
            <strong>{formatDateTime(liveMonitoring.latest_updated_at)}</strong>
          </div>
          <div className="hero-meta-row">
            <span>heartbeat</span>
            <strong>{liveMonitoring.heartbeat_state}</strong>
          </div>
          <div className="hero-meta-row">
            <span>last refresh</span>
            <strong>{formatDateTime(props.lastLoadedAt)}</strong>
          </div>
        </div>
      </header>

      <section className="summary-strip">
        <SummaryCard label="Scenarios" value={feed.summary.scenario_count} accent="ink" hint="farm manifest scope" />
        <SummaryCard label="Completed" value={feed.summary.completed_or_reused_scenarios} accent="teal" hint="completed or reused" />
        <SummaryCard label="Gate Pass" value={feed.summary.gate_pass_count} accent="ember" hint={`${compact(feed.summary.gate_pass_rate * 100, 1)}% pass rate`} />
        <SummaryCard label="Running" value={feed.summary.running_scenarios} accent="teal" hint="currently active" />
        <SummaryCard label="Pool Coverage" value={feed.summary.total_unique_candidate_pool_ids} accent="ink" hint="unique candidate pool ids" />
        <SummaryCard label="Selected Coverage" value={feed.summary.total_unique_selected_candidate_ids} accent="ember" hint="unique selected candidate ids" />
      </section>

      <section className="toolbar">
        <div className="toolbar-left">
          <input
            className="search-input"
            placeholder="Search scenario, status, progress stage"
            value={query}
            onChange={(event) => setQuery(event.target.value)}
          />
          <FilterChip active={gateOnly} label="Gate Pass" onClick={() => setGateOnly((value) => !value)} />
          <FilterChip active={runningOnly} label="Running" onClick={() => setRunningOnly((value) => !value)} />
          <FilterChip active={completedOnly} label="Completed" onClick={() => setCompletedOnly((value) => !value)} />
        </div>
        <div className="toolbar-right">
          <div className="refresh-meta" aria-live="polite">
            <span className="refresh-interval">Auto 5s</span>
            <span className={`refresh-badge ${props.refreshing ? "is-refreshing" : "is-live"}`}>
              {props.refreshing ? "sync" : "live"}
            </span>
          </div>
          <button className="filter-chip" onClick={props.onReload}>Refresh now</button>
          <span>Sort</span>
          <select className="sort-select" value={sortKey} onChange={(event) => setSortKey(event.target.value as typeof sortKey)}>
            <option value="pnl">PnL</option>
            <option value="gate">Gate</option>
            <option value="drawdown">Drawdown</option>
            <option value="cycles">Cycles</option>
            <option value="updated">Updated</option>
          </select>
        </div>
      </section>

      <section className="dashboard-grid">
        <FarmBroomView lines={visibleBroomLines} selectedScenarioName={selectedScenarioName} onSelect={setSelectedScenarioName} />
        <FarmScenarioDetail scenario={selectedScenario} />
        <FarmScenarioTable scenarios={filteredScenarios} selectedScenarioName={selectedScenarioName} onSelect={setSelectedScenarioName} />
        <div className="right-column">
          <FarmMonitoringPanel feed={{ ...feed, monitoring: liveMonitoring }} />
        </div>
      </section>
    </main>
  );
}

function isFarmDashboardFeed(feed: unknown): feed is FarmDashboardFeed {
  return Boolean(feed) && typeof feed === "object" && "source_farm_report" in (feed as Record<string, unknown>);
}

function isCandidateDashboardFeed(feed: unknown): feed is DashboardFeed {
  return Boolean(feed) && typeof feed === "object" && "source_shortlist_report" in (feed as Record<string, unknown>);
}

export default function App() {
  const view = useMemo(
    () => (new URLSearchParams(window.location.search).get("view") === "farm" ? "farm" : "candidate"),
    [],
  );
  const feedUrl = view === "farm" ? FARM_FEED_URL : CANDIDATE_FEED_URL;
  const { feed, loading, refreshing, error, lastLoadedAt, reload } = useFeed(feedUrl);

  if (loading) {
    return <main className="app-shell loading">Loading {view === "farm" ? "farm" : "operator"} dashboard...</main>;
  }

  if (error || !feed) {
    return (
      <main className="app-shell error-shell">
        <div className="error-card">
          <h1>Dashboard feed unavailable</h1>
          <p>{error ?? "Missing dashboard feed."}</p>
          <code>{view === "farm" ? "npm run sync-feed -- --farm" : "npm run sync-feed"}</code>
        </div>
      </main>
    );
  }

  if (view === "farm") {
    if (!isFarmDashboardFeed(feed)) {
      return (
        <main className="app-shell error-shell">
          <div className="error-card">
            <h1>Wrong feed type</h1>
            <p>Requested farm view, but the loaded feed is not a farm dashboard feed.</p>
          </div>
        </main>
      );
    }
    return <FarmDashboardApp feed={feed} refreshing={refreshing} lastLoadedAt={lastLoadedAt} onReload={reload} />;
  }

  if (!isCandidateDashboardFeed(feed)) {
    return (
      <main className="app-shell error-shell">
        <div className="error-card">
          <h1>Wrong feed type</h1>
          <p>Requested candidate view, but the loaded feed is not an operator dashboard feed.</p>
        </div>
      </main>
    );
  }

  return <CandidateDashboardApp feed={feed} />;
}
