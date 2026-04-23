export type CandidatePeriod = {
  pnl: number | null;
  final_balance: number | null;
  max_drawdown_pct: number | null;
  trades: number | null;
  win_rate_pct: number | null;
  full_window: boolean | null;
  beats_flat: boolean | null;
  beats_best_baseline: boolean | null;
  baseline_winner: string | null;
  start_utc: string | null;
  end_utc: string | null;
};

export type CandidateRow = {
  candidate_id: string;
  display_name: string;
  created_at_utc: string;
  status: string;
  tags: string[];
  engine_name: string;
  engine_role: string;
  representation_name: string | null;
  cluster_id: string | null;
  cluster_size: number | null;
  selection_flags: Record<string, boolean>;
  periods: Record<string, CandidatePeriod>;
  resampling: {
    name: string;
    fraction: number;
    pessimistic_net_profit: number;
    pessimistic_max_drawdown_pct: number;
    p05_net_profit: number;
    p95_max_drawdown_pct: number;
    loss_rate: number;
    profitable_rate: number;
    ruin_rate: number;
    original_net_profit: number;
    original_max_drawdown_pct: number;
    iterations: number;
  } | null;
  shortlist: {
    selected: boolean;
    selected_rank: number | null;
    base_score: number | null;
    marginal_score: number | null;
    brightness_hint: number | null;
    exception_flags: string[];
    score_components: Record<string, number>;
  };
  overrides: {
    pin: boolean;
    force_include: boolean;
    exclude: boolean;
    max_cap_fraction: number | null;
    cluster_max_cap_fraction: number | null;
  };
  notes: string | null;
};

export type BroomLine = {
  candidate_id: string;
  display_name: string;
  status: string;
  cluster_id: string | null;
  selected: boolean;
  brightness_hint: number | null;
  exception_flags: string[];
  normalized_balance_history: number[];
  sample_indices: number[];
  final_balance: number;
  max_drawdown_pct: number;
};

export type WeightPayload = {
  candidate_id: string;
  display_name: string;
  cluster_id: string | null;
  normalized_share: number;
  capital_fraction: number;
  capped: boolean;
  cap_reason: string | null;
};

export type ScenarioPayload = {
  name: string;
  objective_score: number;
  requested_risk_fraction: number;
  allocated_risk_fraction: number;
  reserve_fraction: number;
  score_mode: string;
  weights: WeightPayload[];
  curve_sample_indices: number[];
  normalized_balance_history: number[];
  resampling: {
    p05_net_profit: number;
    p25_net_profit: number;
    median_net_profit: number;
    p95_max_drawdown_pct: number;
    profitable_rate: number;
    loss_rate: number;
    ruin_rate: number;
    original_net_profit: number;
    original_max_drawdown_pct: number;
  };
};

export type CombinationScenarioPayload = ScenarioPayload & {
  subset_candidate_ids: string[];
  subset_display_names: string[];
  subset_size: number;
};

export type DashboardFeed = {
  schema_version: string;
  name: string;
  created_at_utc: string;
  source_shortlist_report: string | null;
  source_diversification_report: string | null;
  source_cluster_report: string | null;
  source_override_set: string | null;
  source_allocator_report: string | null;
  source_combination_report: string | null;
  summary: {
    total_candidates: number;
    visible_candidate_rows: number;
    selected_candidate_count: number;
    exception_candidate_count: number;
    oos_positive_count: number;
    pinned_candidate_count: number;
    excluded_candidate_count: number;
    forced_candidate_count: number;
    cluster_count: number;
    status_counts: Record<string, number>;
    engine_counts: Record<string, number>;
    representation_counts: Record<string, number>;
  };
  candidates: CandidateRow[];
  monitoring: {
    latest_candidate_created_at: string | null;
    latest_oos_positive_created_at: string | null;
    latest_selected_created_at: string | null;
    recent_candidates: Array<{
      candidate_id: string;
      display_name: string;
      created_at_utc: string;
      status: string;
      oos_positive: boolean;
      selected: boolean;
      pinned: boolean;
    }>;
    interesting_candidates: Array<{
      candidate_id: string;
      display_name: string;
      exception_flags: string[];
      pinned: boolean;
      selected: boolean;
      oos_positive: boolean;
    }>;
  };
  broom: {
    source_report: string;
    start_utc: string;
    end_utc: string;
    start_step: number;
    max_steps: number;
    line_count: number;
    total_line_count: number;
    lines: BroomLine[];
  } | null;
  clusters: Array<{
    cluster_id: string;
    candidate_ids: string[];
    display_names: string[];
    cluster_size: number;
    mean_return_corr: number;
    mean_downside_corr: number;
    mean_simultaneous_loss_rate: number;
    mean_similarity_score: number;
    selected_count: number;
    pinned_count: number;
    max_cap_fraction: number | null;
  }>;
  overrides: {
    name: string;
    updated_at_utc: string;
    candidate_override_count: number;
    cluster_override_count: number;
    candidate_overrides: Array<Record<string, unknown>>;
    cluster_overrides: Array<Record<string, unknown>>;
    recent_audit_entries: Array<{
      created_at_utc: string;
      actor: string;
      target_type: string;
      target_id: string;
      action: string;
      changes: Record<string, unknown>;
      note: string | null;
    }>;
  } | null;
  allocator: {
    name: string;
    chosen_scenario_name: string | null;
    requested_risk_fractions: number[];
    selected_candidate_ids: string[];
    scenarios: ScenarioPayload[];
  } | null;
  combinations: {
    name: string;
    best_scenario_name: string | null;
    pool_candidate_ids: string[];
    searched_subset_sizes: number[];
    evaluated_combination_count: number;
    evaluated_scenario_count: number;
    scenarios: CombinationScenarioPayload[];
  } | null;
  notes: string | null;
};

export type FarmBroomLine = {
  scenario_name: string;
  status: string;
  mode: string;
  gate_pass: boolean | null;
  progress_stage: string | null;
  brightness_hint: number | null;
  normalized_balance_history: number[];
  sample_indices: number[];
  final_balance: number;
  total_pnl: number;
  max_drawdown_pct: number;
};

export type FarmScenarioRow = {
  scenario_name: string;
  status: string;
  progress_stage: string | null;
  updated_at_utc: string | null;
  mode: string;
  selection_start_utc: string | null;
  selection_days: number | null;
  num_cycles: number | null;
  rolling_report_name: string | null;
  lifecycle_report_name: string | null;
  portfolio_ledger_report_name: string | null;
  portfolio_baselines_report_name: string | null;
  output_dir: string;
  candidate_pool_count: number;
  selected_candidate_count: number;
  final_status_counts: Record<string, number>;
  gate_pass: boolean | null;
  beaten_baselines: string[];
  failed_required_baselines: string[];
  total_pnl: number | null;
  total_return_pct: number | null;
  max_drawdown_pct: number | null;
  evaluated_cycle_count: number | null;
  positive_cycle_count: number | null;
  error_message: string | null;
  log_paths: Record<string, string>;
};

export type FarmDashboardFeed = {
  schema_version: string;
  name: string;
  created_at_utc: string;
  source_farm_report: string;
  summary: {
    scenario_count: number;
    completed_scenarios: number;
    completed_or_reused_scenarios: number;
    failed_scenarios: number;
    planned_scenarios: number;
    running_scenarios: number;
    gate_pass_count: number;
    gate_pass_rate: number;
    total_unique_candidate_pool_ids: number;
    total_unique_selected_candidate_ids: number;
    best_scenario_by_pnl: string | null;
    best_gate_scenario_by_pnl: string | null;
    lowest_drawdown_scenario: string | null;
    ranked_scenarios: string[];
    status_counts: Record<string, number>;
  };
  scenarios: FarmScenarioRow[];
  monitoring: {
    latest_updated_at: string | null;
    running_scenarios: string[];
    recent_gate_pass_scenarios: string[];
    recent_failed_scenarios: string[];
    planned_scenarios: string[];
    progress_stage_counts: Record<string, number>;
    event_count: number;
    completion_event_count: number;
    gate_pass_completion_count: number;
    last_event_at: string | null;
    last_completion_at: string | null;
    last_gate_pass_at: string | null;
    events_per_hour: number | null;
    completion_events_per_hour: number | null;
    gate_pass_events_per_hour: number | null;
    median_event_gap_seconds: number | null;
    median_completion_gap_seconds: number | null;
    seconds_since_last_event: number | null;
    seconds_since_last_completion: number | null;
    seconds_since_last_gate_pass: number | null;
    events_last_15m: number;
    completion_events_last_15m: number;
    gate_pass_events_last_15m: number;
    heartbeat_state: string;
    stagnation_state: string;
    recent_events: Array<{
      sequence: number;
      created_at_utc: string;
      scenario_name: string;
      status: string;
      progress_stage: string | null;
      event_kind: string;
      gate_pass: boolean | null;
      total_pnl: number | null;
      note: string | null;
    }>;
  };
  broom: {
    source_report: string;
    source_reports: string[];
    line_count: number;
    total_line_count: number;
    lines: FarmBroomLine[];
  } | null;
  notes: string | null;
};
