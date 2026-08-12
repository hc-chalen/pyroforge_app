export interface MutationSchema {
  position: number;
  source_aa: string;
  target_aa: string;
}

export interface PredictRequest {
  sequence: string;
  mutations?: MutationSchema[];
}

export type TmConfidence = "high" | "low" | "unavailable";

export interface PredictResponse {
  // null when no model_tm checkpoint is installed (tm_confidence: "unavailable").
  tm: number | null;
  dg?: number | null;
  ddg?: number | null;
  /** ± half-width of the 95% interval, in the metric's own unit. */
  tm_ci95?: number | null;
  dg_ci95?: number | null;
  ddg_ci95?: number | null;
  /** "low" when the sequence is outside model_tm's training length window. */
  tm_confidence?: TmConfidence | null;
  tm_confidence_reason?: string | null;
  /** "low" when the sequence is outside the small-domain regime dG/ΔΔG were trained on. */
  dg_confidence?: TmConfidence | null;
  dg_confidence_reason?: string | null;
}

export interface ScanSummary {
  total_candidates?: number;
  selected_mutations?: number;
  perplexity_threshold?: number;
  candidates_ddg_positive?: number;
}

/** One shortlist entry, with the evidence behind its position in the ranking. */
export interface RankedMutation {
  position: number;
  source_aa: string;
  target_aa: string;
  ddg: number;
  residue_prob: number;
  rank: number;
}

/**
 * Held-out measurement of what the shortlist ordering is worth.
 *
 * The optimizer returns a *ranking*, not a set of established results, so this
 * travels with every optimization and the UI quotes it verbatim. `top_k_precision`
 * is the fraction of a top-k shortlist that is truly stabilizing, directly
 * comparable to `base_rate_stabilizing`.
 */
export interface RankingQuality {
  n_proteins?: number;
  base_rate_stabilizing?: number;
  median_within_protein_spearman?: number;
  headline_k?: number;
  top_k_precision?: Record<string, number>;
  headline_precision_ci95?: [number, number] | number[];
  headline_precision_min?: number;
  headline_precision_max?: number;
  proteins_below_half_precision?: number;
}

export interface OptimizeRequest {
  sequence: string;
  max_mutations?: number;
  perplexity_threshold?: number;
}

export interface OptimizeResponse {
  job_id: string;
  status: string;
}

export interface MutationResult {
  position: number;
  source_aa: string;
  target_aa: string;
}

export type SequenceVariant = "original" | "optimized";

export interface StructureComparison {
  similarity: number;
  cosine_similarity: number;
  mean_absolute_difference: number;
  l1_distance: number;
  max_absolute_difference: number;
  embedding_mean_abs_original: number;
  embedding_mean_abs_optimized: number;
  embedding_norm_original: number;
  embedding_norm_optimized: number;
  embedding_dim: number;
  sequence_a_length: number;
  sequence_b_length: number;
  sequence_identity: number | null;
  mutation_count: number | null;
}

export interface EmbedCompareRequest {
  sequence_original: string;
  sequence_optimized: string;
}

export type EmbedCompareResponse = StructureComparison;

export interface OptimizationResult {
  wild_type_sequence: string;
  optimized_sequence: string;
  mutations: MutationResult[];
  baseline_tm: number | null;
  ddg: number;
  ddg_ci95?: number | null;
  mutation_details?: RankedMutation[];
  ranking_quality?: RankingQuality | null;
  scan_summary: ScanSummary;
  structure_comparison?: StructureComparison;
}

/**
 * Per-variant results. Each entry records both the variant it was computed for and
 * the exact sequence it was computed on — without the sequence, editing the input
 * leaves the old numbers on screen with no indication they no longer apply.
 */
export interface VariantResults {
  prediction?: PredictResponse;
  predictionVariant?: SequenceVariant;
  predictionSequence?: string;
  fold?: FoldResult;
  foldVariant?: SequenceVariant;
  foldSequence?: string;
  registry?: RegistrySearchResponse;
  registryVariant?: SequenceVariant;
  registrySequence?: string;
  procurement?: ProcureResponse;
  procurementVariant?: SequenceVariant;
  procurementSequence?: string;
}

export interface JobStatus {
  job_id: string;
  status: "pending" | "running" | "completed" | "failed";
  progress?: number;
  result?: unknown;
  error?: string;
}

export interface FoldRequest {
  sequence: string;
}

export interface FoldResponse {
  job_id: string;
  status: string;
}

export interface RmsfEntry {
  chain: string;
  position: number;
  rmsf_proxy: number;
  rigidity: number;
}

export interface FoldResult {
  pdb: string;
  pdb_path: string;
  sequence_length: number;
  rmsf: RmsfEntry[];
}

export interface HeatmapRequest {
  sequence: string;
  residue_scores: Record<string, number>;
}

export interface HeatmapResidue {
  position: number;
  aa: string;
  score: number;
  normalized: number;
  color: string;
}

export interface HeatmapResponse {
  residues: HeatmapResidue[];
}

export interface RegistrySearchRequest {
  sequence: string;
  max_hits?: number;
}

export interface RegistryMatch {
  part_id: string;
  identity: number;
  /** null on the pairwise fallback path — no E-value is computable there. */
  evalue: number | null;
  bits: number;
  description: string;
  biobrick_url: string;
  search_method?: string | null;
}

export interface RegistrySearchResponse {
  matches: RegistryMatch[];
  query_length: number;
}

export interface ProcureRequest {
  protein_sequence: string;
  organism?: string;
}

export interface ProcureResponse {
  protein_sequence: string;
  dna_sequence: string;
  organism: string;
  cai: number;
  /** The codon usage reference set the CAI was scored against. */
  cai_reference: string;
  gc_content: number;
  feasibility_flags: string[];
  price_usd?: number;
  lead_time_days?: number;
  twist_status: string;
}

export interface HealthResponse {
  status: string;
  models_ready: string;
  esm_ready?: string;
}

export interface NasValMetrics {
  mae?: number;
  rmse?: number;
  pearson_r?: number;
}

export interface NasModelWeighting {
  loss_task_weights: string;
  weight_decay?: number;
  weight_decay_label: string;
  search_time_share?: number;
}

export interface NasAlgorithmNames {
  search: string;
  architecture: string;
  activation: string;
  optimizer: string;
  normalization: string;
  loss: string;
}

export interface NasHyperparameters {
  activation?: string;
  optimizer?: string;
  input_normalization?: string;
  loss?: string;
  dropout?: number;
  learning_rate?: number;
  weight_decay?: number;
  batch_size?: number;
  hidden_widths?: number[];
}

export interface NasModelSummary {
  key: string;
  label: string;
  unit: string;
  architecture: string;
  n_layers: number;
  best_budget_epochs?: number;
  time_limit_secs?: number;
  algorithm_names: NasAlgorithmNames;
  weighting: NasModelWeighting;
  val_metrics: NasValMetrics;
  hyperparameters: NasHyperparameters;
}

export interface DdgRankingSummary {
  pooled_pearson_r?: number;
  pooled_mae?: number;
  /** The metric that matches how the optimizer uses model_ddg. */
  within_protein_median_spearman?: number;
  within_protein_n_proteins?: number;
  note?: string;
}

export interface NasReportResponse {
  available: boolean;
  ddg_ranking?: DdgRankingSummary | null;
  method?: string;
  design?: string;
  note?: string;
  total_time_limit_secs?: number;
  time_budget?: Record<string, number>;
  search_budget_weights?: Record<string, number>;
  production_weighting?: {
    thermo_dg?: { dg: number };
    optimization_score?: { ddg: number; formula: string };
    adamw_weight_decay?: number;
  };
  seed?: number;
  device?: string;
  models: NasModelSummary[];
  comparison?: Record<string, unknown>;
}
