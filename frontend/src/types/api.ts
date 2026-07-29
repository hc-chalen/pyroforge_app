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
  scan_summary: Record<string, unknown>;
  structure_comparison?: StructureComparison;
}

export interface VariantResults {
  prediction?: PredictResponse;
  predictionVariant?: SequenceVariant;
  fold?: FoldResult;
  foldVariant?: SequenceVariant;
  registry?: RegistrySearchResponse;
  registryVariant?: SequenceVariant;
  procurement?: ProcureResponse;
  procurementVariant?: SequenceVariant;
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
  evalue: number;
  bits: number;
  description: string;
  biobrick_url: string;
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

export interface NasReportResponse {
  available: boolean;
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
