function [HResults, results_mat] = fit_microstate_hierarchical_dataset(input_path, varargin)
% FIT_MICROSTATE_HIERARCHICAL_DATASET Hierarchical VB-GMM microstate fit for a manifest of EEG records.
%
%   [HResults, results_mat] = fit_microstate_hierarchical_dataset(manifest_csv, ...)
%
% Pipeline
%   preprocess and filter -> GFP-peak maps -> global VB-GMM fitted for every candidate K, K chosen by the
%   selection criterion (default: free-energy elbow, see select_spm_vb_k_by_criterion) -> group, condition
%   and participant VB-GMM fits at the selected K, each with its parent's maps as component-mean priors.
%
% Input
%   manifest_csv   CSV with a file_path column; participant, condition and group columns are optional.
%
% Labelling
%   Global states can be named by 'global_manual_labels' (labels in fitted-state order, unique letters
%   A-G, any number of states up to 7) or by a saved global/global_manual_labels.mat that matches the
%   fitted maps. Otherwise, with a template file, the template alignment of the global fit is used.
%   Child fits are aligned one-to-one to the global fit and inherit its labels.
%
% Special request: fit_microstate_hierarchical_dataset(struct('command', ...)) runs one of the narrow
% commands 'extract_peak_record' or 'timestamped_transitions' (see run_recovery_command).

    setup_microstate_paths();
    if nargin >= 1 && isstruct(input_path) && isfield(input_path, 'command')
        [HResults, results_mat] = run_recovery_command(input_path);
        return;
    end
    if nargin < 1 || isempty(input_path)
        error('fit_microstate_hierarchical_dataset requires a CSV manifest path.');
    end

    t0 = tic;
    util = microstate_utilities();
    repo_cfg = util.load_config();
    path_defaults = repo_cfg.paths;
    pre_defaults = repo_cfg.preprocessing;
    hier_defaults = repo_cfg.hierarchical;
    max_child_maps_default = double(hier_defaults.max_global_maps);
    if isfield(hier_defaults, 'max_child_maps')
        max_child_maps_default = double(hier_defaults.max_child_maps);
    end
    spm_prior_weight_default = 1;
    if isfield(hier_defaults, 'spm_prior_weight')
        spm_prior_weight_default = double(hier_defaults.spm_prior_weight);
    end

    p = inputParser;
    addRequired(p, 'input_path', @(x) ischar(x) || isstring(x));
    addParameter(p, 'output_dir', path_defaults.hierarchical_output_dir, @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_file', path_defaults.template_file, @(x) ischar(x) || isstring(x));
    addParameter(p, 'template_label_override', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'global_manual_labels', {}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'require_template_labels', false, @islogical);
    addParameter(p, 'template_reference_K', 7, @(x) isnumeric(x) && isscalar(x) && x >= 2);
    addParameter(p, 'resume', true, @islogical);
    addParameter(p, 'K_candidates', double(hier_defaults.K_candidates(:)'), @isnumeric);
    addParameter(p, 'criterion', char(hier_defaults.criterion), @(x) ischar(x) || isstring(x));
    addParameter(p, 'apply_average_reference', logical(pre_defaults.apply_average_reference), @islogical);
    addParameter(p, 'spatial_filter', char(pre_defaults.spatial_filter), @(x) ischar(x) || isstring(x));
    addParameter(p, 'spatial_filter_neighbours', double(pre_defaults.spatial_filter_neighbours), @isnumeric);
    addParameter(p, 'spatial_filter_strength', double(pre_defaults.spatial_filter_strength), @isnumeric);
    addParameter(p, 'filter_band', double(pre_defaults.filter_band(:)'), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'use_scalp_channels', true, @islogical);
    addParameter(p, 'scalp_channel_max_radius', 0.60, @(x) isnumeric(x) && isscalar(x) && x > 0 && x <= 1);
    addParameter(p, 'exclude_channels', {'PO9', 'PO10'}, @(x) iscell(x) || isstring(x));
    addParameter(p, 'interpolate_missing_channels', true, @islogical);
    addParameter(p, 'gfp_peak_min_distance', 3, @isnumeric);
    addParameter(p, 'gfp_peak_threshold_schedule', [0.50 0.60 0.70 0.80 0.90], @isnumeric);
    addParameter(p, 'min_gfp_peaks_per_file', 20, @isnumeric);
    addParameter(p, 'reject_gfp_peak_outliers', logical(pre_defaults.reject_gfp_peak_outliers), @islogical);
    addParameter(p, 'gfp_outlier_mad_multiplier', double(pre_defaults.gfp_outlier_mad_multiplier), @isnumeric);
    addParameter(p, 'max_maps_per_file', [], @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_global_maps', double(hier_defaults.max_global_maps), @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'max_child_maps', max_child_maps_default, @(x) isempty(x) || isnumeric(x));
    addParameter(p, 'spm_prior_weight', spm_prior_weight_default, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'global_only', false, @islogical);
    addParameter(p, 'run_participant_condition_fits', true, @islogical);
    addParameter(p, 'state_covariance_metrics_csv', 'hierarchical_state_covariance_metrics.csv', @(x) ischar(x) || isstring(x));
    addParameter(p, 'compact_results_mat', false, @islogical);
    addParameter(p, 'random_seed', 1, @isnumeric);
    addParameter(p, 'n_starts', 1, @(x) isnumeric(x) && isscalar(x) && x >= 1 && x == round(x));
    addParameter(p, 'verbose', true, @islogical);
    parse(p, input_path, varargin{:});
    cfg = p.Results;

    cfg.input_path = char(cfg.input_path);
    cfg.output_dir = util.resolve_path(cfg.output_dir, util.project_root());
    cfg.template_file = util.resolve_path(cfg.template_file, util.project_root());
    cfg.template_label_override = cellstr(string(cfg.template_label_override));
    cfg.global_manual_labels = cellstr(string(cfg.global_manual_labels));
    cfg.criterion = lower(char(cfg.criterion));
    cfg.spatial_filter = char(cfg.spatial_filter);
    cfg.exclude_channels = cellstr(string(cfg.exclude_channels));
    cfg.state_covariance_metrics_csv = char(cfg.state_covariance_metrics_csv);
    cfg.K_candidates = unique(round(double(cfg.K_candidates(:)')));
    cfg.K_candidates = cfg.K_candidates(cfg.K_candidates >= 2);
    if isempty(cfg.K_candidates)
        error('K_candidates must contain at least one value >= 2.');
    end
    rng(cfg.random_seed, 'twister');

    ensure_dir(cfg.output_dir);
    dirs = make_output_dirs(cfg.output_dir);

    [spm_ok, spm_info] = util.ensure_spm_mix('', path_defaults.spm_mixture_paths, cfg.verbose);
    if ~spm_ok
        error('SPM spm_mix not found. Checked: %s', strjoin(spm_info.attempted, ', '));
    end
    if exist('spm_mix_prior', 'file') ~= 2
        error('spm_mix_prior.m is required for hierarchical fitting and was not found on the path.');
    end

    manifest = read_manifest(cfg.input_path);
    writetable(manifest, fullfile(cfg.output_dir, 'normalised_input_manifest.csv'));
    if cfg.verbose
        fprintf('\nHierarchical VB-GMM microstates\n');
        fprintf('Input rows: %d\nOutput: %s\n', height(manifest), cfg.output_dir);
        fprintf('K candidates: %s (criterion: %s)\n', mat2str(cfg.K_candidates), cfg.criterion);
        fprintf('Excluded channels: %s\n', strjoin(cfg.exclude_channels, ', '));
        fprintf('Channel interpolation: %s\n\n', onoff(cfg.interpolate_missing_channels));
    end

    common_metadata_file = fullfile(cfg.output_dir, 'common_channel_metadata.mat');
    peak_cache_complete = true;
    for cache_i = 1:height(manifest)
        cache_file = fullfile(dirs.peaks, sprintf('%03d_%s_%s_gfp_peaks.mat', cache_i, ...
            clean_key(manifest.participant(cache_i)), clean_key(manifest.condition(cache_i))));
        peak_cache_complete = peak_cache_complete && isfile(cache_file);
    end
    using_complete_peak_cache = cfg.resume && peak_cache_complete && isfile(common_metadata_file);
    if using_complete_peak_cache
        cached_common = load(common_metadata_file, 'common_labels', 'common_can', 'common_chanlocs', 'common_pos');
        common_labels = cached_common.common_labels;
        common_can = cached_common.common_can;
        common_chanlocs = cached_common.common_chanlocs;
        common_pos = cached_common.common_pos;
        meta = repmat(struct(), height(manifest), 1);
        if cfg.verbose
            fprintf('Reusing complete GFP-peak and common-channel cache.\n');
        end
    else
        meta = inspect_files(manifest, cfg, util);
        [common_labels, common_can, common_chanlocs, common_pos] = common_channel_set(meta, cfg);
        save(common_metadata_file, 'common_labels', 'common_can', 'common_chanlocs', 'common_pos', '-v7.3');
    end
    if numel(common_labels) < 8
        error('Only %d common channels remain after filtering/exclusion.', numel(common_labels));
    end
    writetable(table((1:numel(common_labels))', string(common_labels(:)), ...
        'VariableNames', {'index', 'label'}), fullfile(cfg.output_dir, 'common_channels.csv'));
    if cfg.verbose
        fprintf('Common channels: %d\n\n', numel(common_labels));
    end

    peak_records = cell(height(manifest), 1);
    pooled_maps_parts = cell(height(manifest), 1);
    pooled_gfp_parts = cell(height(manifest), 1);
    pooled_row_parts = cell(height(manifest), 1);
    for i = 1:height(manifest)
        if cfg.verbose
            fprintf('[%d/%d] GFP peaks: %s %s\n', i, height(manifest), manifest.participant(i), manifest.condition(i));
        end
        peak_file = fullfile(dirs.peaks, sprintf('%03d_%s_%s_gfp_peaks.mat', i, clean_key(manifest.participant(i)), clean_key(manifest.condition(i))));
        rec = [];
        if cfg.resume && isfile(peak_file)
            cached = load(peak_file, 'rec');
            if isfield(cached, 'rec')
                rec = cached.rec;
            end
        end
        if isempty(rec)
            if using_complete_peak_cache
                error('Complete GFP cache entry is unreadable: %s', peak_file);
            end
            rec = extract_file_peaks(manifest.file_path(i), meta(i), common_can, common_pos, cfg, util);
            save(peak_file, 'rec', 'cfg', '-v7.3');
        end
        peak_records{i} = rec;

        n = size(rec.maps_norm, 1);
        pooled_maps_parts{i} = double(rec.maps_norm);
        pooled_gfp_parts{i} = double(rec.gfp_peak(:));
        rows = table();
        rows.file_index = repmat(i, n, 1);
        rows.participant = repmat(manifest.participant(i), n, 1);
        rows.condition = repmat(manifest.condition(i), n, 1);
        rows.group = repmat(manifest.group(i), n, 1);
        rows.file_path = repmat(manifest.file_path(i), n, 1);
        rows.peak_sample = rec.peak_sample(:);
        rows.gfp = double(rec.gfp_peak(:));
        pooled_row_parts{i} = rows;
    end
    pooled_maps = vertcat(pooled_maps_parts{:});
    pooled_gfp = vertcat(pooled_gfp_parts{:});
    pooled_rows = vertcat(pooled_row_parts{:});
    clear pooled_maps_parts pooled_gfp_parts pooled_row_parts
    pooled_manifest_file = fullfile(cfg.output_dir, 'pooled_gfp_peak_manifest.csv');
    if ~cfg.resume || ~isfile(pooled_manifest_file)
        writetable(pooled_rows, pooled_manifest_file);
    end

    fit_idx = deterministic_subsample(size(pooled_maps, 1), cfg.max_global_maps);
    global_maps = pooled_maps(fit_idx, :);
    global_gfp = pooled_gfp(fit_idx);
    if cfg.verbose && numel(fit_idx) < size(pooled_maps, 1)
        fprintf('Global fit uses deterministic cap: %d/%d peaks\n', numel(fit_idx), size(pooled_maps, 1));
    end

    if cfg.verbose
        fprintf('\nGlobal VB-GMM fit (K candidates %s)...\n', mat2str(cfg.K_candidates));
    end
    global_fit = load_cached_fit(dirs.global, 'global', cfg);
    if isempty(global_fit)
        global_fit = fit_spm_vb_peak_maps(global_maps, global_gfp, cfg.K_candidates, cfg, 'global', [], 'none');
    end
    [manual_labels, global_labels_changed, manual_label_source] = resolve_global_manual_labels( ...
        global_fit, cfg, dirs.global);
    if isempty(manual_labels)
        global_fit = attach_template_alignment_if_available(global_fit, cfg, common_labels, common_chanlocs);
        global_labels_changed = false;
    else
        global_fit = apply_manual_global_labels_by_state_order(global_fit, manual_labels, cfg, common_labels, common_chanlocs);
        global_fit.template_alignment.manual_label_source = manual_label_source;
    end
    global_rows = pooled_rows(fit_idx, :);
    global_rows.cluster_label = assign_by_abs_correlation(global_maps, global_fit.centers);
    save_fit_bundle(dirs.global, 'global', global_fit, global_rows, common_labels, common_chanlocs, cfg);
    if global_labels_changed && cfg.verbose
        fprintf('Global labels changed: reusing GFP preprocessing and fitted maps.\n');
    end

    % Every level below the global fit uses the selected K.
    K = global_fit.K_estimated;
    child_cfg = cfg;
    child_cfg.K_candidates = K;
    if cfg.verbose
        fprintf('Selected K = %d (criterion %s).\n', K, cfg.criterion);
    end

    group_nodes = empty_nodes(size(global_fit.centers, 2));
    condition_nodes = empty_nodes(size(global_fit.centers, 2));
    participant_nodes = empty_nodes(size(global_fit.centers, 2));
    participant_condition_nodes = empty_nodes(size(global_fit.centers, 2));
    group_summary = table();
    condition_summary = table();
    participant_summary = table();
    participant_condition_summary = table();
    if cfg.global_only
        if cfg.verbose
            fprintf('Global-only mode: skipping group, condition and participant fits.\n');
        end
    else
        [group_nodes, group_summary] = fit_group_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, child_cfg, dirs.groups, common_labels, common_chanlocs);
        [condition_nodes, condition_summary] = fit_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, child_cfg, dirs.conditions, common_labels, common_chanlocs);
        [participant_nodes, participant_summary] = fit_participant_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, child_cfg, dirs.participants, common_labels, common_chanlocs);
        if cfg.run_participant_condition_fits
            [participant_condition_nodes, participant_condition_summary] = fit_participant_condition_nodes( ...
                pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, ...
                participant_nodes, child_cfg, dirs.participant_conditions, common_labels, common_chanlocs);
        elseif cfg.verbose
            fprintf('Skipping participant-condition fits.\n');
        end
        assert_fixed_hierarchy_k(K, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes);
    end

    hierarchy_summary = [ ...
        make_summary_row('global', "global", "", "", "", global_fit, height(manifest), size(global_maps, 1), dirs.global); ...
        group_summary; condition_summary; participant_summary; participant_condition_summary];
    writetable(hierarchy_summary, fullfile(cfg.output_dir, 'hierarchical_fit_summary.csv'));
    state_covariance_metrics = compile_state_covariance_metrics(global_fit, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes, size(global_maps, 1), dirs.global);
    state_covariance_metrics_csv = output_csv_path(cfg.output_dir, cfg.state_covariance_metrics_csv);
    if ~isempty(state_covariance_metrics)
        writetable(state_covariance_metrics, state_covariance_metrics_csv);
    end

    HResults = struct();
    HResults.source = 'fit_microstate_hierarchical_dataset';
    HResults.created = char(datetime('now', 'Format', 'yyyyMMdd''T''HHmmss'));
    HResults.cfg = cfg;
    HResults.manifest = manifest;
    HResults.common_labels = common_labels;
    HResults.common_channel_labels = common_labels;
    HResults.common_chanlocs = common_chanlocs;
    HResults.common_pos = common_pos;
    HResults.selected_K = K;
    HResults.k_selection = global_fit.model_comparison;
    HResults.global = build_node('global', "", "all", "", "global", global_fit, size(global_maps, 1), dirs.global);
    HResults.groups = group_nodes;
    HResults.conditions = condition_nodes;
    HResults.participants = participant_nodes;
    HResults.participant_conditions = participant_condition_nodes;
    HResults.hierarchy_summary = hierarchy_summary;
    HResults.state_covariance_metrics = state_covariance_metrics;
    HResults.state_covariance_metrics_csv = state_covariance_metrics_csv;
    HResults.peak_records = peak_records;
    HResults.runtime_s = toc(t0);

    results_mat = fullfile(cfg.output_dir, 'hierarchical_microstate_results.mat');
    if cfg.compact_results_mat
        payload = struct('HResults', compact_hresults_for_save(HResults));
        save(results_mat, '-struct', 'payload', '-v7');
    else
        save(results_mat, 'HResults', '-v7.3');
    end

    if cfg.verbose
        fprintf('\nDone in %.1f s\nResults: %s\n', HResults.runtime_s, results_mat);
        if ~isempty(state_covariance_metrics_csv)
            fprintf('State covariance metrics CSV: %s\n', state_covariance_metrics_csv);
        end
        print_alignment_line('Global', global_fit);
    end
end

function dirs = make_output_dirs(root_dir)
    dirs = struct();
    dirs.root = root_dir;
    dirs.peaks = fullfile(root_dir, 'gfp_peaks');
    dirs.global = fullfile(root_dir, 'global');
    dirs.groups = fullfile(root_dir, 'groups');
    dirs.conditions = fullfile(root_dir, 'conditions');
    dirs.participants = fullfile(root_dir, 'participants');
    dirs.participant_conditions = fullfile(root_dir, 'participant_conditions');
    names = fieldnames(dirs);
    for i = 1:numel(names)
        ensure_dir(dirs.(names{i}));
    end
end

function delimiter = manifest_delimiter(input_path)
    [~, ~, ext] = fileparts(char(input_path));
    if strcmpi(ext, '.tsv')
        delimiter = '\t';
    else
        delimiter = ',';
    end
end

function participant = column_or_infer(T, col, file_path, fallback)
    if isempty(col)
        participant = strings(numel(file_path), 1);
        for i = 1:numel(file_path)
            participant(i) = infer_participant(file_path(i), i);
        end
        return;
    end
    participant = string(T{:, col});
    bad = ismissing(participant) | strlength(strtrim(participant)) == 0;
    for i = find(bad(:))'
        participant(i) = infer_participant(file_path(i), i);
    end
    participant(strlength(strtrim(participant)) == 0) = fallback;
end

function values = column_or_default(T, col, default_value, n)
    if isempty(col)
        values = repmat(string(default_value), n, 1);
    else
        values = string(T{:, col});
        values(ismissing(values) | strlength(strtrim(values)) == 0) = string(default_value);
    end
end

function names = normalise_names(names_in)
    names = lower(regexprep(string(names_in), '[^a-zA-Z0-9]', ''));
end

function idx = find_first(names, choices)
    idx = [];
    choices = lower(regexprep(string(choices), '[^a-zA-Z0-9]', ''));
    for i = 1:numel(choices)
        hit = find(names == choices(i), 1, 'first');
        if ~isempty(hit)
            idx = hit;
            return;
        end
    end
end

function tf = is_absolute_path(pth)
    pth = char(pth);
    tf = startsWith(pth, filesep) || ~isempty(regexp(pth, '^[A-Za-z]:[\\/]', 'once'));
end

function name = path_basename(pth)
    parts = regexp(char(pth), '[\\/]', 'split');
    name = parts{end};
end

function pid = infer_participant(file_path, row_index)
    [parent, stem] = fileparts(char(file_path));
    hit = regexp(stem, '(sub-[A-Za-z0-9]+)', 'match', 'once');
    if isempty(hit)
        [~, hit] = fileparts(parent);
    end
    if isempty(hit)
        hit = sprintf('row%03d', row_index);
    end
    pid = string(hit);
end

function meta = inspect_files(manifest, cfg, util)
    meta = repmat(struct('file_path', "", 'labels', {{}}, 'canonical', {{}}, ...
        'keep_index', [], 'chanlocs', [], 'pos', []), height(manifest), 1);
    for i = 1:height(manifest)
        [data, ~, chanlocs, labels, pos] = load_eeg_file(manifest.file_path(i), cfg, util);
        n = size(data, 1);
        keep = true(n, 1);
        if cfg.use_scalp_channels
            keep = util.scalp_channel_mask(chanlocs, n, cfg.scalp_channel_max_radius);
        end
        can = util.canonical_channel_labels(labels);
        exclude_can = util.canonical_channel_labels(cfg.exclude_channels);
        keep = keep & ~ismember(can(:), exclude_can(:));
        if nnz(keep) < 8
            error('Too few channels remain for %s.', manifest.file_path(i));
        end
        meta(i).file_path = manifest.file_path(i);
        meta(i).labels = labels(keep);
        meta(i).canonical = can(keep);
        meta(i).keep_index = find(keep);
        if ~isempty(chanlocs)
            meta(i).chanlocs = chanlocs(keep);
        end
        if ~isempty(pos)
            meta(i).pos = pos(keep, :);
        end
    end
end

function [common_labels, common_can, common_chanlocs, common_pos] = common_channel_set(meta, cfg)
    n_by_file = arrayfun(@(m) numel(m.canonical), meta);
    if cfg.interpolate_missing_channels
        max_n = max(n_by_file);
        ref_candidates = find(n_by_file == max_n);
        ref = ref_candidates(1);
        common_can = meta(ref).canonical(:);
    else
        common_can = meta(1).canonical(:);
        for i = 2:numel(meta)
            common_can = common_can(ismember(common_can, meta(i).canonical));
        end
        if isempty(common_can)
            error('No common channels across files. Rerun with ''interpolate_missing_channels'', true to use the densest montage and fill missing channels.');
        end
        max_n = max(n_by_file);
        if max_n >= 32 && numel(common_can) < 0.5 * max_n
            error('Strict common-channel intersection kept only %d/%d channels. Rerun with ''interpolate_missing_channels'', true.', numel(common_can), max_n);
        end
        ref = 1;
    end
    common_labels = cell(numel(common_can), 1);
    common_chanlocs = [];
    common_pos = [];
    idx_first = zeros(numel(common_can), 1);
    for c = 1:numel(common_can)
        idx_first(c) = find(strcmp(meta(ref).canonical, common_can{c}), 1, 'first');
        common_labels{c} = meta(ref).labels{idx_first(c)};
    end
    if ~isempty(meta(ref).chanlocs)
        common_chanlocs = meta(ref).chanlocs(idx_first);
    end
    if ~isempty(meta(ref).pos)
        common_pos = meta(ref).pos(idx_first, :);
    end
end

function rec = extract_file_peaks(file_path, meta, common_can, common_pos, cfg, util)
    [data, sfreq, chanlocs, labels, pos] = load_eeg_file(file_path, cfg, util);
    can = util.canonical_channel_labels(labels);
    target_data = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            target_data(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if ~cfg.interpolate_missing_channels
            error('Missing channel %s in %s.', common_can{missing(1)}, file_path);
        end
        target_data = fill_missing_channels(target_data, missing, pos, idx, common_pos, file_path);
    end
    data = target_data;
    good_t = all(isfinite(data), 1);
    data = data(:, good_t);
    Sim = struct();
    Sim.X_noisy = data;
    Sim.X_clean = data;
    Sim.sfreq = sfreq;
    Sim.chanlocs = target_chanlocs(chanlocs, idx, common_can);
    Sim.channel_labels = common_can(:)';
    Sim.pos = common_pos;
    Sim.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule, ...
        'reject_gfp_peak_outliers', cfg.reject_gfp_peak_outliers, ...
        'gfp_outlier_mad_multiplier', cfg.gfp_outlier_mad_multiplier, ...
        'min_peak_count_after_gfp_rejection', cfg.min_gfp_peaks_per_file);
    [maps_norm, peak_sample, gfp_vec, ~, ~, maps_raw, preproc_info] = util.preprocess_maps(Sim);
    gfp_peak = gfp_vec(peak_sample(:));
    keep = deterministic_subsample(size(maps_norm, 1), cfg.max_maps_per_file);

    rec = struct();
    rec.file_path = char(file_path);
    rec.common_labels = Sim.channel_labels;
    rec.sfreq = sfreq;
    rec.maps_norm = single(maps_norm(keep, :));
    rec.maps_raw = single(maps_raw(keep, :));
    rec.peak_sample = peak_sample(keep);
    rec.gfp_peak = gfp_peak(keep);
    rec.n_peaks_raw = numel(peak_sample);
    rec.n_peaks_used = numel(keep);
    rec.preprocessing_info = preproc_info;
    rec.n_interpolated_channels = numel(missing);
    rec.interpolated_channel_labels = common_can(missing);
    rec.excluded_channel_labels = setdiff(meta.labels, Sim.channel_labels, 'stable');
end

function chanlocs_out = target_chanlocs(chanlocs, idx, common_can)
    if isempty(chanlocs) || any(~isfinite(idx))
        chanlocs_out = [];
    else
        chanlocs_out = chanlocs(idx);
    end
    if isempty(chanlocs_out)
        chanlocs_out = repmat(struct('labels', ''), 1, numel(common_can));
        for i = 1:numel(common_can)
            chanlocs_out(i).labels = common_can{i};
        end
    end
end

function data = fill_missing_channels(data, missing, source_pos, direct_idx, target_pos, file_path)
    if isempty(source_pos) || isempty(target_pos)
        error('Cannot interpolate missing channels in %s without channel positions.', file_path);
    end
    available = find(isfinite(direct_idx));
    if numel(available) < 4
        error('Cannot interpolate missing channels in %s with fewer than 4 observed target channels.', file_path);
    end
    source_xyz = source_pos(direct_idx(available), :);
    if size(source_xyz, 1) ~= numel(available) || size(target_pos, 1) < max(missing)
        error('Cannot interpolate missing channels in %s because channel positions are incomplete.', file_path);
    end
    for i = 1:numel(missing)
        t = missing(i);
        d = sqrt(sum((source_xyz - target_pos(t, :)).^2, 2));
        [ds, ord] = sort(d, 'ascend');
        m = min(6, numel(ord));
        ord = ord(1:m);
        ds = ds(1:m);
        if ds(1) <= eps
            w = zeros(m, 1);
            w(1) = 1;
        else
            w = 1 ./ (ds.^2 + eps);
            w = w ./ sum(w);
        end
        data(t, :) = w' * data(available(ord), :);
    end
end

function [data, sfreq, chanlocs, labels, pos, chanloc_source] = load_eeg_file(file_path, ~, util)
    file_path = char(file_path);
    if ~isfile(file_path)
        error('EEG file not found: %s', file_path);
    end
    [~, ~, ext] = fileparts(file_path);
    ext = lower(ext);
    chanlocs = [];
    chanloc_source = 'none';
    sfreq = 250;
    switch ext
        case '.set'
            if exist('pop_loadset', 'file') ~= 2
                error('EEGLAB pop_loadset not found on MATLAB path.');
            end
            EEG = pop_loadset(file_path);
            data = double(EEG.data);
            sfreq = double(EEG.srate);
            chanlocs = EEG.chanlocs;
            chanloc_source = 'file';
        case '.mat'
            S = load(file_path);
            if isfield(S, 'EEG')
                data = double(S.EEG.data);
                if isfield(S.EEG, 'srate'), sfreq = double(S.EEG.srate); end
                if isfield(S.EEG, 'chanlocs')
                    chanlocs = S.EEG.chanlocs;
                    chanloc_source = 'file';
                end
            elseif isfield(S, 'eeg_data')
                data = double(S.eeg_data);
            elseif isfield(S, 'data')
                data = double(S.data);
            else
                error('No EEG data field found in %s.', file_path);
            end
            if isfield(S, 'sfreq'), sfreq = double(S.sfreq); end
            if isfield(S, 'srate'), sfreq = double(S.srate); end
            if isfield(S, 'chanlocs')
                chanlocs = S.chanlocs;
                chanloc_source = 'file';
            end
        otherwise
            [data, sfreq, chanlocs] = load_external_eeg_file(file_path, ext);
            chanloc_source = 'file';
    end
    data = squeeze(data);
    if ~ismatrix(data)
        data = reshape(data, size(data, 1), []);
    end
    if size(data, 1) > size(data, 2) && size(data, 2) < 128
        data = data';
    end
    [chanlocs, used_sidecar] = apply_chanloc_sidecar(chanlocs, file_path, size(data, 1), util);
    if used_sidecar
        chanloc_source = 'sidecar';
    end
    labels = util.channel_labels_from_chanlocs(chanlocs, size(data, 1));
    pos = util.positions_from_chanlocs(chanlocs, size(data, 1));
end

function [data, sfreq, chanlocs] = load_external_eeg_file(file_path, ext)
    errors = {};
    if exist('ft_read_header', 'file') == 2 && exist('ft_read_data', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_fieldtrip_file(file_path);
            return;
        catch ME
            errors{end+1} = sprintf('FieldTrip: %s', ME.message);
        end
    end
    if any(strcmpi(ext, {'.fif', '.fiff'})) && exist('fiff_setup_read_raw', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_mne_fif_file(file_path);
            return;
        catch ME
            errors{end+1} = sprintf('MNE-MATLAB: %s', ME.message);
        end
    end
    if exist('pop_fileio', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_eeglab_import(file_path, @pop_fileio);
            return;
        catch ME
            errors{end+1} = sprintf('EEGLAB pop_fileio: %s', ME.message);
        end
    end
    if exist('pop_biosig', 'file') == 2
        try
            [data, sfreq, chanlocs] = load_eeglab_import(file_path, @pop_biosig);
            return;
        catch ME
            errors{end+1} = sprintf('EEGLAB pop_biosig: %s', ME.message);
        end
    end

    detail = '';
    if ~isempty(errors)
        detail = sprintf(' Reader errors: %s', strjoin(errors, ' | '));
    end
    error(['Unsupported EEG extension or missing reader for %s.%s ', ...
        'Install FieldTrip, EEGLAB FileIO/Biosig, or MNE-MATLAB, or convert to .set while preserving EEG.chanlocs.'], ...
        ext, detail);
end

function [data, sfreq, chanlocs] = load_fieldtrip_file(file_path)
    hdr = ft_read_header(file_path);
    data = double(ft_read_data(file_path));
    sfreq = field_value_or_default(hdr, 'Fs', 250);
    labels = field_value_or_default(hdr, 'label', {});
    labels = cellstr(string(labels(:)));
    keep = fieldtrip_eeg_channel_mask(hdr, numel(labels));
    if any(keep) && numel(keep) == numel(labels)
        data = data(keep, :, :);
        labels = labels(keep);
    end
    chanlocs = chanlocs_from_fieldtrip_header(hdr, labels);
end

function keep = fieldtrip_eeg_channel_mask(hdr, n)
    keep = true(n, 1);
    if n == 0 || ~isfield(hdr, 'chantype') || numel(hdr.chantype) < n
        return;
    end
    types = lower(string(hdr.chantype(:)));
    eeg = types(1:n) == "eeg" | contains(types(1:n), "eeg");
    if any(eeg)
        keep = eeg;
    end
end

function chanlocs = chanlocs_from_fieldtrip_header(hdr, labels)
    chanlocs = blank_chanlocs(labels);
    pos = nan(numel(labels), 3);
    if isfield(hdr, 'elec') && ~isempty(hdr.elec)
        pos = labelled_positions(hdr.elec, labels);
    end
    chanlocs = apply_chanloc_positions(chanlocs, pos);
end

function [data, sfreq, chanlocs] = load_mne_fif_file(file_path)
    raw = fiff_setup_read_raw(file_path);
    [data, ~] = fiff_read_raw_segment(raw, raw.first_samp, raw.last_samp);
    sfreq = double(raw.info.sfreq);
    labels = mne_channel_labels(raw.info);
    keep = mne_eeg_channel_mask(raw.info, numel(labels));
    chanlocs_all = chanlocs_from_mne_info(raw.info, labels);
    if any(keep)
        data = data(keep, :);
        labels = labels(keep); %#ok<NASGU>
        chanlocs = chanlocs_all(keep);
    else
        chanlocs = chanlocs_all;
    end
end

function keep = mne_eeg_channel_mask(info, n)
    keep = true(n, 1);
    if ~isfield(info, 'chs') || numel(info.chs) < n || exist('fiff_define_constants', 'file') ~= 2
        return;
    end
    FIFF = fiff_define_constants();
    if ~isfield(FIFF, 'FIFFV_EEG_CH')
        return;
    end
    kinds = nan(n, 1);
    for i = 1:n
        if isfield(info.chs(i), 'kind') && ~isempty(info.chs(i).kind)
            kinds(i) = double(info.chs(i).kind);
        end
    end
    eeg = kinds == double(FIFF.FIFFV_EEG_CH);
    if any(eeg)
        keep = eeg;
    end
end

function labels = mne_channel_labels(info)
    if isfield(info, 'ch_names') && ~isempty(info.ch_names)
        labels = cellstr(string(info.ch_names(:)));
        return;
    end
    n = numel(info.chs);
    labels = cell(n, 1);
    for i = 1:n
        if isfield(info.chs(i), 'ch_name') && ~isempty(info.chs(i).ch_name)
            labels{i} = char(info.chs(i).ch_name);
        else
            labels{i} = sprintf('Ch%03d', i);
        end
    end
end

function chanlocs = chanlocs_from_mne_info(info, labels)
    chanlocs = blank_chanlocs(labels);
    pos = nan(numel(labels), 3);
    if isfield(info, 'chs')
        n = min(numel(labels), numel(info.chs));
        for i = 1:n
            if isfield(info.chs(i), 'loc') && numel(info.chs(i).loc) >= 3
                pos(i, :) = double(info.chs(i).loc(1:3));
            end
        end
    end
    chanlocs = apply_chanloc_positions(chanlocs, pos);
end

function [data, sfreq, chanlocs] = load_eeglab_import(file_path, loader)
    EEG = loader(file_path);
    data = double(EEG.data);
    sfreq = field_value_or_default(EEG, 'srate', 250);
    if isfield(EEG, 'chanlocs') && ~isempty(EEG.chanlocs)
        chanlocs = EEG.chanlocs;
    else
        chanlocs = blank_chanlocs({});
    end
end

function chanlocs = blank_chanlocs(labels)
    labels = cellstr(string(labels(:)));
    chanlocs = repmat(struct('labels', '', 'X', [], 'Y', [], 'Z', []), 1, numel(labels));
    for i = 1:numel(labels)
        chanlocs(i).labels = labels{i};
    end
end

function chanlocs = apply_chanloc_positions(chanlocs, pos)
    n = min(numel(chanlocs), size(pos, 1));
    for i = 1:n
        xyz = double(pos(i, :));
        if numel(xyz) == 3 && all(isfinite(xyz)) && norm(xyz) > eps
            chanlocs(i).X = xyz(1);
            chanlocs(i).Y = xyz(2);
            chanlocs(i).Z = xyz(3);
        end
    end
end

function pos = labelled_positions(geom, labels)
    pos = nan(numel(labels), 3);
    geom_labels = field_value_or_default(geom, 'label', {});
    geom_labels = cellstr(string(geom_labels(:)));
    coords = [];
    for field_name = {'chanpos', 'elecpos', 'pnt'}
        if isfield(geom, field_name{1}) && size(geom.(field_name{1}), 2) >= 3
            coords = double(geom.(field_name{1})(:, 1:3));
            break;
        end
    end
    if isempty(coords)
        return;
    end
    if isempty(geom_labels) && size(coords, 1) == numel(labels)
        pos = coords(1:numel(labels), :);
        return;
    end
    can_geom = simple_canonical_labels(geom_labels);
    can_labels = simple_canonical_labels(labels);
    for i = 1:numel(labels)
        idx = find(strcmp(can_geom, can_labels{i}), 1, 'first');
        if ~isempty(idx) && idx <= size(coords, 1)
            pos(i, :) = coords(idx, :);
        end
    end
end

function can = simple_canonical_labels(labels)
    labels = cellstr(string(labels(:)));
    can = cell(size(labels));
    for i = 1:numel(labels)
        s = lower(strtrim(labels{i}));
        s = regexprep(s, '\s+', '');
        s = regexprep(s, '^eeg', '');
        can{i} = regexprep(s, '[^a-z0-9]', '');
    end
end

function value = field_value_or_default(S, field_name, default_value)
    value = default_value;
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    end
end

function [chanlocs_out, used_sidecar] = apply_chanloc_sidecar(chanlocs_in, file_path, n_channels, util)
    chanlocs_out = chanlocs_in;
    used_sidecar = false;

    sidecar = load_chanloc_sidecar(file_path);
    if isempty(sidecar) || ~has_usable_chanloc_positions(sidecar, numel(sidecar), util)
        return;
    end

    current_labels = util.channel_labels_from_chanlocs(chanlocs_in, n_channels);
    side_labels = util.channel_labels_from_chanlocs(sidecar, numel(sidecar));
    current_can = util.canonical_channel_labels(current_labels);
    side_can = util.canonical_channel_labels(side_labels);

    if isempty(chanlocs_out) || numel(chanlocs_out) < n_channels
        chanlocs_out = blank_chanlocs(current_labels);
    end
    chanlocs_out = ensure_xyz_fields(chanlocs_out);

    n_hit = 0;
    for i = 1:n_channels
        j = find(strcmp(side_can, current_can{i}), 1, 'first');
        if isempty(j) && numel(sidecar) == n_channels
            j = i;
        end
        if isempty(j)
            continue;
        end
        [chanlocs_out(i), ok] = copy_chanloc_position(sidecar(j), chanlocs_out(i));
        if ok
            n_hit = n_hit + 1;
        end
    end
    used_sidecar = n_hit >= max(4, ceil(0.5 * n_channels));
    if ~used_sidecar
        chanlocs_out = chanlocs_in;
    end
end

function tf = has_usable_chanloc_positions(chanlocs, n_channels, util)
    tf = false;
    if isempty(chanlocs)
        return;
    end
    pos = util.positions_from_chanlocs(chanlocs, n_channels);
    good = all(isfinite(pos), 2);
    if sum(good) < 4
        return;
    end
    tf = norm(max(pos(good, :), [], 1) - min(pos(good, :), [], 1)) > eps;
end

function chanlocs = load_chanloc_sidecar(file_path)
    chanlocs = [];
    [folder, stem] = fileparts(file_path);
    candidates = { ...
        fullfile(folder, [stem '_chanlocs.mat']), ...
        fullfile(folder, [stem '_montage.mat'])};
    for i = 1:numel(candidates)
        if ~isfile(candidates{i})
            continue;
        end
        S = load(candidates{i});
        if isfield(S, 'chanlocs') && ~isempty(S.chanlocs)
            chanlocs = S.chanlocs;
            return;
        end
        if isfield(S, 'labels') && isfield(S, 'pos')
            chanlocs = blank_chanlocs(cellstr(string(S.labels)));
            chanlocs = apply_chanloc_positions(chanlocs, double(S.pos));
            return;
        end
    end
end

function [dst, ok] = copy_chanloc_position(src, dst)
    for field_name = {'X', 'Y', 'Z'}
        f = field_name{1};
        if isfield(src, f) && ~isempty(src.(f))
            dst.(f) = src.(f);
        end
    end
    xyz = [field_double(dst, 'X'), field_double(dst, 'Y'), field_double(dst, 'Z')];
    ok = all(isfinite(xyz)) && norm(xyz) > eps;
    if ok
        for field_name = {'theta', 'radius', 'sph_theta', 'sph_phi', 'sph_radius'}
            f = field_name{1};
            if isfield(dst, f)
                dst.(f) = [];
            end
        end
    end
end

function chanlocs = ensure_xyz_fields(chanlocs)
    for field_name = {'X', 'Y', 'Z'}
        f = field_name{1};
        if ~isfield(chanlocs, f)
            [chanlocs.(f)] = deal([]);
        end
    end
end

function value = field_double(S, field_name)
    value = NaN;
    if isfield(S, field_name) && ~isempty(S.(field_name))
        tmp = double(S.(field_name));
        value = tmp(1);
    end
end

function [nodes, summary] = fit_group_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, cfg, out_root, common_labels, common_chanlocs)
    levels = meaningful_levels(pooled_rows.group, "all");
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        level = levels(i);
        mask = pooled_rows.group == level;
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);
        out_dir = fullfile(out_root, clean_key(level));
        ensure_dir(out_dir);
        Fit = load_cached_fit(out_dir, 'group', cfg);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, ['group_' char(level)], global_fit.centers, 'global');
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'group', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('group', "", level, "", level, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('group', level, level, "", "", Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Group ' char(level)], Fit);
    end
end

function [nodes, summary] = fit_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, global_fit, cfg, out_root, common_labels, common_chanlocs)
    levels = meaningful_levels(pooled_rows.condition, "condition");
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        level = levels(i);
        mask = pooled_rows.condition == level;
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);
        out_dir = fullfile(out_root, clean_key(level));
        ensure_dir(out_dir);
        Fit = load_cached_fit(out_dir, 'condition', cfg);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, ['condition_' char(level)], global_fit.centers, 'global');
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'condition', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('condition', "", "", level, level, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('condition', level, "", level, "", Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Condition ' char(level)], Fit);
    end
end

function [nodes, summary] = fit_participant_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, cfg, out_root, common_labels, common_chanlocs)
    levels = unique(pooled_rows.participant, 'stable');
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        participant = levels(i);
        mask = pooled_rows.participant == participant;
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);
        row_mask = manifest.participant == participant;
        group_value = manifest.group(find(row_mask, 1, 'first'));
        [prior_maps, prior_source] = parent_prior_maps(global_fit, group_nodes, group_value);
        out_dir = fullfile(out_root, clean_key(participant));
        ensure_dir(out_dir);
        Fit = load_cached_fit(out_dir, 'participant', cfg);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, ['participant_' char(participant)], prior_maps, prior_source);
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'participant', Fit, rows, common_labels, common_chanlocs, cfg);
        nodes(end+1) = build_node('participant', participant, group_value, "", participant, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('participant', participant, group_value, "", participant, Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Participant ' char(participant)], Fit);
    end
end

function [nodes, summary] = fit_participant_condition_nodes(pooled_maps, pooled_gfp, pooled_rows, manifest, global_fit, group_nodes, condition_nodes, participant_nodes, cfg, out_root, common_labels, common_chanlocs)
    key = string(pooled_rows.participant) + "|" + string(pooled_rows.condition);
    levels = unique(key, 'stable');
    nodes = empty_nodes(size(global_fit.centers, 2));
    summary = table();
    for i = 1:numel(levels)
        parts = split(levels(i), "|");
        participant = parts(1);
        condition = parts(2);
        mask = pooled_rows.participant == participant & pooled_rows.condition == condition;
        if ~any(mask)
            continue;
        end
        fit_idx = capped_mask_indices(mask, cfg.max_child_maps);

        row_mask = manifest.participant == participant & manifest.condition == condition;
        if ~any(row_mask)
            row_mask = manifest.participant == participant;
        end
        group_value = "all";
        if any(row_mask)
            group_value = manifest.group(find(row_mask, 1, 'first'));
        end

        g = find_node(group_nodes, "", group_value, "");
        if ~isempty(g)
            prior_maps = g.fit.centers;
            prior_source = 'group';
        else
            prior_maps = global_fit.centers;
            prior_source = 'global';
        end
        c = find_node(condition_nodes, "", "", condition);
        if ~isempty(c)
            prior_maps = c.fit.centers;
            prior_source = 'condition';
        end
        p = find_node(participant_nodes, participant, "", "");
        if ~isempty(p)
            prior_maps = p.fit.centers;
            prior_source = 'participant';
        end

        out_dir = fullfile(out_root, clean_key(participant), clean_key(condition));
        ensure_dir(out_dir);
        fit_name = ['participant_condition_' char(participant) '_' char(condition)];
        Fit = load_cached_fit(out_dir, 'participant_condition', cfg);
        if isempty(Fit)
            Fit = fit_spm_vb_peak_maps(pooled_maps(fit_idx, :), pooled_gfp(fit_idx), cfg.K_candidates, cfg, fit_name, prior_maps, prior_source);
        end
        Fit = align_fit_to_reference(Fit, global_fit);
        rows = pooled_rows(fit_idx, :);
        rows.cluster_label = assign_by_abs_correlation(pooled_maps(fit_idx, :), Fit.centers);
        save_fit_bundle(out_dir, 'participant_condition', Fit, rows, common_labels, common_chanlocs, cfg);

        node_name = participant + "_" + condition;
        nodes(end+1) = build_node('participant_condition', participant, group_value, condition, node_name, Fit, numel(fit_idx), out_dir); %#ok<AGROW>
        summary = [summary; make_summary_row('participant_condition', node_name, group_value, condition, participant, Fit, numel(unique(rows.file_index)), numel(fit_idx), out_dir)]; %#ok<AGROW>
        print_alignment_line(['Participant-condition ' char(participant) ' ' char(condition)], Fit);
    end
end

function [prior_maps, prior_source] = parent_prior_maps(global_fit, group_nodes, group_value)
    prior_maps = global_fit.centers;
    prior_source = 'global';
    g = find_node(group_nodes, "", group_value, "");
    if ~isempty(g)
        prior_maps = g.fit.centers;
        prior_source = 'group';
    end
end

function rec = find_node(nodes, participant, group, condition)
    rec = [];
    for i = 1:numel(nodes)
        if (strlength(participant) == 0 || nodes(i).participant == participant) && ...
                (strlength(group) == 0 || nodes(i).group == group) && ...
                (strlength(condition) == 0 || nodes(i).condition == condition)
            rec = nodes(i);
            return;
        end
    end
end

function out = output_csv_path(output_dir, csv_name)
    out = char(string(csv_name));
    if isempty(out)
        out = output_dir;
    elseif ~is_absolute_path(out)
        out = fullfile(output_dir, out);
    end
end

function T = compile_state_covariance_metrics(global_fit, group_nodes, condition_nodes, participant_nodes, participant_condition_nodes, global_n_maps, global_fit_path)
    rows = {state_covariance_rows_for_fit("global", "global", "all", "", "", global_fit, global_n_maps, global_fit_path)};
    rows = append_node_covariance_rows(rows, group_nodes);
    rows = append_node_covariance_rows(rows, condition_nodes);
    rows = append_node_covariance_rows(rows, participant_nodes);
    rows = append_node_covariance_rows(rows, participant_condition_nodes);
    T = vertcat_nonempty(rows);
    if ~isempty(T)
        T = sortrows(T, {'level', 'group', 'condition', 'participant', 'template_label', 'state_index'});
    end
end

function rows = append_node_covariance_rows(rows, nodes)
    for i = 1:numel(nodes)
        if isfield(nodes(i), 'fit') && isstruct(nodes(i).fit) && isfield(nodes(i).fit, 'centers') && ~isempty(nodes(i).fit.centers)
            rows{end+1, 1} = state_covariance_rows_for_fit(nodes(i).level, nodes(i).name, ...
                nodes(i).group, nodes(i).condition, nodes(i).participant, nodes(i).fit, nodes(i).n_maps, nodes(i).fit_path); %#ok<AGROW>
        end
    end
end

function T = state_covariance_rows_for_fit(level, name, group, condition, participant, Fit, n_maps, fit_path)
    K = size(Fit.centers, 1);
    [state_labels, state_order, template_corr] = state_labels_for_metrics(Fit);
    summary = selected_mix_summary(Fit);
    feature_dim = double(get_field_or(summary, 'feature_dim', NaN));
    free_energy = double(get_field_or(summary, 'free_energy', NaN));
    priors = state_vector(get_field_or(summary, 'priors', []), K);
    traces = state_vector(get_field_or(summary, 'covariance_traces', []), K);
    logdets = state_vector(get_field_or(summary, 'covariance_logdets', []), K);
    cluster_weights = state_vector(get_field_or(Fit, 'cluster_weights', []), K);
    trace_per_feature = traces ./ max(feature_dim, eps);
    trace_per_feature(~isfinite(trace_per_feature)) = NaN;
    logdet_per_feature = logdets ./ max(feature_dim, eps);
    logdet_per_feature(~isfinite(logdet_per_feature)) = NaN;

    rows = cell(numel(state_order), 1);
    for j = 1:numel(state_order)
        k = state_order(j);
        rows{j} = table(string(level), string(name), string(group), string(condition), string(participant), ...
            double(k), string(state_labels{k}), double(K), double(n_maps), double(feature_dim), double(free_energy), ...
            double(priors(k)), double(cluster_weights(k)), double(traces(k)), double(trace_per_feature(k)), ...
            double(logdets(k)), double(logdet_per_feature(k)), double(template_corr(k)), string(fit_path), ...
            'VariableNames', {'level', 'name', 'group', 'condition', 'participant', ...
            'state_index', 'template_label', 'K_estimated', 'n_maps', 'feature_dim', 'free_energy', ...
            'spm_prior', 'cluster_weight', 'covariance_trace', 'covariance_trace_per_feature', ...
            'covariance_logdet', 'covariance_logdet_per_feature', 'template_match_abs_correlation', 'fit_path'});
    end
    T = vertcat(rows{:});
end

function summary = selected_mix_summary(Fit)
    if isfield(Fit, 'selected_spm_mix_model') && isstruct(Fit.selected_spm_mix_model)
        summary = Fit.selected_spm_mix_model;
    else
        summary = empty_mix_summary();
    end
end

function v = state_vector(x, K)
    v = nan(K, 1);
    x = double(x(:));
    n = min(K, numel(x));
    if n > 0
        v(1:n) = x(1:n);
    end
end

function value = get_field_or(S, field_name, fallback)
    if isstruct(S) && isfield(S, field_name) && ~isempty(S.(field_name))
        value = S.(field_name);
    else
        value = fallback;
    end
end

function T = vertcat_nonempty(rows)
    rows = rows(~cellfun(@isempty, rows));
    if isempty(rows)
        T = table();
    else
        T = vertcat(rows{:});
    end
end

function [Sim_backfit, gfp, X_metric] = prepare_backfit_record(file_path, common_can, common_pos, cfg, util)
    [data, sfreq, ~, labels, pos] = load_eeg_file(file_path, cfg, util);
    can = util.canonical_channel_labels(labels);
    target_data = nan(numel(common_can), size(data, 2));
    idx = nan(numel(common_can), 1);
    for c = 1:numel(common_can)
        local = find(strcmp(can, common_can{c}), 1, 'first');
        if ~isempty(local)
            idx(c) = local;
            target_data(c, :) = double(data(local, :));
        end
    end
    missing = find(~isfinite(idx));
    if ~isempty(missing)
        if ~cfg.interpolate_missing_channels
            error('Missing channel %s in %s.', common_can{missing(1)}, file_path);
        end
        target_data = fill_missing_channels(target_data, missing, pos, idx, common_pos, file_path);
    end
    good_t = all(isfinite(target_data), 1);
    target_data = target_data(:, good_t);

    Sim_backfit = struct();
    Sim_backfit.X_noisy = target_data;
    Sim_backfit.sfreq = sfreq;
    Sim_backfit.channel_labels = common_can(:)';
    Sim_backfit.chanlocs = [];
    Sim_backfit.pos = common_pos;
    Sim_backfit.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'spatial_filter', cfg.spatial_filter, ...
        'spatial_filter_neighbours', cfg.spatial_filter_neighbours, ...
        'spatial_filter_strength', cfg.spatial_filter_strength, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule);
    X_metric = preprocess_backfit_metric_matrix(Sim_backfit, util);
    gfp = std(X_metric, 0, 1, 'omitnan')';
end

function X = preprocess_backfit_metric_matrix(Sim_backfit, util)
    X = double(Sim_backfit.X_noisy);
    if isfield(Sim_backfit.preprocessing, 'apply_average_reference') && Sim_backfit.preprocessing.apply_average_reference
        X = X - mean(X, 1, 'omitnan');
    end
    [X, ~] = util.apply_spatial_filter(X, Sim_backfit);
    if isfield(Sim_backfit.preprocessing, 'filter_band') && ~isempty(Sim_backfit.preprocessing.filter_band)
        X = util.bandpass_filter(X, Sim_backfit.sfreq, Sim_backfit.preprocessing.filter_band);
    end
end

function hash = sha256_array(values)
    if isempty(values)
        hash = "none";
        return;
    end
    bytes = typecast(double(values(:)), 'uint8');
    digest = java.security.MessageDigest.getInstance('SHA-256');
    digest.update(bytes);
    hash = string(lower(reshape(dec2hex(typecast(digest.digest(), 'uint8'), 2)', 1, [])));
end

function [state_labels, state_order, template_corr] = state_labels_for_metrics(Results)
    K = size(Results.centers, 1);
    state_labels = arrayfun(@(k) sprintf('state_%02d', k), 1:K, 'UniformOutput', false);
    template_corr = nan(K, 1);
    if isfield(Results, 'template_alignment') && isstruct(Results.template_alignment)
        A = Results.template_alignment;
        if isfield(A, 'labels') && numel(A.labels) >= K
            state_labels = cellstr(string(A.labels(:)));
        end
        if isfield(A, 'correlations') && numel(A.correlations) >= K
            template_corr = abs(double(A.correlations(:)));
        end
    end
    [~, state_order] = sort(lower(string(state_labels)));
end

function levels = meaningful_levels(values, default_value)
    levels = unique(string(values), 'stable');
    levels = levels(~ismissing(levels) & strlength(strtrim(levels)) > 0);
    levels = levels(levels ~= string(default_value));
    levels = levels(:)';
end

function Fit = fit_spm_vb_peak_maps(maps, gfp, K_candidates, cfg, fit_name, prior_maps, prior_source)
    if nargin < 7 || isempty(prior_source)
        prior_source = 'unspecified';
    end
    t0 = tic;
    X = normalize_maps(double(maps));
    gfp = double(gfp(:));
    if numel(gfp) ~= size(X, 1)
        gfp = ones(size(X, 1), 1);
    end
    ok = all(isfinite(X), 2) & isfinite(gfp) & gfp > 0;
    X = X(ok, :);
    gfp = gfp(ok);
    K_requested = unique(round(double(K_candidates(:)')));
    K_candidates = K_requested(K_requested >= 2 & K_requested < size(X, 1));
    if isempty(K_candidates)
        error('No valid K candidates for %s with %d maps.', fit_name, size(X, 1));
    end
    if numel(K_candidates) < numel(K_requested)
        warning('K candidates %s dropped for %s: a K-component fit needs more than K maps (%d available).', ...
            mat2str(setdiff(K_requested, K_candidates)), fit_name, size(X, 1));
    end

    prior_maps = normalize_maps(double(prior_maps));
    nK = numel(K_candidates);
    free_energy = -inf(nK, 1);
    silhouette_vals = nan(nK, 1);
    gev_vals = nan(nK, 1);
    wss_vals = inf(nK, 1);
    starts_free_energy = nan(nK, max(1, cfg.n_starts));
    centers_by_K = cell(nK, 1);
    labels_by_K = cell(nK, 1);
    prior_beta_by_K = cell(nK, 1);
    summaries = repmat(empty_mix_summary(), nK, 1);

    for iK = 1:nK
        K = K_candidates(iK);
        try
            prior_K = ensure_k_prior_maps(prior_maps, X, K);
            [features, feature_info, prior_features] = spm_features(X, prior_K);
            if size(features, 1) ~= size(X, 1)
                error('Feature observation count differs from the %d real GFP-peak maps for %s.', size(X, 1), fit_name);
            end
            if isempty(prior_K)
                % Without a parent solution SPM starts from k-means, which is random: fit from n_starts seeded
                % starts and keep the one with the highest free energy.
                [mix, starts_free_energy(iK, :)] = best_of_starts(features, K, double(cfg.random_seed), cfg.n_starts);
            else
                % Seeded by the parent solution, so there is nothing random to restart.
                rng(double(cfg.random_seed) + K, 'twister');
                mix = spm_mix_prior(features, K, 0, ...
                    struct('means', prior_features, 'mean_weight', cfg.spm_prior_weight));
                beta_0 = double(mix.prior.beta_0(:));
                if numel(beta_0) ~= K || any(abs(beta_0 - cfg.spm_prior_weight) > 100 * eps(max(1, cfg.spm_prior_weight)))
                    error('SPM mean-prior weights were not retained for %s K=%d.', fit_name, K);
                end
                prior_beta_by_K{iK} = beta_0;
            end
            if ~valid_mix(mix, K)
                continue;
            end
            labels0 = assign_mix(features, mix);
            centers0 = recover_centers(X, labels0, K, prior_K);
            centers = refine_centers(X, centers0, prior_K, 20);
            labels = assign_by_abs_correlation(X, centers);
            metrics = topo_metrics(X, labels, centers, gfp);

            free_energy(iK) = double(real(mix.fm));
            if ~isempty(prior_K)
                starts_free_energy(iK, 1) = free_energy(iK);
            end
            silhouette_vals(iK) = metrics.silhouette;
            gev_vals(iK) = metrics.gev;
            wss_vals(iK) = metrics.wss;
            centers_by_K{iK} = centers;
            labels_by_K{iK} = labels;
            summaries(iK) = summarise_mix(mix, K, size(features, 2), feature_info);
        catch ME
            warning('VB-GMM fit failed for %s K=%d: %s', fit_name, K, ME.message);
        end
    end

    valid = isfinite(free_energy) & ~cellfun(@isempty, centers_by_K);
    if ~any(valid)
        error('All VB-GMM fits failed for %s.', fit_name);
    end
    selection_payload = struct('K_candidates', K_candidates(:), ...
        'free_energy_vals', free_energy(:), ...
        'silhouette_vals', silhouette_vals(:), ...
        'gev_vals', gev_vals(:), ...
        'calinski_harabasz_vals', nan(nK, 1), ...
        'spm_mix_model_summaries', summaries);
    [K_est, best_score, score_by_k, selection_details] = select_spm_vb_k_by_criterion(selection_payload, cfg.criterion);
    if ~isfinite(K_est)
        error('K selection by criterion "%s" failed for %s.', cfg.criterion, fit_name);
    end
    best_idx = find(K_candidates == K_est, 1, 'first');

    model_comparison = table(K_candidates(:), free_energy(:), score_by_k(:), K_candidates(:) == K_est, ...
        silhouette_vals(:), gev_vals(:), wss_vals(:), ...
        'VariableNames', {'K', 'free_energy', 'selection_score', 'selected', 'silhouette', 'gev', 'wss'});

    Fit = struct();
    Fit.name = char(fit_name);
    Fit.method = 'spm_vb_hierarchical';
    Fit.k_selection_rule = selection_details.rule;
    Fit.criterion = cfg.criterion;
    Fit.K_candidates = K_candidates;
    Fit.K_estimated = K_est;
    Fit.best_index = best_idx;
    Fit.best_criterion_value = best_score;
    Fit.centers = centers_by_K{best_idx};
    Fit.labels = labels_by_K{best_idx};
    Fit.cluster_weights = cluster_weights(Fit.labels, K_est);
    Fit.maps_nc = single(X);
    Fit.backfit_peak_labels = double(Fit.labels(:));
    Fit.preprocessing = struct( ...
        'apply_average_reference', cfg.apply_average_reference, ...
        'filter_band', cfg.filter_band, ...
        'gfp_peak_min_distance', cfg.gfp_peak_min_distance, ...
        'gfp_peak_threshold_schedule', cfg.gfp_peak_threshold_schedule);
    Fit.free_energy_vals = free_energy;
    Fit.free_energy_starts = starts_free_energy;
    Fit.n_starts = max(1, cfg.n_starts);
    Fit.random_seed = double(cfg.random_seed);
    Fit.silhouette_vals = silhouette_vals;
    Fit.gev_vals = gev_vals;
    Fit.within_ss = wss_vals;
    Fit.selection_score_by_k = score_by_k;
    Fit.selection_details = selection_details;
    Fit.model_comparison = model_comparison;
    Fit.spm_mix_model_summaries = summaries;
    Fit.selected_spm_mix_model = summaries(best_idx);
    Fit.centers_by_K = centers_by_K;
    Fit.labels_by_K = labels_by_K;
    Fit.n_maps = size(X, 1);
    Fit.real_observation_count = size(X, 1);
    Fit.n_channels = size(X, 2);
    Fit.prior_n_maps = size(prior_maps, 1);
    Fit.prior_source = char(prior_source);
    Fit.prior_state_index = 1:Fit.K_estimated;
    Fit.parent_map_sha256 = sha256_array(prior_maps);
    if isempty(prior_maps)
        Fit.prior_mode = 'none';
    else
        Fit.prior_mode = 'spm_mix_prior_component_means';
    end
    Fit.spm_prior_weight = cfg.spm_prior_weight;
    Fit.prior_beta_0 = prior_beta_by_K{best_idx};
    Fit.spm_version = spm('Ver');
    if Fit.n_maps ~= size(X, 1)
        error('Child observation count changed during hierarchical fitting.');
    end
    Fit.runtime = toc(t0);
end

function [output, detail] = run_recovery_command(request)
% Narrow requests answered from a prepared hierarchy.
%   'extract_peak_record'      GFP-peak maps of one EEG file for the common channel set
%   'timestamped_transitions'  exact sample-level state changes of one EEG file under a fitted solution
    command = lower(strtrim(char(string(request.command))));
    detail = '';
    util = microstate_utilities();
    switch command
        case 'extract_peak_record'
            required = {'file_path','common_can','common_pos','cfg'};
            for i = 1:numel(required)
                if ~isfield(request, required{i})
                    error('Request is missing %s.', required{i});
                end
            end
            manifest = table(string(request.file_path), 'VariableNames', {'file_path'});
            meta = inspect_files(manifest, request.cfg, util);
            output = extract_file_peaks(request.file_path, meta(1), request.common_can, ...
                request.common_pos, request.cfg, util);
        case 'timestamped_transitions'
            required = {'file_path','common_can','common_pos','cfg','fit'};
            for i = 1:numel(required)
                if ~isfield(request, required{i})
                    error('Request is missing %s.', required{i});
                end
            end
            [Sim, ~, X_metric] = prepare_backfit_record(request.file_path, request.common_can, ...
                request.common_pos, request.cfg, util);
            validate_full = isfield(request, 'validate_full_backfit') && request.validate_full_backfit;
            sfreq = double(Sim.sfreq);
            if ~validate_full
                Sim.X_noisy = [];
            end
            gfp = sqrt(mean((double(X_metric) - mean(double(X_metric), 1)).^2, 1));
            min_dist = max(1, round(double(request.cfg.gfp_peak_min_distance)));
            peak_sample = exact_local_peak_finder(gfp, min_dist);
            peak_maps = util.normalize_maps(double(X_metric(:, peak_sample))');
            centers = util.normalize_maps(double(request.fit.centers));
            [~, peak_assignment] = max(abs(peak_maps * centers'), [], 2);
            changes = find(peak_assignment(2:end) ~= peak_assignment(1:end-1));
            transition_sample = floor((peak_sample(changes) + peak_sample(changes + 1)) / 2) + 1;
            output = struct( ...
                'file_path', char(request.file_path), ...
                'sfreq', sfreq, ...
                'n_samples', double(size(X_metric, 2)), ...
                'peak_sample', uint32(peak_sample(:)), ...
                'peak_assignment', uint8(peak_assignment(:)), ...
                'transition_sample', uint32(transition_sample(:)), ...
                'from_state', uint8(peak_assignment(changes)), ...
                'to_state', uint8(peak_assignment(changes + 1)));
            output.full_backfit_validation = false;
            if validate_full
                backfit = backfit_microstate_timecourse(Sim, request.fit);
                if ~backfit.ok || ~strcmp(backfit.hard.mode, 'gfp_peaks_interpolated')
                    error('Full backfit validation failed for %s: %s', request.file_path, backfit.message);
                end
                full_assignment = uint8(backfit.hard.assignments(:));
                full_sample = uint32(find(full_assignment(2:end) ~= full_assignment(1:end-1)) + 1);
                if ~isequal(full_sample(:), output.transition_sample(:)) || ...
                        ~isequal(full_assignment(double(full_sample) - 1), output.from_state(:)) || ...
                        ~isequal(full_assignment(double(full_sample)), output.to_state(:))
                    error('Compact transition export disagrees with the full hard backfit for %s.', request.file_path);
                end
                output.full_backfit_validation = true;
            end
        otherwise
            error('Unknown command: %s', command);
    end
end

function idx = exact_local_peak_finder(x, min_dist)
% Identical selection rule to backfit_microstate_timecourse.
    x = double(x(:)');
    idx = [];
    if numel(x) < 3, return; end
    cand = find(x(2:end-1) >= x(1:end-2) & x(2:end-1) > x(3:end)) + 1;
    if isempty(cand)
        [~, idx] = max(x);
        return;
    end
    [~, order] = sort(x(cand), 'descend');
    cand = cand(order);
    taken = false(size(x));
    keep = false(size(cand));
    for i = 1:numel(cand)
        c = cand(i);
        bounds = max(1, c - min_dist):min(numel(x), c + min_dist);
        if any(taken(bounds)), continue; end
        keep(i) = true;
        taken(bounds) = true;
    end
    idx = sort(cand(keep), 'ascend');
end

function prior_K = ensure_k_prior_maps(prior_maps, X, K)
    if isempty(prior_maps)
        prior_K = [];
        return;
    end
    prior_K = normalize_maps(prior_maps);
    if size(prior_K, 1) ~= K
        error('A hierarchical parent must contain exactly K=%d maps; received %d.', K, size(prior_K, 1));
    end
    if size(prior_K, 2) ~= size(X, 2)
        error('Parent and child maps must have the same channel count (%d versus %d).', size(prior_K, 2), size(X, 2));
    end
end

function [features, info, seed_features] = spm_features(X, seed_maps)
    if nargin < 2 || isempty(seed_maps)
        seed_maps = [];
    else
        seed_maps = normalize_maps(seed_maps);
    end
    X = normalize_maps(X);
    x_mean = mean(X, 1);
    Xc = X - x_mean;
    if isempty(seed_maps)
        seed_Xc = zeros(0, size(X, 2));
    else
        seed_Xc = seed_maps - x_mean;
    end
    [~, S, V] = svd(Xc, 'econ');
    latent = diag(S).^2 ./ max(1, size(Xc, 1) - 1);
    total = sum(latent);
    if total <= eps
        linear = Xc(:, 1:min(size(Xc, 2), 2));
        seed_linear = seed_Xc(:, 1:size(linear, 2));
        rank_est = size(linear, 2);
        var_pct = 0;
    else
        rank_est = nnz(latent > max(size(Xc)) * eps(max(latent)));
        cumv = cumsum(latent) ./ total;
        nd = find(cumv >= 0.999, 1, 'first');
        if isempty(nd), nd = rank_est; end
        nd = max(1, min([nd, rank_est, size(Xc, 1) - 1, size(Xc, 2) - 1, 8]));
        linear = Xc * V(:, 1:nd);
        seed_linear = seed_Xc * V(:, 1:nd);
        var_pct = 100 * cumv(nd);
    end
    sd = std(linear, 0, 1);
    sd(sd < eps) = 1;
    linear = linear ./ sd;
    seed_linear = seed_linear ./ sd;

    Y = normalize_rows(linear);
    seed_Y = normalize_rows(seed_linear);
    features = projective_features(Y);
    seed_features = projective_features(seed_Y);
    feature_mean = mean(features, 1);
    features = features - feature_mean;
    seed_features = seed_features - feature_mean;
    fsd = std(features, 0, 1);
    keep = fsd > 10 * eps;
    features = features(:, keep);
    seed_features = seed_features(:, keep);
    fsd = fsd(keep);
    if isempty(features)
        features = linear;
        seed_features = seed_linear;
        fsd = std(features, 0, 1);
    end
    fsd(fsd < eps) = 1;
    features = features ./ fsd;
    seed_features = seed_features ./ fsd;
    info = struct('rank_est', rank_est, 'linear_dims', size(linear, 2), ...
        'feature_dim', size(features, 2), 'variance_explained_pct', var_pct);
end

function features = projective_features(Y)
    D = size(Y, 2);
    features = zeros(size(Y, 1), D * (D + 1) / 2);
    col = 0;
    for a = 1:D
        for b = a:D
            col = col + 1;
            if a == b, scale = 1; else, scale = sqrt(2); end
            features(:, col) = scale * Y(:, a) .* Y(:, b);
        end
    end
end

function [best, starts] = best_of_starts(features, K, base_seed, n_starts)
% Fit spm_mix from n_starts seeded initialisations and return the fit with the highest free energy (the first
% start wins a tie). Start s uses seed base_seed + K + 1000*(s-1), so start 1 is the single-start fit and
% every start is reproducible. A start that fails or returns a non-finite free energy is recorded as NaN and
% does not compete. starts holds the free energy of every start (1 x n_starts).
    n_starts = max(1, n_starts);
    starts = nan(1, n_starts);
    best = [];
    best_fm = -inf;
    for s = 1:n_starts
        rng(base_seed + K + 1000 * (s - 1), 'twister');
        try
            evalc('mix = spm_mix(features, K, 0);');
        catch
            continue;
        end
        if ~valid_mix(mix, K)
            continue;
        end
        starts(s) = double(real(mix.fm));
        if starts(s) > best_fm
            best = mix;
            best_fm = starts(s);
        end
    end
    if isempty(best)
        error('All %d start(s) failed for K=%d.', n_starts, K);
    end
end

function tf = valid_mix(mix, K)
    tf = isstruct(mix) && isfield(mix, 'state') && numel(mix.state) >= K && ...
        isfield(mix, 'fm') && isfinite(double(real(mix.fm)));
end

function labels = assign_mix(X, mix)
    K = mix.m;
    logp = -inf(size(X, 1), K);
    for k = 1:K
        mu = double(real(mix.state(k).m(:)'));
        C = double(real(mix.state(k).C));
        if isfield(mix.state(k), 'prior') && ~isempty(mix.state(k).prior)
            prior = double(real(mix.state(k).prior));
        else
            prior = 1 / K;
        end
        logp(:, k) = log(prior + eps) + log_gaussian(X, mu, C);
    end
    [~, labels] = max(logp, [], 2);
end

function y = log_gaussian(X, mu, C)
    D = size(X, 2);
    mu = mu(1:min(numel(mu), D));
    if numel(mu) < D
        mu = [mu zeros(1, D - numel(mu))];
    end
    C = coerce_cov(C, D);
    C = (C + C') / 2;
    ridge = max(1e-6, mean(diag(C), 'omitnan') * 1e-6);
    [R, p] = chol(C + ridge * eye(D));
    tries = 0;
    while p ~= 0 && tries < 5
        ridge = ridge * 10;
        [R, p] = chol(C + ridge * eye(D));
        tries = tries + 1;
    end
    if p ~= 0
        C = diag(max(var(X, 0, 1), 1e-6));
        R = chol(C);
    end
    Z = bsxfun(@minus, X, mu) / R;
    q = sum(Z.^2, 2);
    logdet = 2 * sum(log(diag(R)));
    y = -0.5 * (D * log(2*pi) + logdet + q);
end

function C = coerce_cov(C, D)
    if isscalar(C)
        C = C * eye(D);
    elseif isvector(C)
        v = C(:);
        C = diag(v(1:min(D, numel(v))));
    end
    if size(C, 1) ~= D || size(C, 2) ~= D
        out = eye(D);
        n = min([D, size(C, 1), size(C, 2)]);
        out(1:n, 1:n) = C(1:n, 1:n);
        C = out;
    end
end

function centers = recover_centers(X, labels, K, fallback)
    X = normalize_maps(X);
    fallback = normalize_maps(fallback);
    centers = zeros(K, size(X, 2));
    for k = 1:K
        idx = labels == k;
        if any(idx)
            Xk = X(idx, :);
            if size(Xk, 1) == 1
                c = Xk(1, :);
            else
                [~, ~, V] = svd(Xk, 'econ');
                c = V(:, 1)';
                if dot(c, mean(Xk, 1)) < 0
                    c = -c;
                end
            end
        elseif size(fallback, 1) >= k
            c = fallback(k, :);
        else
            c = X(mod(k - 1, size(X, 1)) + 1, :);
        end
        centers(k, :) = c;
    end
    centers = normalize_maps(centers);
end

function centers = refine_centers(X, centers, fallback, max_iter)
    duplicate_threshold = 0.90;
    centers = normalize_maps(centers);
    labels = assign_by_abs_correlation(X, centers);
    [labels, centers] = repair_duplicate_centers(X, labels, centers, fallback, duplicate_threshold);
    for it = 1:max_iter
        old = labels;
        old_centers = centers;
        centers = recover_centers(X, labels, size(centers, 1), fallback);
        labels = assign_by_abs_correlation(X, centers);
        [labels, centers] = repair_duplicate_centers(X, labels, centers, fallback, duplicate_threshold);
        if isequal(old, labels) && max(abs(centers(:) - old_centers(:))) < 1e-8
            break;
        end
    end
end

function [labels, centers] = repair_duplicate_centers(X, labels, centers, fallback, threshold)
    K = size(centers, 1);
    for attempt = 1:K
        pairs = duplicate_center_pairs(centers, threshold);
        if isempty(pairs)
            return;
        end
        pair = pairs(1, 1:2);
        counts = accumarray(labels(:), 1, [K, 1], @sum, 0);
        if counts(pair(1)) <= counts(pair(2))
            replace_idx = pair(1);
        else
            replace_idx = pair(2);
        end
        keep_idx = setdiff(1:K, replace_idx);
        sims = abs(normalize_maps(X) * normalize_maps(centers(keep_idx, :))');
        novelty = 1 - max(sims, [], 2);
        [~, seed_idx] = max(novelty);
        centers(replace_idx, :) = normalize_maps(X(seed_idx, :));
        labels = assign_by_abs_correlation(X, centers);
        centers = recover_centers(X, labels, K, fallback);
    end
end

function pairs = duplicate_center_pairs(centers, threshold)
    centers = normalize_maps(centers);
    signed_corr = centers * centers';
    pairs = [];
    for i = 1:(size(centers, 1) - 1)
        for j = (i + 1):size(centers, 1)
            if abs(signed_corr(i, j)) >= threshold
                pairs(end+1, :) = [i, j, signed_corr(i, j), abs(signed_corr(i, j))]; %#ok<AGROW>
            end
        end
    end
end

function metrics = topo_metrics(X, labels, centers, gfp)
    X = normalize_maps(X);
    centers = normalize_maps(centers);
    sim = abs(X * centers');
    [maxsim, ~] = max(sim, [], 2);
    w = double(gfp(:)).^2;
    if numel(w) ~= size(X, 1) || sum(w) <= eps
        w = ones(size(X, 1), 1);
    end
    metrics = struct();
    metrics.wss = sum(w .* (1 - maxsim.^2));
    metrics.gev = sum(w .* maxsim.^2) / sum(w);
    metrics.silhouette = silhouette_centroid(X, labels, centers);
end

function sil = silhouette_centroid(X, labels, centers)
    K = size(centers, 1);
    if K < 2 || size(X, 1) <= K
        sil = NaN;
        return;
    end
    dist = 1 - abs(X * centers');
    own = labels(:);
    a = dist(sub2ind(size(dist), (1:size(dist, 1))', own));
    dist(sub2ind(size(dist), (1:size(dist, 1))', own)) = Inf;
    b = min(dist, [], 2);
    vals = (b - a) ./ max(a, b);
    vals(~isfinite(vals)) = NaN;
    sil = mean(vals, 'omitnan');
end

function w = cluster_weights(labels, K)
    w = zeros(1, K);
    for k = 1:K
        w(k) = mean(labels == k);
    end
    w = w ./ max(sum(w), eps);
end

function s = empty_mix_summary()
    s = struct('K_candidate', NaN, 'feature_dim', NaN, 'free_energy', NaN, ...
        'priors', [], 'means', [], 'covariance_traces', [], 'covariance_logdets', [], 'feature_info', struct());
end

function s = summarise_mix(mix, K, feature_dim, feature_info)
    s = empty_mix_summary();
    s.K_candidate = K;
    s.feature_dim = feature_dim;
    s.free_energy = double(real(mix.fm));
    s.priors = nan(K, 1);
    s.means = nan(K, feature_dim);
    s.covariance_traces = nan(K, 1);
    s.covariance_logdets = nan(K, 1);
    s.feature_info = feature_info;
    for k = 1:K
        if isfield(mix.state(k), 'prior')
            s.priors(k) = double(real(mix.state(k).prior));
        end
        if isfield(mix.state(k), 'm')
            m = double(real(mix.state(k).m(:)'));
            s.means(k, 1:min(feature_dim, numel(m))) = m(1:min(feature_dim, numel(m)));
        end
        if isfield(mix.state(k), 'C')
            C = coerce_cov(double(real(mix.state(k).C)), feature_dim);
            s.covariance_traces(k) = trace(C);
            s.covariance_logdets(k) = safe_logdet(C);
        end
    end
end

function val = safe_logdet(C)
    val = NaN;
    if isempty(C) || any(~isfinite(C(:)))
        return;
    end
    C = (C + C') / 2;
    try
        [R, p] = chol(C);
        if p == 0
            val = 2 * sum(log(diag(R)));
        end
    catch
    end
end

function Fit = attach_template_alignment_if_available(Fit, cfg, common_labels, common_chanlocs)
    if isempty(cfg.template_file) || ~isfile(cfg.template_file)
        if cfg.require_template_labels
            error('Template labelling is required, but the template file was not found: %s', cfg.template_file);
        end
        return;
    end
    try
        alignment = align_microstates_to_template(Fit.centers, cfg.template_file, ...
            'estimated_channel_labels', common_labels, ...
            'estimated_chanlocs', common_chanlocs, ...
            'template_K', cfg.template_reference_K, ...
            'label_override', cfg.template_label_override, ...
            'strong_threshold', 0.5);
        Fit.template_alignment = alignment;
        Fit.centers = alignment.aligned_maps;
    catch ME
        if cfg.require_template_labels
            error('Template alignment failed for %s: %s', Fit.name, ME.message);
        end
        Fit.template_alignment = struct('labels', {{}}, 'mean_correlation', NaN, ...
            'n_strong_matches', NaN, 'message', ME.message);
    end
    if cfg.require_template_labels && ...
            (~isfield(Fit, 'template_alignment') || ...
             ~isfield(Fit.template_alignment, 'labels') || ...
             numel(Fit.template_alignment.labels) ~= Fit.K_estimated)
        error('Template alignment for %s did not provide exactly K=%d labels.', Fit.name, Fit.K_estimated);
    end
end

function save_fit_bundle(out_dir, prefix, Fit, rows, common_labels, common_chanlocs, cfg)
    ensure_dir(out_dir);
    save(fullfile(out_dir, [prefix '_solution.mat']), 'Fit', 'rows', 'common_labels', 'common_chanlocs', 'cfg', '-v7.3');
    write_matrix_csv(fullfile(out_dir, [prefix '_centers.csv']), Fit.centers);
    writetable(Fit.model_comparison, fullfile(out_dir, [prefix '_model_comparison.csv']));
    if istable(rows) && ~isempty(rows)
        writetable(rows, fullfile(out_dir, [prefix '_gfp_peak_manifest.csv']));
    end
end

function write_matrix_csv(file_path, X)
    T = array2table(double(X));
    names = cell(1, size(X, 2));
    for i = 1:size(X, 2)
        names{i} = sprintf('ch_%03d', i);
    end
    T.Properties.VariableNames = names;
    writetable(T, file_path);
end

function row = make_summary_row(level, name, group, condition, participant, Fit, n_units, n_maps, fit_path)
    [mean_corr, strong] = alignment_score(Fit);
    prior_source = string(field_value_or_default(Fit, 'prior_source', ''));
    prior_mode = string(field_value_or_default(Fit, 'prior_mode', ''));
    spm_prior_weight = double(field_value_or_default(Fit, 'spm_prior_weight', NaN));
    prior_state_index = string(strjoin(string(field_value_or_default(Fit, 'prior_state_index', [])), ';'));
    prior_beta_0 = string(strjoin(compose('%.12g', double(field_value_or_default(Fit, 'prior_beta_0', []))), ';'));
    parent_map_sha256 = string(field_value_or_default(Fit, 'parent_map_sha256', ''));
    spm_version = string(field_value_or_default(Fit, 'spm_version', ''));
    real_observation_count = double(field_value_or_default(Fit, 'real_observation_count', n_maps));
    row = table(string(level), string(name), string(group), string(condition), string(participant), ...
        double(Fit.K_estimated), double(n_units), double(n_maps), real_observation_count, ...
        prior_source, prior_mode, spm_prior_weight, prior_state_index, prior_beta_0, parent_map_sha256, ...
        spm_version, double(mean_corr), double(strong), string(fit_path), ...
        'VariableNames', {'level', 'name', 'group', 'condition', 'participant', 'K_estimated', 'n_units', 'n_maps', ...
        'real_observation_count','prior_source','prior_mode','spm_prior_weight','prior_state_index','prior_beta_0', ...
        'parent_map_sha256','spm_version','template_mean_corr','template_strong_matches','fit_path'});
end

function [mean_corr, strong] = alignment_score(Fit)
    mean_corr = NaN;
    strong = NaN;
    if isfield(Fit, 'template_alignment')
        A = Fit.template_alignment;
        if isfield(A, 'mean_correlation'), mean_corr = A.mean_correlation; end
        if isfield(A, 'n_strong_matches'), strong = A.n_strong_matches; end
    end
end

function node = build_node(level, participant, group, condition, name, Fit, n_maps, fit_path)
    node = struct();
    node.level = string(level);
    node.participant = string(participant);
    node.group = string(group);
    node.condition = string(condition);
    node.name = string(name);
    node.centers = Fit.centers;
    node.K_estimated = Fit.K_estimated;
    node.n_maps = n_maps;
    node.fit = Fit;
    if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels')
        node.canonical_state_labels = Fit.template_alignment.labels;
        node.template_alignment = Fit.template_alignment;
    else
        node.canonical_state_labels = {};
        node.template_alignment = struct();
    end
    node.fit_path = string(fit_path);
end

function [labels, changed, source] = resolve_global_manual_labels(Fit, cfg, out_dir)
% Labels in fitted-state order from the call ('global_manual_labels') or from a saved label file whose
% reference maps match the current fit. Empty when neither exists. A label set that came from the call is
% saved to out_dir/global_manual_labels.{mat,csv}.
    labels = {};
    source = '';
    old_labels = {};
    if isfield(Fit, 'template_alignment') && isfield(Fit.template_alignment, 'labels')
        old_labels = cellstr(string(Fit.template_alignment.labels(:)));
    end

    label_file = fullfile(out_dir, 'global_manual_labels.mat');
    explicit = cellstr(string(cfg.global_manual_labels(:)));
    if ~isempty(explicit)
        labels = validate_global_labels(explicit, size(Fit.centers, 1));
        source = 'command_line';
    elseif cfg.resume && isfile(label_file)
        cached = load(label_file, 'labels', 'reference_centers');
        if isfield(cached, 'labels') && isfield(cached, 'reference_centers') && ...
                same_state_maps(Fit.centers, cached.reference_centers)
            labels = validate_global_labels(cached.labels, size(Fit.centers, 1));
            source = 'saved_manual_labels';
            if cfg.verbose
                fprintf('Reusing saved global labels: %s\n', strjoin(labels, ', '));
            end
        elseif cfg.verbose
            fprintf('Ignoring the saved global-label file because the global maps changed.\n');
        end
    end

    changed = ~isempty(labels) && ~same_labels(labels, old_labels);
    if isempty(labels) || strcmp(source, 'saved_manual_labels')
        return;
    end
    reference_centers = Fit.centers;
    selected_utc = string(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
    label_source = source;
    save(label_file, 'labels', 'reference_centers', 'selected_utc', 'label_source', '-v7');
    writetable(table((1:numel(labels))', string(labels(:)), repmat(selected_utc, numel(labels), 1), ...
        repmat(string(source), numel(labels), 1), ...
        'VariableNames', {'state_index','label','selected_utc','source'}), ...
        fullfile(out_dir, 'global_manual_labels.csv'));
end

function C = compact_hresults_for_save(H)
    C = H;
    embedded = {'state_covariance_metrics', 'peak_records'};
    for i = 1:numel(embedded)
        if isfield(C, embedded{i}), C = rmfield(C, embedded{i}); end
    end
    node_fields = {'global', 'groups', 'conditions', 'participants', 'participant_conditions'};
    for i = 1:numel(node_fields)
        field = node_fields{i};
        if isfield(C, field) && isstruct(C.(field)) && isfield(C.(field), 'fit')
            C.(field) = rmfield(C.(field), 'fit');
        end
    end
    C.storage_mode = 'compact_hierarchy';
end

function [labels, valid] = parse_global_labels(raw, K)
% K unique single letters A-G; no class is required to be present.
    labels = regexp(upper(strtrim(char(raw))), '[,;/\s]+', 'split');
    labels = labels(~cellfun('isempty', labels));
    labels = labels(:);
    valid = numel(labels) == K && numel(unique(labels)) == K && ...
        all(~cellfun('isempty', regexp(labels, '^[A-G]$', 'once')));
    if ~valid
        labels = {};
    end
end

function labels = validate_global_labels(labels, K)
    [labels, valid] = parse_global_labels(strjoin(cellstr(string(labels(:))), ','), K);
    if ~valid
        error('Global labels must be %d unique letters from A-G.', K);
    end
end

function tf = same_labels(a, b)
    tf = numel(a) == numel(b) && all(upper(strtrim(string(a(:)))) == upper(strtrim(string(b(:)))));
end

function tf = same_state_maps(a, b)
    if ~isequal(size(a), size(b))
        tf = false;
        return;
    end
    a = normalize_maps(double(a));
    b = normalize_maps(double(b));
    tf = all(abs(sum(a .* b, 2)) > 1 - 1e-8);
end

function Fit = apply_manual_global_labels_by_state_order(Fit, labels, cfg, common_labels, common_chanlocs)
    labels = cellstr(string(labels(:)));
    if isempty(labels)
        return;
    end
    K = size(Fit.centers, 1);
    if numel(labels) ~= K || numel(unique(lower(string(labels)))) ~= K
        error('global_manual_labels must contain exactly %d unique labels.', K);
    end
    alignment = align_microstates_to_template(Fit.centers, cfg.template_file, ...
        'estimated_channel_labels', common_labels, ...
        'estimated_chanlocs', common_chanlocs, ...
        'template_K', cfg.template_reference_K, ...
        'label_override', labels, ...
        'strong_threshold', 0.5);
    alignment.manual_label_override = true;
    alignment.manual_label_note = 'Labels assigned by global fitted-state order.';
    Fit.template_alignment = alignment;
    Fit.centers = alignment.aligned_maps;
end

function Fit = align_fit_to_reference(Fit, reference_fit)
    reference = normalize_maps(double(reference_fit.centers));
    estimated = normalize_maps(double(Fit.centers));
    K = size(reference, 1);
    if size(estimated, 1) ~= K
        error('Reference alignment requires equal K; reference=%d fitted=%d.', K, size(estimated, 1));
    end
    recovery = microstate_partial_alignment(reference, estimated, 'threshold', 0, 'polarity', true);
    assignment = recovery.match_assignment;
    if size(assignment, 1) ~= K || numel(unique(assignment(:, 1))) ~= K || numel(unique(assignment(:, 2))) ~= K
        error('Reference alignment failed to obtain a one-to-one K=%d assignment.', K);
    end
    signed = estimated * reference';
    centers = nan(size(estimated));
    correlations = nan(K, 1);
    polarity = ones(K, 1);
    old_to_new = nan(K, 1);
    for r = 1:size(assignment, 1)
        old = assignment(r, 1);
        new = assignment(r, 2);
        s = sign(signed(old, new));
        if s == 0 || ~isfinite(s), s = 1; end
        centers(new, :) = s * estimated(old, :);
        correlations(new) = abs(signed(old, new));
        polarity(new) = s;
        old_to_new(old) = new;
    end
    Fit.centers = centers;
    if isfield(Fit, 'labels'), Fit.labels = remap_state_indices_local(Fit.labels, old_to_new); end
    if isfield(Fit, 'backfit_peak_labels'), Fit.backfit_peak_labels = remap_state_indices_local(Fit.backfit_peak_labels, old_to_new); end
    if isfield(Fit, 'cluster_weights'), Fit.cluster_weights = reorder_state_rows(Fit.cluster_weights, old_to_new); end
    if isfield(Fit, 'centers_by_K') && numel(Fit.centers_by_K) >= Fit.best_index
        Fit.centers_by_K{Fit.best_index} = centers;
    end
    if isfield(Fit, 'selected_spm_mix_model')
        Fit.selected_spm_mix_model = reorder_mix_summary(Fit.selected_spm_mix_model, old_to_new);
    end
    if isfield(Fit, 'spm_mix_model_summaries') && numel(Fit.spm_mix_model_summaries) >= Fit.best_index
        Fit.spm_mix_model_summaries(Fit.best_index) = Fit.selected_spm_mix_model;
    end
    [labels, ~, reference_corr] = state_labels_for_metrics(reference_fit);
    Fit.template_alignment = struct('labels', {labels}, 'correlations', correlations, ...
        'reference_correlations', reference_corr, 'polarity', polarity, ...
        'aligned_maps', centers, 'old_order_to_new_order', old_to_new, ...
        'reference_state_indices', (1:K)', 'mean_correlation', mean(correlations), ...
        'median_correlation', median(correlations), 'min_correlation', min(correlations), ...
        'n_strong_matches', sum(correlations >= 0.5), 'strong_threshold', 0.5, ...
        'manual_label_override', true, 'alignment_source', 'global_manual_reference');
end

function labels = remap_state_indices_local(labels, old_to_new)
    original = labels;
    for old = 1:numel(old_to_new)
        labels(original == old) = old_to_new(old);
    end
end

function values = reorder_state_rows(values, old_to_new)
    original = values;
    values = nan(size(original));
    if isvector(original)
        for old = 1:numel(old_to_new), values(old_to_new(old)) = original(old); end
    else
        for old = 1:numel(old_to_new), values(old_to_new(old), :) = original(old, :); end
    end
end

function summary = reorder_mix_summary(summary, old_to_new)
    fields = {'priors','means','covariance_traces','covariance_logdets'};
    for f = 1:numel(fields)
        name = fields{f};
        if isfield(summary, name) && ~isempty(summary.(name))
            summary.(name) = reorder_state_rows(summary.(name), old_to_new);
        end
    end
end

function Fit = load_cached_fit(out_dir, prefix, cfg)
    Fit = [];
    solution_file = fullfile(out_dir, [prefix '_solution.mat']);
    if ~cfg.resume || ~isfile(solution_file)
        return;
    end
    cached = load(solution_file, 'Fit');
    if isfield(cached, 'Fit')
        Fit = cached.Fit;
        if cfg.verbose
            fprintf('Resuming completed fit: %s\n', solution_file);
        end
    end
end

function nodes = empty_nodes(n_channels)
    nodes = repmat(build_node("", "", "", "", "", struct('centers', zeros(1, n_channels), 'K_estimated', 0), 0, ""), 0, 1);
end

function assert_fixed_hierarchy_k(K, varargin)
    for a = 1:numel(varargin)
        nodes = varargin{a};
        for i = 1:numel(nodes)
            if double(nodes(i).K_estimated) ~= double(K) || size(nodes(i).centers, 1) ~= double(K)
                error('Hierarchy node %s has K=%d; all nodes must retain global K=%d.', ...
                    char(nodes(i).name), double(nodes(i).K_estimated), double(K));
            end
        end
    end
end

function print_alignment_line(label, Fit)
    if ~isfield(Fit, 'template_alignment')
        return;
    end
    A = Fit.template_alignment;
    if isfield(A, 'mean_correlation') && isfinite(A.mean_correlation)
        fprintf('%s template fit: mean r=%.3f, strong=%d/%d\n', label, A.mean_correlation, A.n_strong_matches, Fit.K_estimated);
    end
end

function labels = assign_by_abs_correlation(maps, centers)
    X = normalize_maps(double(maps));
    C = normalize_maps(double(centers));
    [~, labels] = max(abs(X * C'), [], 2);
end

function Xn = normalize_maps(X)
    X = double(X);
    if isempty(X)
        Xn = X;
        return;
    end
    X = X - mean(X, 2, 'omitnan');
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function Xn = normalize_rows(X)
    denom = sqrt(sum(X.^2, 2));
    denom(~isfinite(denom) | denom <= eps) = 1;
    Xn = X ./ denom;
end

function idx = deterministic_subsample(N, maxN)
    if isempty(maxN) || ~isfinite(maxN) || N <= maxN
        idx = (1:N)';
        return;
    end
    idx = unique(round(linspace(1, N, round(maxN))))';
    if numel(idx) > maxN
        idx = idx(1:maxN);
    end
end

function idx = capped_mask_indices(mask, maxN)
    idx = find(mask);
    keep = deterministic_subsample(numel(idx), maxN);
    idx = idx(keep);
end

function ensure_dir(pth)
    if ~exist(pth, 'dir')
        mkdir(pth);
    end
end

function s = onoff(tf)
    if tf
        s = 'ON';
    else
        s = 'OFF';
    end
end

function key = clean_key(value)
    key = char(string(value));
    key = regexprep(key, '[^A-Za-z0-9_-]', '_');
    if isempty(key)
        key = 'empty';
    end
end

function manifest = read_manifest(input_path)
    input_path = char(input_path);
    if ~isfile(input_path)
        error('Manifest not found: %s', input_path);
    end
    opts = detectImportOptions(input_path, 'FileType', 'text', 'TextType', 'string', ...
        'Delimiter', manifest_delimiter(input_path));
    opts.VariableNamingRule = 'preserve';
    T = readtable(input_path, opts);
    names = normalise_names(T.Properties.VariableNames);
    file_col = find_first(names, ["filepath", "file_path", "file", "path", "filename"]);
    if isempty(file_col)
        error('Manifest must contain a file_path column.');
    end
    file_path = string(T{:, file_col});
    base_dir = fileparts(input_path);
    for i = 1:numel(file_path)
        if ~is_absolute_path(file_path(i))
            file_path(i) = string(fullfile(base_dir, file_path(i)));
        end
        if ~isfile(file_path(i))
            local_path = fullfile(base_dir, path_basename(file_path(i)));
            if isfile(local_path)
                file_path(i) = string(local_path);
            end
        end
    end
    p_col = find_first(names, ["participant", "subject", "sub", "id", "subjectid", "participantid"]);
    c_col = find_first(names, ["condition", "state", "task", "eyes", "session"]);
    g_col = find_first(names, ["group", "diagnosis", "cohort", "class"]);
    participant = column_or_infer(T, p_col, file_path, "participant");
    condition = column_or_default(T, c_col, "condition", numel(file_path));
    group = column_or_default(T, g_col, "all", numel(file_path));

    [file_path, ord] = sort(file_path);
    participant = participant(ord);
    condition = condition(ord);
    group = group(ord);
    manifest = table(participant(:), condition(:), group(:), file_path(:), ...
        'VariableNames', {'participant', 'condition', 'group', 'file_path'});
end
