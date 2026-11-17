function result = global_state_labels(hierarchy_dir, varargin)
%GLOBAL_STATE_LABELS Name the states of a fitted global solution and record the K selection.
%
%   result = global_state_labels(hierarchy_dir)
%
%   Reads <hierarchy_dir>/global/global_solution.mat (Fit, common_labels, common_chanlocs, cfg), matches the
%   K fitted states to the MetaMaps classes with match_template_labels (see that file for the rule; every
%   class A-G is available, none is forced) and writes into <hierarchy_dir>/global/
%       global_manual_labels.csv   state_index,label,selected_utc,source
%       global_manual_labels.mat   labels, reference_centers, selected_utc, label_source
%                                  (the format fit_microstate_hierarchical_dataset reads back)
%       labelling_audit.json       rule, template and script SHA-256, correlation matrices, assignment,
%                                  labelling_quality, states below threshold, channel-matching mode,
%                                  MATLAB/SPM versions
%       state_labels.csv           fitted_state,label,abs_r,below_threshold,runner_up_label
%       k_selection.csv            K,free_energy,elbow_score,selected,n_starts,free_energy_median
%                                  (one row per candidate K; free_energy is the best of the starts)
%       free_energy_starts.csv     K,start,seed,free_energy,best   (every random start of every candidate K)
%       k_selection.json           rule text, criterion, selected_K, labels (class label per fitted state),
%                                  candidate curve
%       global_centres.csv         channel,state_label,fitted_state,value: the selected global maps in long
%                                  format (standard 10-20/10-10 electrode labels only; each map is
%                                  zero-mean, unit-norm, with polarity fixed to the matched template)
%
%   A low assigned correlation (|r| < 0.5) never stops the run: it sets below_threshold for that state and
%   labelling_quality = 'below_threshold' in the audit, and the state keeps its assigned label. Only
%   technical failures that make the template comparison meaningless (channel mode other than
%   labels/coordinates_idw_to_template, too few channels, non-finite maps, K outside 2..7, tied
%   assignment) write global/LABELLING_REJECTED.json, write no label files and error with
%   FSMS:LabellingRejected.
%
%   Idempotent: a repeat call with an unchanged global solution leaves existing label files untouched.
%   Optional 'template_file' overrides the MetaMaps path (the template SHA-256 is recorded).

    p = inputParser;
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'provenance', struct(), @isstruct);
    parse(p, varargin{:});
    env = fsms_paths();
    const = match_template_labels_constants();
    hierarchy_dir = char(string(hierarchy_dir));
    global_dir = fullfile(hierarchy_dir, 'global');
    solution_file = fullfile(global_dir, 'global_solution.mat');
    if ~isfile(solution_file)
        error('FSMS:GlobalSolutionMissing', 'Global solution not found: %s', solution_file);
    end
    template_file = char(string(p.Results.template_file));
    if isempty(template_file)
        template_file = fullfile(env.vg_root, 'MetaMaps_2023_06.set');
    end
    rejection_file = fullfile(global_dir, 'LABELLING_REJECTED.json');
    audit_file = fullfile(global_dir, 'labelling_audit.json');
    csv_file = fullfile(global_dir, 'global_manual_labels.csv');
    mat_file = fullfile(global_dir, 'global_manual_labels.mat');
    state_labels_file = fullfile(global_dir, 'state_labels.csv');
    k_csv_file = fullfile(global_dir, 'k_selection.csv');
    k_json_file = fullfile(global_dir, 'k_selection.json');
    centres_file = fullfile(global_dir, 'global_centres.csv');

    S = load(solution_file, 'Fit', 'common_labels', 'common_chanlocs', 'cfg');
    centers = double(S.Fit.centers);
    K = size(centers, 1);

    % Technical pre-checks that make the template comparison impossible.
    pre = {};
    if K < 2 || K > const.max_states
        pre{end + 1} = sprintf('K_OUTSIDE_TEMPLATE_RANGE:states=%d,allowed=2..%d', K, const.max_states);
    end
    if any(~isfinite(centers(:)))
        pre{end + 1} = 'NONFINITE_MAPS';
    end
    if ~isempty(pre)
        evidence = struct('rule_id', const.rule_id, 'accepted', false, 'rejection_reasons', {pre(:)}, ...
            'n_fitted_states', K, 'global_solution_sha256', fsms_sha256_file(solution_file), ...
            'nominal_label_semantics', const.nominal_label_semantics, ...
            'action_required', 'Technical failure: the template comparison is meaningless. The stage has stopped.', ...
            'created_utc', now_utc());
        fsms_write_json(rejection_file, evidence);
        error('FSMS:LabellingRejected', 'Template labelling rejected (%s). Evidence written to %s', ...
            strjoin(pre, '; '), rejection_file);
    end

    rule = match_template_labels(centers, S.common_labels, S.common_chanlocs, 'template_file', template_file);
    audit = build_audit(rule, env, solution_file, template_file, centers, p.Results.provenance);

    if ~rule.accepted
        audit.accepted = false;
        audit.rejection_reasons = rule.reasons;
        audit.action_required = ['Technical failure: the template comparison is meaningless. The stage has ' ...
            'stopped. Investigate the evidence below; do not substitute hand-chosen labels.'];
        fsms_write_json(rejection_file, audit);
        error('FSMS:LabellingRejected', 'Template labelling rejected (%s). Evidence written to %s', ...
            strjoin(rule.reasons, '; '), rejection_file);
    end
    if isfile(rejection_file)
        error('FSMS:LabellingRejectionMarkerPresent', ...
            ['A LABELLING_REJECTED.json marker exists in %s but the labelling is now accepted. ' ...
             'The directory is inconsistent; use a fresh output directory.'], global_dir);
    end

    write_k_selection(S.Fit, S.cfg, rule.labels, k_csv_file, k_json_file);

    labels = rule.labels(:);
    existing = existing_labels(mat_file, csv_file);
    if ~isempty(existing)
        if ~strcmp(existing.source, const.rule_id)
            error('FSMS:ForeignLabelFile', ['global_manual_labels was written by source "%s", not by %s. ' ...
                'Refusing to mix labelling sources in %s.'], existing.source, const.rule_id, global_dir);
        end
        if ~isequal(existing.labels(:), labels(:))
            error('FSMS:LabelsChanged', ['Existing labels (%s) differ from the labels now derived (%s). ' ...
                'The global solution changed; use a fresh output directory.'], ...
                strjoin(existing.labels, ','), strjoin(labels, ','));
        end
        result = rule;
        result.audit_file = audit_file;
        result.labels_csv = csv_file;
        result.reused_existing = true;
        if ~isfile(audit_file), fsms_write_json(audit_file, audit); end
        if ~isfile(state_labels_file), write_state_labels(rule, state_labels_file); end
        if ~isfile(centres_file), write_global_centres(centers, rule, S.common_labels, centres_file); end
        return;
    end

    selected_utc = now_utc();
    audit.selected_utc = selected_utc;
    reference_centers = centers; %#ok<NASGU>
    payload = struct('labels', {labels}, 'reference_centers', reference_centers, ...
        'selected_utc', string(selected_utc), 'label_source', const.rule_id);
    fsms_atomic_save(mat_file, payload, '-v7');
    T = table((1:K)', string(labels), repmat(string(selected_utc), K, 1), repmat(string(const.rule_id), K, 1), ...
        'VariableNames', {'state_index', 'label', 'selected_utc', 'source'});
    fsms_atomic_writetable(T, csv_file);
    write_state_labels(rule, state_labels_file);
    write_global_centres(centers, rule, S.common_labels, centres_file);
    fsms_write_json(audit_file, audit);
    result = rule;
    result.audit_file = audit_file;
    result.labels_csv = csv_file;
    result.reused_existing = false;
    fprintf('State labels (K=%d): %s | labelling_quality=%s (assigned |r|: %s)\n', K, ...
        strjoin(compose('%d=%s', (1:K)', string(labels))', ', '), rule.labelling_quality, ...
        strjoin(compose('%.3f', rule.assigned_abs_r(:))', ' '));
    if strcmp(rule.labelling_quality, 'below_threshold')
        fprintf(['NOTE: %d state(s) have assigned |r| < %.2f; they keep their assigned labels and are ' ...
            'flagged below_threshold.\n'], numel(rule.states_below_threshold), const.strong_threshold);
    end
end

function write_state_labels(rule, file)
    T = table(rule.state_index(:), string(rule.labels(:)), rule.assigned_abs_r(:), rule.below_threshold(:), ...
        string(rule.runner_up_label(:)), ...
        'VariableNames', {'fitted_state', 'label', 'abs_r', 'below_threshold', 'runner_up_label'});
    fsms_atomic_writetable(T, file);
end

function write_global_centres(centers, rule, channel_labels, file)
% Long-format global maps for the release bundle. Only standard 10-20/10-10 electrode labels are written.
    standard = ["Fp1","Fpz","Fp2","AF7","AF3","AFz","AF4","AF8","F9","F7","F5","F3","F1","Fz","F2","F4","F6","F8","F10", ...
        "FT9","FT7","FC5","FC3","FC1","FCz","FC2","FC4","FC6","FT8","FT10","T9","T7","T3","C5","C3","C1","Cz","C2","C4","C6", ...
        "T8","T4","T10","TP9","TP7","CP5","CP3","CP1","CPz","CP2","CP4","CP6","TP8","TP10","P9","P7","T5","P5","P3","P1", ...
        "Pz","P2","P4","P6","P8","T6","P10","PO7","PO3","POz","PO4","PO8","O1","Oz","O2","Iz","A1","A2"];
    channel_labels = string(channel_labels(:));
    keep = ismember(lower(channel_labels), lower(standard));
    if ~any(keep)
        error('FSMS:NoStandardChannels', 'None of the common channels is a standard 10-20/10-10 electrode label.');
    end
    maps = double(centers) .* double(rule.polarity(:));
    maps = maps - mean(maps, 2);
    maps = maps ./ max(sqrt(sum(maps .^ 2, 2)), eps);
    K = size(maps, 1);
    idx = find(keep);
    n = numel(idx);
    channel = repmat(channel_labels(idx), K, 1);
    state_label = repelem(string(rule.labels(:)), n);
    fitted_state = repelem((1:K)', n);
    value = reshape(maps(:, idx)', [], 1);
    T = table(channel, state_label, fitted_state, value);
    fsms_atomic_writetable(T, file);
end

function write_start_table(starts, K, cfg, file)
% Free energy of every random start for every candidate K; best marks the start whose fit was kept.
    seed0 = 1;
    if isfield(cfg, 'random_seed'), seed0 = double(cfg.random_seed); end
    [nK, n_starts] = size(starts);
    Kcol = repelem(K(:), n_starts);
    start = repmat((1:n_starts)', nK, 1);
    seed = seed0 + Kcol + 1000 * (start - 1);
    free_energy = reshape(starts', [], 1);
    [~, best_col] = max(starts, [], 2);
    best = double(start == repelem(best_col, n_starts));
    T = table(Kcol, start, seed, free_energy, best, ...
        'VariableNames', {'K', 'start', 'seed', 'free_energy', 'best'});
    fsms_atomic_writetable(T, file);
end

function write_k_selection(Fit, cfg, labels, csv_file, json_file)
% One row per fitted candidate K, and the selection rule with the curve it was applied to.
    mc = Fit.model_comparison;
    is_elbow = strcmp(char(string(Fit.criterion)), 'free_energy_elbow');
    elbow = nan(height(mc), 1);
    if is_elbow
        elbow = double(mc.selection_score);
    end
    n_starts = 1;
    starts = double(mc.free_energy);
    if isfield(Fit, 'free_energy_starts') && ~isempty(Fit.free_energy_starts)
        starts = double(Fit.free_energy_starts);
        n_starts = size(starts, 2);
    end
    median_fe = median(starts, 2, 'omitnan');
    T = table(double(mc.K), double(mc.free_energy), elbow, double(mc.selected), ...
        repmat(n_starts, height(mc), 1), median_fe, ...
        'VariableNames', {'K', 'free_energy', 'elbow_score', 'selected', 'n_starts', 'free_energy_median'});
    fsms_atomic_writetable(T, csv_file);
    write_start_table(starts, double(mc.K), cfg, fullfile(fileparts(csv_file), 'free_energy_starts.csv'));
    requested = [];
    if isfield(cfg, 'K_candidates'), requested = double(cfg.K_candidates(:))'; end
    details = struct();
    if isfield(Fit, 'selection_details') && isstruct(Fit.selection_details)
        details = Fit.selection_details;
    end
    info = struct('criterion', char(string(Fit.criterion)), ...
        'rule', char(string(Fit.k_selection_rule)), ...
        'selected_K', double(Fit.K_estimated), ...
        'labels', {cellstr(string(labels(:)))}, ...
        'K_requested', requested, ...
        'K_fitted', double(Fit.K_candidates(:))', ...
        'degenerate_curve', isfield(details, 'degenerate') && logical(details.degenerate), ...
        'n_starts', n_starts, ...
        'start_rule', ['each K is fitted from n_starts seeded k-means initialisations (seed = random_seed + K + ' ...
            '1000*(start-1)); the fit with the highest free energy is kept and enters the K-selection curve'], ...
        'n_maps', double(Fit.n_maps), ...
        'curve', struct('K', double(mc.K(:))', 'free_energy', double(mc.free_energy(:))', ...
            'elbow_score', elbow(:)', 'selected', double(mc.selected(:))'), ...
        'created_utc', now_utc());
    fsms_write_json(json_file, info);
end

function audit = build_audit(rule, env, solution_file, template_file, centers, provenance)
    here = fileparts(mfilename('fullpath'));
    vg = env.vg_root;
    scripts = {fullfile(here, 'global_state_labels.m'), ...
        fullfile(here, 'match_template_labels.m'), ...
        fullfile(here, 'match_template_labels_constants.m'), ...
        fullfile(vg, '_internal', 'matlab', 'align_microstates_to_template.m'), ...
        fullfile(vg, '_internal', 'matlab', 'load_metamaps_templates.m'), ...
        fullfile(vg, '_internal', 'matlab', 'microstate_utilities.m')};
    script_sha = struct();
    for i = 1:numel(scripts)
        [~, name] = fileparts(scripts{i});
        script_sha.(name) = fsms_sha256_file(scripts{i});
    end
    K = rule.n_states;
    audit = struct();
    audit.rule_id = rule.rule_id;
    audit.accepted = rule.accepted;
    audit.n_fitted_states = K;
    audit.description = sprintf(['Exact optimal one-to-one assignment of the %d global fitted states to the ' ...
        'MetaMaps classes A-G (all seven available, none forced) maximising the summed polarity-invariant ' ...
        '|spatial correlation| with MetaMaps_2023_06 K=7. accepted=true means the comparison was ' ...
        'technically sound; low similarity (assigned |r| < strong_threshold) is reported per state and in ' ...
        'labelling_quality, and does not change any label.'], K);
    audit.nominal_label_semantics = rule.nominal_label_semantics;
    audit.labelling_quality = rule.labelling_quality;
    audit.states_below_threshold = num2cell(rule.states_below_threshold(:))';   % always a JSON array
    audit.labels_by_fitted_state = rule.labels(:);
    audit.state_index = rule.state_index(:);
    audit.assigned_abs_r = rule.assigned_abs_r(:);
    audit.assigned_signed_r = rule.assigned_signed_r(:);
    audit.below_threshold = rule.below_threshold(:);
    audit.polarity = rule.polarity(:);
    audit.runner_up_label = rule.runner_up_label(:);
    audit.next_best_abs_r = rule.runner_up_abs_r(:);
    audit.classes_available = rule.classes(:);
    audit.classes_unused = rule.classes_unused(:);
    audit.correlation_matrix_abs = rule.corr_abs;
    audit.correlation_matrix_abs_columns = rule.classes(:);
    audit.correlation_matrix_abs_all_template_classes = rule.corr_abs_all_template_classes;
    audit.correlation_matrix_abs_all_template_classes_columns = rule.template_class_order(:);
    audit.assignment_class_index = rule.assignment_class_index(:);
    audit.total_abs_r = rule.total_abs_r;
    audit.runner_up_total_abs_r = rule.runner_up_total_abs_r;
    audit.margin_total_abs_r = rule.margin_total_abs_r;
    audit.n_tied_optimal_assignments = rule.n_tied_optimal_assignments;
    audit.strong_threshold = rule.strong_threshold;
    audit.tie_rule = ['lexicographically smallest assignment vector (entry k = index of the class ' ...
        'A..G given to fitted state k); total |r| tolerance 1e-9; more than one tied optimum is a technical failure'];
    audit.channel_match_mode = rule.channel_match_mode;
    audit.n_common_channels = rule.n_common_channels;
    audit.n_estimated_channels_used = rule.n_estimated_channels_used;
    audit.min_label_matches = rule.min_label_matches;
    audit.matched_channel_labels = rule.matched_channel_labels(:);
    audit.template_file = relative_to(template_file, env.repo_root);
    audit.template_sha256 = fsms_sha256_file(template_file);
    audit.template_K = rule.template_K;
    audit.global_solution_file = relative_to(solution_file, env.repo_root);
    audit.global_solution_sha256 = fsms_sha256_file(solution_file);
    audit.reference_centers_sha256 = fsms_sha256_array(centers);
    audit.script_sha256 = script_sha;
    audit.matlab_version = env.matlab_version;
    audit.matlab_release = env.matlab_release;
    audit.spm = env.spm;
    audit.created_utc = now_utc();
    if ~isempty(fieldnames(provenance))
        audit.provenance = provenance;
    end
end

function s = now_utc()
    s = char(datetime('now', 'TimeZone', 'UTC', 'Format', 'yyyy-MM-dd''T''HH:mm:ssXXX'));
end

function out = relative_to(path_value, root)
    path_value = char(string(path_value));
    root = char(string(root));
    if startsWith(lower(path_value), lower(root))
        out = strrep(path_value(numel(root) + 2:end), '\', '/');
    else
        out = strrep(path_value, '\', '/');
    end
end

function existing = existing_labels(mat_file, csv_file)
    existing = [];
    if isfile(mat_file)
        cached = load(mat_file);
        source = '';
        if isfield(cached, 'label_source'), source = char(string(cached.label_source)); end
        existing = struct('labels', {cellstr(string(cached.labels(:)))}, 'source', source);
    elseif isfile(csv_file)
        T = readtable(csv_file, 'TextType', 'string', 'Delimiter', ',');
        T = sortrows(T, 'state_index');
        existing = struct('labels', {cellstr(T.label)}, 'source', char(T.source(1)));
    end
end
