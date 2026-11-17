function outputs = run_vg_microstates(manifest_csv, output_dir, varargin)
%RUN_VG_MICROSTATES Headless VG-microstates hierarchy for a manifest of EEG files.
%
%   outputs = run_vg_microstates(manifest_csv, output_dir, ...)
%
% Fits the global VB-GMM from the pooled GFP-peak maps for every candidate K, selects K with the criterion
% (default: free-energy elbow, see select_spm_vb_k_by_criterion), names the global states, then fits the
% group, condition and participant levels at the selected K (parent-prior chain). There is no interactive
% step and no hard-coded path: manifest_csv and output_dir are required.
%
% LABELLING (name 'labelling_mode')
%   'template_match' (default)  every selected state is matched to the MetaMaps classes A-G by the exact
%                    optimal one-to-one assignment maximising the summed |r| (match_template_labels). No
%                    class is forced; weak matches (|r| < 0.5) are flagged, never rejected. Implemented in
%                    two passes: pass 1 fits the global solution without template alignment, the labels are
%                    derived from it (global_state_labels, which also writes k_selection.csv/json and
%                    state_labels.csv), and pass 2 runs the full hierarchy from the cached global fit and the
%                    saved labels. The global fit is identical to a single-pass run.
%   'none'           no labelling; states keep their fitted order (used for peak extraction/screening).
%
% Name-value options
%   K_candidates 2:7, criterion 'free_energy_elbow', spm_prior_weight 1, global_only false,
%   run_participant_condition_fits false, compact_results_mat true, redo false (true recomputes every cache
%   in place; not allowed with template_match), template_file '' (vendored MetaMaps),
%   require_template_labels true (pass 2 only), extra_hierarchy_args {} (extra name-value pairs for
%   fit_microstate_hierarchical_dataset; diagnostics and tests only).
%
% Candidates K that are not smaller than the number of maps are dropped with a warning; the K actually
% selected is returned in outputs.selected_K and written to global/k_selection.{csv,json}.

    if nargin < 2 || isempty(manifest_csv) || isempty(output_dir)
        error('FSMS:RunVgArguments', 'run_vg_microstates(manifest_csv, output_dir, ...) needs both arguments.');
    end
    p = inputParser;
    addRequired(p, 'manifest_csv', @(x) ischar(x) || isstring(x));
    addRequired(p, 'output_dir', @(x) ischar(x) || isstring(x));
    addParameter(p, 'K_candidates', 2:7, @(x) isnumeric(x) && ~isempty(x) && all(x >= 2));
    addParameter(p, 'criterion', 'free_energy_elbow', @(x) ischar(x) || isstring(x));
    addParameter(p, 'labelling_mode', 'template_match', @(x) ischar(x) || isstring(x));
    addParameter(p, 'spm_prior_weight', 1, @(x) isnumeric(x) && isscalar(x) && x > 0);
    addParameter(p, 'global_only', false, @islogical);
    addParameter(p, 'run_participant_condition_fits', false, @islogical);
    addParameter(p, 'compact_results_mat', true, @islogical);
    addParameter(p, 'redo', false, @islogical);
    addParameter(p, 'require_template_labels', true, @islogical);
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    addParameter(p, 'extra_hierarchy_args', {}, @iscell);
    parse(p, manifest_csv, output_dir, varargin{:});
    cfg = p.Results;
    env = fsms_paths();
    manifest_csv = char(string(cfg.manifest_csv));
    output_dir = char(string(cfg.output_dir));
    if ~isfile(manifest_csv)
        error('FSMS:ManifestMissing', 'Manifest not found: %s', manifest_csv);
    end
    if ~isfolder(output_dir), mkdir(output_dir); end

    mode = char(string(cfg.labelling_mode));
    if ~any(strcmp(mode, {'template_match', 'none'}))
        error('FSMS:LabellingMode', 'Unknown labelling_mode "%s" (use template_match or none).', mode);
    end
    K_candidates = unique(round(double(cfg.K_candidates(:)')));
    if strcmp(mode, 'template_match') && any(K_candidates > 7)
        error('FSMS:KAboveTemplate', ['Template labelling matches states one-to-one to the seven classes ' ...
            'A-G; K_candidates must not exceed 7 (got %s).'], mat2str(K_candidates));
    end

    common = struct( ...
        'K_candidates', K_candidates, ...
        'criterion', char(string(cfg.criterion)), ...
        'spm_prior_weight', double(cfg.spm_prior_weight), ...
        'compact_results_mat', logical(cfg.compact_results_mat), ...
        'run_participant_condition_fits', logical(cfg.run_participant_condition_fits), ...
        'resume', ~logical(cfg.redo));
    template_args = {};
    if ~isempty(char(string(cfg.template_file)))
        template_args = {'template_file', char(string(cfg.template_file))};
    end
    extra = cfg.extra_hierarchy_args;

    labelling = struct();
    switch mode
        case 'template_match'
            if cfg.redo
                error('FSMS:RedoNotSupported', ['redo is not supported with template labelling (stale child ' ...
                    'caches would be mixed with a refitted global solution). Use a fresh output_dir.']);
            end
            solution_file = fullfile(output_dir, 'global', 'global_solution.mat');
            labels_mat = fullfile(output_dir, 'global', 'global_manual_labels.mat');
            if ~isfile(solution_file) || ~isfile(labels_mat)
                fprintf('Pass 1/2: global fit for K in %s without template alignment...\n', mat2str(K_candidates));
                run_hierarchy(manifest_csv, output_dir, common, ...
                    [{'global_only', true, 'template_file', '', 'require_template_labels', false, ...
                      'global_manual_labels', {}}, extra]);
            end
            labelling = global_state_labels(output_dir, 'template_file', char(string(cfg.template_file)));
            fprintf('Pass 2/2: hierarchy at K=%d with the derived labels...\n', numel(labelling.labels));
            [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, ...
                [{'global_only', logical(cfg.global_only), 'require_template_labels', logical(cfg.require_template_labels), ...
                  'global_manual_labels', {}}, template_args, extra]);
            verify_labels_applied(output_dir, labelling.labels);
        case 'none'
            [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, ...
                [{'global_only', logical(cfg.global_only), 'require_template_labels', false, ...
                  'global_manual_labels', {}, 'template_file', ''}, extra]);
    end

    outputs = struct('HResults', HResults, 'results_mat', results_mat, ...
        'selected_K', double(HResults.selected_K), 'k_selection', HResults.k_selection, ...
        'labelling_mode', mode, 'labelling', labelling, 'env', env);
end

function [HResults, results_mat] = run_hierarchy(manifest_csv, output_dir, common, overrides)
    args = [struct_to_args(common), overrides];
    [HResults, results_mat] = fit_microstate_hierarchical_csv(manifest_csv, 'output_dir', output_dir, args{:});
end

function args = struct_to_args(S)
    names = fieldnames(S);
    args = cell(1, 2 * numel(names));
    for i = 1:numel(names)
        args{2 * i - 1} = names{i};
        args{2 * i} = S.(names{i});
    end
end

function verify_labels_applied(output_dir, labels)
% Pass 2 must have applied exactly the derived labels, via the saved-label path.
    S = load(fullfile(output_dir, 'global', 'global_solution.mat'), 'Fit');
    fit_labels = fsms_global_fit_labels(output_dir);
    if ~isequal(cellstr(fit_labels(:)), cellstr(string(labels(:))))
        error('FSMS:LabelsNotApplied', 'Hierarchy fitted-state labels (%s) differ from the derived labels (%s).', ...
            strjoin(fit_labels, ','), strjoin(string(labels), ','));
    end
    if size(S.Fit.centers, 1) ~= numel(labels)
        error('FSMS:LabelsNotApplied', 'The global solution has %d states but %d labels were derived.', ...
            size(S.Fit.centers, 1), numel(labels));
    end
    source = '';
    if isfield(S.Fit.template_alignment, 'manual_label_source')
        source = char(string(S.Fit.template_alignment.manual_label_source));
    end
    if ~strcmp(source, 'saved_manual_labels')
        error('FSMS:LabelsNotApplied', 'Pass 2 did not take the saved labels (manual_label_source = "%s").', source);
    end
end
