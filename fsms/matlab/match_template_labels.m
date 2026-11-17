function result = match_template_labels(centers, common_labels, common_chanlocs, varargin)
%MATCH_TEMPLATE_LABELS Name fitted microstates by optimal one-to-one matching to the MetaMaps classes.
%
%   result = match_template_labels(centers, common_labels, common_chanlocs, 'template_file', F)
%
%   centers          K x C matrix (fitted state x channel), 2 <= K <= 7
%   common_labels    C-element cellstr of channel labels
%   common_chanlocs  EEGLAB-style struct array of those channels (positions in MNE RAS head coordinates)
%                    or []
%
%   RULE
%     1. Spatial correlation: |r| between every fitted map and every map of the MetaMaps_2023_06 K = 7
%        template (classes A-G), polarity invariant. Channels are matched by label when at least 16
%        labels match (or half of the smaller montage), otherwise by interpolating the fitted maps to the
%        template electrode positions (inverse-distance weighting on unit-sphere coordinates).
%        The 'first_common_channels' fallback of align_microstates_to_template is not accepted.
%     2. Assignment: all seven classes are available to every fitted state and no class is required to be
%        present. The assignment is the exact optimum over all injective maps of the K fitted states onto
%        the seven classes (7!/(7-K)! maps, at most 5040) maximising the SUM of assigned |r|.
%     3. Tie-break: among assignments whose total is within tie_tolerance (1e-9) of the maximum, the
%        lexicographically smallest assignment vector is taken (entry k = index 1..7 of the class
%        A..G given to fitted state k). More than one tied optimum is reported
%        (n_tied_optimal_assignments) and counts as a technical failure, because it only occurs for
%        identical fitted maps.
%     4. Quality flag (descriptive, never a stop): labelling_quality = 'ok' when every assigned |r| >=
%        strong_threshold (0.5), otherwise 'below_threshold' with the list of the states below it. The
%        flag does not change the label of any state.
%     5. Technical failures (result.accepted = false; the caller stops): channel-matching mode not in
%        {labels, coordinates_idw_to_template}; fewer than 8 fitted-map channels in the comparison;
%        non-finite correlations; tied optimal assignments. A K outside 2..7 or non-finite maps raise an
%        error.
%
%   Constants live in match_template_labels_constants.m. The function is pure: it reads the template and
%   returns the evidence; it writes nothing.

    p = inputParser;
    addParameter(p, 'template_file', '', @(x) ischar(x) || isstring(x));
    parse(p, varargin{:});
    const = match_template_labels_constants();
    template_file = char(string(p.Results.template_file));
    if isempty(template_file)
        env = fsms_paths();
        template_file = fullfile(env.vg_root, 'MetaMaps_2023_06.set');
    end

    centers = double(centers);
    K = size(centers, 1);
    if ~ismatrix(centers) || K < 2 || K > const.max_states
        error('FSMS:LabellingBadInput', 'Template matching needs between 2 and %d fitted maps; got %d.', ...
            const.max_states, K);
    end
    common_labels = cellstr(string(common_labels(:)));
    if numel(common_labels) ~= size(centers, 2)
        error('FSMS:LabellingBadInput', 'common_labels has %d entries but the maps have %d channels.', ...
            numel(common_labels), size(centers, 2));
    end
    if any(~isfinite(centers(:)))
        error('FSMS:LabellingBadInput', 'Fitted global maps contain non-finite values.');
    end
    reasons = {};

    alignment = align_microstates_to_template(centers, template_file, ...
        'estimated_channel_labels', common_labels, ...
        'estimated_chanlocs', common_chanlocs, ...
        'template_K', const.template_K, ...
        'strong_threshold', const.strong_threshold);
    template_order = cellstr(string(alignment.template_labels(:)));
    full_abs = alignment.corr_matrix;                        % K x 7, |r|, columns in template order
    class_columns = zeros(1, numel(const.classes));
    for c = 1:numel(const.classes)
        hit = find(strcmp(template_order, const.classes{c}), 1);
        if isempty(hit)
            error('FSMS:LabellingTemplate', 'Template %s lacks class %s.', template_file, const.classes{c});
        end
        class_columns(c) = hit;
    end
    corr_abs = full_abs(:, class_columns);                   % K x 7, columns A..G
    signed_full = alignment.estimated_common_maps * alignment.template_common_maps';
    corr_signed = signed_full(:, class_columns);

    mode = char(string(alignment.channel_match_mode));
    n_used = numel(alignment.estimated_channel_indices);
    if ~any(strcmp(mode, const.allowed_modes))
        reasons{end + 1} = sprintf('CHANNEL_MATCH_MODE_NOT_ALLOWED:%s', mode); %#ok<AGROW>
    end
    if n_used < const.min_estimated_channels
        reasons{end + 1} = sprintf('TOO_FEW_COMMON_CHANNELS:n=%d,minimum=%d', n_used, const.min_estimated_channels); %#ok<AGROW>
    end
    if any(~isfinite(corr_abs(:)))
        reasons{end + 1} = 'NONFINITE_CORRELATION'; %#ok<AGROW>
    end

    [assignment, best_total, runner_up, n_ties] = optimal_assignment(corr_abs, const.tie_tolerance);
    if n_ties > 1
        reasons{end + 1} = sprintf('AMBIGUOUS_ASSIGNMENT:%d_tied_optimal_assignments', n_ties); %#ok<AGROW>
    end
    labels = const.classes(assignment)';
    assigned_abs = zeros(K, 1);
    assigned_signed = zeros(K, 1);
    runner_up_label = cell(K, 1);
    runner_up_abs = zeros(K, 1);
    for k = 1:K
        assigned_abs(k) = corr_abs(k, assignment(k));
        assigned_signed(k) = corr_signed(k, assignment(k));
        others = corr_abs(k, :);
        others(assignment(k)) = -Inf;
        [runner_up_abs(k), j] = max(others);
        runner_up_label{k} = const.classes{j};
    end
    if numel(unique(labels)) ~= K
        reasons{end + 1} = 'LABELS_NOT_UNIQUE'; %#ok<AGROW>
    end
    polarity = ones(K, 1);
    polarity(assigned_signed < 0) = -1;

    below = find(~(assigned_abs >= const.strong_threshold));
    if isempty(below)
        quality = 'ok';
        states_below = struct('state_index', {}, 'label', {}, 'assigned_abs_r', {});
    else
        quality = 'below_threshold';
        states_below = struct('state_index', num2cell(below(:))', 'label', labels(below)', ...
            'assigned_abs_r', num2cell(assigned_abs(below))');
    end

    result = struct();
    result.rule_id = const.rule_id;
    result.accepted = isempty(reasons);                      % false only for technical failures
    result.reasons = reasons(:);
    result.n_states = K;
    result.labelling_quality = quality;
    result.states_below_threshold = states_below;
    result.nominal_label_semantics = const.nominal_label_semantics;
    result.classes = const.classes(:);
    result.classes_unused = setdiff(const.classes(:), labels(:), 'stable');
    result.labels = labels;
    result.state_index = (1:K)';
    result.assignment_class_index = assignment;
    result.assigned_abs_r = assigned_abs;
    result.assigned_signed_r = assigned_signed;
    result.below_threshold = ~(assigned_abs >= const.strong_threshold);
    result.polarity = polarity;
    result.runner_up_label = runner_up_label;
    result.runner_up_abs_r = runner_up_abs;
    result.corr_abs = corr_abs;
    result.corr_abs_all_template_classes = full_abs;
    result.template_class_order = template_order;
    result.total_abs_r = best_total;
    result.runner_up_total_abs_r = runner_up;
    result.margin_total_abs_r = best_total - runner_up;
    result.n_tied_optimal_assignments = n_ties;
    result.strong_threshold = const.strong_threshold;
    result.tie_tolerance = const.tie_tolerance;
    result.channel_match_mode = mode;
    result.n_common_channels = double(alignment.n_common_channels);
    result.n_estimated_channels_used = n_used;
    result.min_label_matches = double(alignment.min_label_matches);
    result.matched_channel_labels = cellstr(string(alignment.matched_channel_labels(:)));
    result.template_file = template_file;
    result.template_K = const.template_K;
end

function [assignment, best_total, runner_up, n_ties] = optimal_assignment(corr_abs, tolerance)
% Exact maximiser of the summed |r| over all injective maps of the K rows onto the columns.
    [K, n_classes] = size(corr_abs);
    maps = injective_maps(K, n_classes);                      % rows in ascending lexicographic order
    n_maps = size(maps, 1);
    score = zeros(n_maps, 1);
    for k = 1:K
        score = score + corr_abs(sub2ind(size(corr_abs), repmat(k, n_maps, 1), maps(:, k)));
    end
    safe = score;
    safe(~isfinite(safe)) = -Inf;
    best_total = max(safe);
    tied = safe >= best_total - tolerance;
    n_ties = sum(tied);
    best_row = find(tied, 1, 'first');
    assignment = maps(best_row, :)';
    lower_scores = safe(~tied);
    if isempty(lower_scores)
        runner_up = NaN;
    else
        runner_up = max(lower_scores);
    end
end

function maps = injective_maps(K, n_classes)
% All K-tuples of distinct integers in 1..n_classes, ascending lexicographic order.
    combos = nchoosek(1:n_classes, K);
    maps = zeros(0, K);
    for i = 1:size(combos, 1)
        maps = [maps; perms(combos(i, :))]; %#ok<AGROW>
    end
    maps = sortrows(maps);
end
