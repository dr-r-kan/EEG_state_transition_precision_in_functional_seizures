function [K_selected, best_score, score_by_k, details] = select_spm_vb_k_by_criterion(Results, criterion)
%SELECT_SPM_VB_K_BY_CRITERION Choose the number of microstates K from the VB-GMM free energy.
%
%   [K_selected, best_score, score_by_k, details] = select_spm_vb_k_by_criterion(Results, criterion)
%
%   Results     struct with K_candidates (vector) and free_energy_vals (vector, same length; the SPM
%               variational free energy, i.e. the lower bound on the log evidence, larger is better)
%   criterion   'free_energy_elbow' (default) or 'free_energy'
%
%   'free_energy_elbow'
%     1. Candidates whose free energy is not finite or is exactly 0 (failed fits) are ignored.
%     2. K and free energy of the remaining candidates are each min-max scaled to [0, 1]; the scaled curve
%        is the sequence of points (k_i, f_i) in ascending K.
%     3. The elbow score of candidate i is the perpendicular distance from (k_i, f_i) to the chord joining
%        the first and the last scaled point (one-sided: only points above the chord count, a dip below the chord scores 0). The first and
%        the last candidate score 0 by construction, so the smallest and the largest candidate K are
%        never selected by the elbow itself.
%     4. K_selected is the candidate with the largest elbow score; exact ties go to the smallest K.
%     5. Degenerate curve (fewer than three valid candidates, or every elbow score <= eps): K_selected is
%        the candidate with the largest free energy (ties to the smallest K).
%
%   'free_energy'  the candidate with the largest free energy (ties to the smallest K).
%
%   score_by_k has one entry per element of Results.K_candidates: the elbow score for 'free_energy_elbow'
%   (NaN for ignored candidates), the free energy for 'free_energy'. details.rule holds the rule as text.

    if nargin < 2 || isempty(criterion)
        criterion = 'free_energy_elbow';
    end
    criterion = lower(strtrim(char(string(criterion))));
    K_all = double(Results.K_candidates(:));
    free_energy = double(Results.free_energy_vals(:));
    if numel(free_energy) ~= numel(K_all)
        error('free_energy_vals has %d entries but there are %d K candidates.', numel(free_energy), numel(K_all));
    end

    score_by_k = nan(size(K_all));
    K_selected = NaN;
    best_score = NaN;
    details = struct('criterion', criterion, 'rule', '', 'valid_K', [], 'curve', free_energy, ...
        'degenerate', false);

    valid = isfinite(free_energy) & free_energy ~= 0;
    if ~any(valid)
        return;
    end
    [K_valid, order] = sort(K_all(valid));
    valid_idx = find(valid);
    valid_idx = valid_idx(order);
    fe_valid = free_energy(valid_idx);
    details.valid_K = K_valid;

    switch criterion
        case 'free_energy_elbow'
            details.rule = ['Free-energy elbow: K and free energy of the valid candidates are min-max scaled ' ...
                'to [0,1]; the score of a candidate is its perpendicular distance above the chord joining ' ...
                'the first and last candidate (points on or below the chord and the endpoints score 0); the candidate with the largest score is ' ...
                'selected, ties to the smallest K; if fewer than 3 candidates are valid or all scores are ' ...
                '<= eps the candidate with the largest free energy is selected.'];
            scores = elbow_scores(K_valid, fe_valid);
            score_by_k(valid_idx) = scores;
            if numel(K_valid) < 3 || all(scores <= eps)
                details.degenerate = true;
                [~, pick] = max(fe_valid);
            else
                [~, pick] = max(scores);
            end
        case 'free_energy'
            details.rule = 'Largest free energy among the valid candidates; ties to the smallest K.';
            score_by_k(valid_idx) = fe_valid;
            [~, pick] = max(fe_valid);
        otherwise
            error('Unknown K-selection criterion "%s" (use free_energy_elbow or free_energy).', criterion);
    end
    K_selected = K_valid(pick);
    best_score = score_by_k(valid_idx(pick));
    details.selected_K = K_selected;
end

function scores = elbow_scores(K, fe)
    n = numel(K);
    scores = zeros(n, 1);
    if n < 3
        return;
    end
    k = scale01(K);
    f = scale01(fe);
    p1 = [k(1), f(1)];
    chord = [k(end), f(end)] - p1;
    chord_length = hypot(chord(1), chord(2));
    if chord_length <= eps
        return;
    end
    for i = 2:(n - 1)
        offset = [k(i), f(i)] - p1;
        scores(i) = max(0, (chord(1) * offset(2) - chord(2) * offset(1)) / chord_length);   % above the chord only
    end
end

function x = scale01(x)
    x = double(x(:));
    span = max(x) - min(x);
    if span <= eps
        x(:) = 0;
    else
        x = (x - min(x)) / span;
    end
end
