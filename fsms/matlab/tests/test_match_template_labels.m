function test_match_template_labels()
%TEST_MATCH_TEMPLATE_LABELS Unit tests of match_template_labels on synthetic maps built from the template.
%
%   Run from MATLAB:   addpath('fsms/matlab/tests'); test_match_template_labels
%   Headless:          matlab -batch "addpath('fsms/matlab/tests'); test_match_template_labels"
%
%   Checks that any subset of the seven classes (including D) is recovered with no class forced, that K = 7
%   returns all seven classes, that the assignment is the exact optimum (against an independent search and
%   matchpairs), that weak matches are flagged and never relabelled, and that technical failures are reported.

    here = fileparts(mfilename('fullpath'));
    addpath(fileparts(here));
    env = fsms_paths();
    template_file = fullfile(env.vg_root, 'MetaMaps_2023_06.set');
    [T, template_labels, channel_labels, chanlocs] = load_metamaps_templates(template_file, 'K', 7);
    template_labels = cellstr(string(template_labels(:)))';
    assert(isequal(template_labels, {'A', 'B', 'C', 'D', 'E', 'F', 'G'}), 'template classes are not A-G');
    rng(7, 'twister');
    n_failed = 0;

    % 1. K = 4 states built from classes B, D, F, G (shuffled order, random polarity, noise)
    classes = {'B', 'D', 'F', 'G'};
    order = [3 1 4 2];                                       % fitted state k carries classes(order(k))
    [maps, expected] = synthetic_states(T, template_labels, classes(order), 0.15);
    r = match_template_labels(maps, channel_labels, chanlocs, 'template_file', template_file);
    n_failed = check(isequal(r.labels(:)', expected), 'K=4 {B,D,F,G}: labels %s expected %s', ...
        strjoin(r.labels, ''), strjoin(expected, ''), n_failed);
    n_failed = check(r.accepted && strcmp(r.labelling_quality, 'ok'), 'K=4: accepted and quality ok', n_failed);
    n_failed = check(any(strcmp(r.labels, 'D')), 'K=4: class D is allowed and present', n_failed);
    n_failed = check(isequal(sort(r.classes_unused(:))', {'A', 'C', 'E'}), 'K=4: unused classes are A, C, E', n_failed);
    n_failed = check(r.n_states == 4 && size(r.corr_abs, 2) == 7, 'K=4: correlation matrix is 4 x 7', n_failed);

    % 2. K = 7: all seven classes, shuffled
    perm = randperm(7);
    [maps, expected] = synthetic_states(T, template_labels, template_labels(perm), 0.15);
    r = match_template_labels(maps, channel_labels, chanlocs, 'template_file', template_file);
    n_failed = check(isequal(r.labels(:)', expected) && isequal(sort(r.labels(:))', template_labels), ...
        'K=7: all seven classes returned', n_failed);

    % 3. Small K, including classes that a six-class rule would exclude
    subsets = {{'D', 'E'}, {'A', 'D', 'G'}, {'C', 'D'}, {'B', 'C', 'E', 'F', 'G'}, {'A', 'B', 'C', 'D', 'E', 'F'}};
    for s = 1:numel(subsets)
        sub = subsets{s}(randperm(numel(subsets{s})));
        [maps, expected] = synthetic_states(T, template_labels, sub, 0.10);
        r = match_template_labels(maps, channel_labels, chanlocs, 'template_file', template_file);
        n_failed = check(isequal(r.labels(:)', expected), 'K=%d {%s}: labels %s expected %s', ...
            numel(sub), strjoin(sub, ''), strjoin(r.labels, ''), strjoin(expected, ''), n_failed);
    end

    % 4. Exactness: total |r| equals the optimum found by independent methods on noisy maps
    for K = [2 3 5 6]
        maps = T(randperm(7, K), :) + 0.9 * randn(K, size(T, 2)) .* std(T(:));
        r = match_template_labels(maps, channel_labels, chanlocs, 'template_file', template_file);
        C = r.corr_abs;
        best_perm = brute_force_total(C);
        assignment = matchpairs(2 - C, 100);                 % all rows matched: cost 2 - |r|
        best_mp = sum(C(sub2ind(size(C), assignment(:, 1), assignment(:, 2))));
        n_failed = check(abs(r.total_abs_r - best_perm) < 1e-9 && abs(r.total_abs_r - best_mp) < 1e-9, ...
            'K=%d: total |r| %.6f vs brute force %.6f vs matchpairs %.6f', K, r.total_abs_r, best_perm, best_mp, n_failed);
        n_failed = check(numel(unique(r.labels)) == K, 'K=%d: labels unique', K, n_failed);
    end

    % 5. Weak matches are flagged, keep their label, and never stop the run
    [maps, ~] = synthetic_states(T, template_labels, {'B', 'F', 'G'}, 0.10);
    maps(2, :) = randn(1, size(maps, 2));                    % a state unrelated to every template
    r = match_template_labels(maps, channel_labels, chanlocs, 'template_file', template_file);
    n_failed = check(r.accepted, 'weak state: run is not rejected', n_failed);
    n_failed = check(any(r.below_threshold) && strcmp(r.labelling_quality, 'below_threshold'), ...
        'weak state: flagged below_threshold', n_failed);
    n_failed = check(all(~cellfun(@isempty, r.labels)) && numel(unique(r.labels)) == 3, ...
        'weak state: every state still has a unique label', n_failed);

    % 6. Technical failures
    n_failed = check(throws(@() match_template_labels(randn(8, numel(channel_labels)), channel_labels, chanlocs, 'template_file', template_file)), ...
        'K=8 is rejected', n_failed);
    n_failed = check(throws(@() match_template_labels(randn(1, numel(channel_labels)), channel_labels, chanlocs, 'template_file', template_file)), ...
        'K=1 is rejected', n_failed);
    bad = T(1:3, :); bad(2, 5) = NaN;
    n_failed = check(throws(@() match_template_labels(bad, channel_labels, chanlocs, 'template_file', template_file)), ...
        'non-finite maps are rejected', n_failed);
    unlabelled = arrayfun(@(i) sprintf('X%d', i), 1:numel(channel_labels), 'UniformOutput', false);
    r = match_template_labels(T(1:3, :), unlabelled, [], 'template_file', template_file);
    n_failed = check(~r.accepted && any(contains(r.reasons, 'CHANNEL_MATCH_MODE_NOT_ALLOWED')), ...
        'silent first-common-channels fallback is a technical failure', n_failed);
    r = match_template_labels(T(1:3, :), unlabelled, chanlocs, 'template_file', template_file);
    n_failed = check(r.accepted && strcmp(r.channel_match_mode, 'coordinates_idw_to_template'), ...
        'coordinate interpolation is accepted when labels do not match', n_failed);
    same = [T(2, :); T(2, :); T(5, :)];
    r = match_template_labels(same, channel_labels, chanlocs, 'template_file', template_file);
    n_failed = check(~r.accepted && r.n_tied_optimal_assignments > 1, 'identical maps: tied optimum reported as failure', n_failed);

    if n_failed > 0
        error('FSMS:TestFailed', '%d check(s) of test_match_template_labels failed.', n_failed);
    end
    fprintf('test_match_template_labels: all checks passed.\n');
end

function [maps, expected] = synthetic_states(T, template_labels, classes, noise)
% Template maps of the given classes, random polarity, additive noise (relative to the map spread).
    K = numel(classes);
    maps = zeros(K, size(T, 2));
    for k = 1:K
        row = find(strcmp(template_labels, classes{k}), 1);
        maps(k, :) = (2 * (rand < 0.5) - 1) * T(row, :) + noise * std(T(row, :)) * randn(1, size(T, 2));
    end
    expected = classes(:)';
end

function best = brute_force_total(C)
% Maximum summed |r| over all injective maps, by plain enumeration (independent of the code under test).
    [K, n] = size(C);
    best = -Inf;
    idx = nchoosek(1:n, K);
    for i = 1:size(idx, 1)
        P = perms(idx(i, :));
        for j = 1:size(P, 1)
            best = max(best, sum(C(sub2ind(size(C), 1:K, P(j, :)))));
        end
    end
end

function tf = throws(fn)
    tf = false;
    try
        fn();
    catch
        tf = true;
    end
end

function n_failed = check(condition, varargin)
% check(cond, msg, [msg args...], n_failed) increments the failure counter and reports.
    n_failed = varargin{end};
    msg = varargin{1};
    args = varargin(2:end - 1);
    if ~condition
        n_failed = n_failed + 1;
        fprintf(2, 'FAIL: %s\n', sprintf(msg, args{:}));
    else
        fprintf('ok:   %s\n', sprintf(msg, args{:}));
    end
end
