function c = match_template_labels_constants()
%MATCH_TEMPLATE_LABELS_CONSTANTS Parameters of the template-matching rule used to name fitted microstates.
%   The values are constants rather than function arguments so that a different value cannot be passed by
%   accident; changing one alters the script fingerprint recorded with every stage output.
%
%   rule_id                 identifier written to every output that depends on the labels
%   classes                 the MetaMaps template classes a fitted state can be matched to (A-G)
%   template_K              MetaMaps solution used as reference (K = 7: classes A-G)
%   max_states              largest number of fitted states that can be matched one-to-one (= number of classes)
%   strong_threshold        a state whose assigned |spatial correlation| is below this value is flagged
%                           (labelling_quality = 'below_threshold'); it never stops the run
%   allowed_modes           channel-matching modes accepted; 'first_common_channels' is never accepted
%   min_estimated_channels  fewer fitted-map channels than this make the template comparison meaningless
%   tie_tolerance           total-|r| tolerance within which two assignments count as tied
    c = struct();
    c.rule_id = 'match_template_labels';
    c.classes = {'A', 'B', 'C', 'D', 'E', 'F', 'G'};
    c.template_K = 7;
    c.max_states = 7;
    c.strong_threshold = 0.5;
    c.allowed_modes = {'labels', 'coordinates_idw_to_template'};
    c.min_estimated_channels = 8;
    c.tie_tolerance = 1e-9;
    c.nominal_label_semantics = ['State names are nominal template-class names: each fitted state is given the ' ...
        'class it is assigned to by the optimal one-to-one match; below_threshold means the descriptive ' ...
        'per-state name rests on a weak spatial correlation (|r| < strong_threshold).'];
end
