function root_dir = setup_microstate_paths(root_dir)
% SETUP_MICROSTATE_PATHS Add the helper folders of the vendored VG-microstates code to the path.

    if nargin < 1 || isempty(root_dir)
        root_dir = fileparts(mfilename('fullpath'));
    else
        root_dir = char(root_dir);
    end

    add_if_dir(fullfile(root_dir, '_internal', 'matlab'));
    add_if_dir(fullfile(root_dir, '_internal', 'matlab', 'compat', 'spm_bsd'));
end

function add_if_dir(path_to_add)
    if isfolder(path_to_add) && ~contains(path, path_to_add)
        addpath(path_to_add);
    end
end
