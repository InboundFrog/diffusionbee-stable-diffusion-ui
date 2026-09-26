# PyInstaller runtime hook. PyInstaller stores bundled module code with relative co_filename (torch/nn/...).
# torch.compile (Dynamo) tells torch internals from model code by absolute path prefix, so frozen it traces into
# all of torch and fails with "maximum recursion depth exceeded". Give each module's code its absolute path,
# as CPython's own SourceLoader does.
import _imp

import pyimod02_importers

_get_code = pyimod02_importers.PyiFrozenLoader.get_code


def get_code(self, fullname):
    code = _get_code(self, fullname)
    if code is not None:
        _imp._fix_co_filename(code, self.path)
    return code


pyimod02_importers.PyiFrozenLoader.get_code = get_code
