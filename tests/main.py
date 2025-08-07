from mod_b import B1

import sys
import importlib
import site
import os
from import_deps import ModuleSet
from pathlib import Path
from import_deps import ast_imports


b = B1(10, 20)

ast_imports('C:/Users/jonaf/Documents/projects/DeepUtilsJ-1/tests/mod_b.py')

# pkg_paths = Path("C:/Users/jonaf/Documents/projects/DeepUtilsJ-1/tests").glob('**/*.py')
# module_set = ModuleSet([str(p) for p in pkg_paths])

# # then you can get the set of imports
# for imported in module_set.mod_imports('foo.foo_a'):
#     print(imported)


print('done')