from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(name="tree",
      ext_modules=cythonize("tree.pyx",
                            annotate=True,
                            language='c++',
                            ),
      include_dirs=[numpy.get_include()]
)
