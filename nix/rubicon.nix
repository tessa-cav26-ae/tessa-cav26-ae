{ python3Packages, stormpy, fetchFromGitHub }:

python3Packages.buildPythonPackage rec {

  pname = "rubicon";
  version = "ce2db622470a98b85da5be8dfdad7d9a41f8757d";
  # https://github.com/sjunges/rubicon
  src = fetchFromGitHub {
    owner = "sjunges";
    repo = "rubicon";
    rev = version;
    hash = "sha256-A7cTTBEqLZtO7avzNzvpDA3AnhvdVEYiiwcIekgSWdE=";
  };

  postPatch = ''
    cat > rubicon-inline.patch <<'PATCH'
diff --git a/rubicon/regression.py b/rubicon/regression.py
index 32568cb..fd4a657 100644
--- a/rubicon/regression.py
+++ b/rubicon/regression.py
@@ -2,8 +2,8 @@ import os.path
 import json
 import logging
 import rubicon
-import dice_wrapper
-import storm_wrapper
+from . import dice_wrapper
+from . import storm_wrapper
 import click
 import numpy as np
 from pathlib import Path
diff --git a/rubicon/rubicon.py b/rubicon/rubicon.py
index efb6767..e78eb4d 100644
--- a/rubicon/rubicon.py
+++ b/rubicon/rubicon.py
@@ -2,7 +2,7 @@ import itertools
 import re
 import math
 import os
-import dice_wrapper
+from . import dice_wrapper

 import click
 import stormpy
diff --git a/setup.py b/setup.py
index be847fe..d2ad1d2 100644
--- a/setup.py
+++ b/setup.py
@@ -20,4 +20,9 @@ setup(
         "stormpy>=1.3.0", "click", "numpy"
     ],
     python_requires='>=3',
+    entry_points={
+        'console_scripts': [
+            'rubicon=rubicon.rubicon:translate_cli',
+        ],
+    },
 )
diff --git a/rubicon/storm_wrapper.py b/rubicon/storm_wrapper.py
index 7a7d596..0f4c0f2 100644
--- a/rubicon/storm_wrapper.py
+++ b/rubicon/storm_wrapper.py
@@ -10,7 +10,7 @@ class Storm:
     def __init__(self, cwd, path, arguments, symbolic, timeout):
         self._path = path.split(" ")
         self._cwd = cwd
         self._timeout = timeout
-        self._id = "storm" + ("-dd" if symbolic else "-sparse")
+        self._id = "storm" + ("-add" if symbolic else "-spm")
         if len(arguments) > 0:
             self._id += "-" + "-".join(arguments)
         if arguments is None:
PATCH

    patch -p1 < rubicon-inline.patch
  '';

  propagatedBuildInputs = (with python3Packages; [
    setuptools
    click
    numpy
  ]) ++ [ stormpy ];

  format = "setuptools";

  pythonImportsCheck = [ "rubicon" ];
}