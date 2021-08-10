Writing documentation
=======================================

To edit::

   cd docs/source
   # Change things.

To build::

   cd docs
   python -m sphinx source .

Open ``./index.html`` to examine the updated documentation.

To commit::

   cd ..
   isort .
   black .
   git add $FILES
   git commit -m "Update doc"
   git push origin main