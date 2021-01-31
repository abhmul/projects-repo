# Resources

1. https://tfhub.dev/google/ - for finding pretrained models and embeddings
2. https://github.com/tensorflow/datasets - for finding datasets to test on

# Structure

## lib

- Contains files of code that can depend on each other
- Common patterns
  - File of simple functions (possibly wrapped in class) whose functionality is related and dependencies are the same
  - File of library-related utils (prefixed with library name). For example
    - np_utils.py
    - tf_assertions.py

### Dependencies

No folder structure is used, instead all code files are stored in the same folder. I'm using this structure because the code structure does not have well-defined hierarchy. Rather all these files may be equally useful by scripts. Removing a folder hierarchy (counterintuitively) makes it easier to organize my code, as I don't need to think about what folder a piece of code goes in (this is especially confusing when I didn't have a good sense of what folder structure I needed and a codefile partially belongs to multiple folders). The lack of folder structure also allows me to create a good dependency graph as I go.

If I'm unsure whether a new addition to a file will create a circular dependency, then err on the side of just creating a new file. Safe cases:

- Newly added code to a file only uses existing dependencies or underlying python stdlib or installed libraries.
- Newly added code only uses known leaf dependencies (e.g. util files)

**Utils** files should not take on any dependencies from `lib`. This will ensure they are always leaf nodes.

TODO: Should I merge all utils files into a single utils file?

## Scripts

These are executable code files that potentially take in command line arguments. They can depend on code from `lib`. **Scripts** shoujd not depend on each other. If they need to, then the dependee code should be moved into `lib`.

## Notebooks

Similar to `scripts`, but jupyter notebooks instead executable code files.