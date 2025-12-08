# GitHub Actions Configuration

## docs.yml - README Processing Workflow

This workflow automatically processes `README.template.md` and generates `README.md` with expanded MyST include directives.

### How it works:

1. **Source**: Edit `README.template.md` which contains MyST `{include}` directives
2. **Processing**: The Python script `.github/scripts/process_readme.py` expands the includes
3. **Output**: Generates `README.md` with full content for GitHub/PyPI display

### Triggers:

- Push to `README.template.md`
- Changes to any files in `docs/source/_snippets/`  
- Manual dispatch via GitHub Actions UI

### Benefits:

- **DRY Principle**: Keep content in snippet files, reuse across docs and README
- **Sphinx Compatibility**: `README.template.md` works with MyST parser in Sphinx
- **GitHub Compatibility**: Generated `README.md` displays properly on GitHub/PyPI
- **Automated**: No manual sync needed between docs and README

### Usage:

1. Edit content in `docs/source/_snippets/` files
2. Reference them in `README.template.md` using `{include}` directives
3. Push changes - GitHub Action will auto-update `README.md`