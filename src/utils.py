import os


def configure_r():
    """
    Configure R settings for the cdt package.
    Is necessary for SID metric evaluation.
    """
    import cdt

    cdt.SETTINGS.rpath = '/usr/bin/Rscript'

    # Configure R to run non-interactively and disable pagination
    os.environ['R_INTERACTIVE'] = 'FALSE'  # Force non-interactive mode
    os.environ['R_PAPERSIZE'] = 'letter'   # Avoid unnecessary prompts
    os.environ['PAGER'] = 'cat'            # Disable R’s pager (no "q" prompt)

    # Set R options to suppress interactive checks (e.g., package updates)
    os.environ['R_OPTS'] = '--no-save --no-restore --quiet'  # Silent execution

    os.environ['R_LIBS_USER'] = os.path.expanduser('~/R/library')
