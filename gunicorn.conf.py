# Loaded automatically by gunicorn from the working directory.
# A search runs subsequence DTW against every track, which takes minutes on a
# small shared CPU; gunicorn's default 30 s timeout would kill the worker mid-search.
timeout = 600
workers = 1
