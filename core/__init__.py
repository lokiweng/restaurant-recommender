"""
core -- the recommendation engine.

Everything in this package is plain Python: no Streamlit imports anywhere.
That boundary is deliberate. It means the models can be exercised by the test
suite and by the command-line evaluation script without starting a web server,
and it keeps the algorithms readable as algorithms rather than as page code.
"""
