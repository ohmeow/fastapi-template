import os

# for vscode debugging
def wait_for_debugger():
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()