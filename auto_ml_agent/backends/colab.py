class ColabAccessRequired(RuntimeError):
    pass

def run_colab_job(*args, **kwargs):
    
    raise ColabAccessRequired(
        "Google Colab execution is not configured. The Colab API access request "
        "must be approved and the supported API credentials/runtime workflow "
        "must be set up before remote execution can start. No job was submitted."
    )
