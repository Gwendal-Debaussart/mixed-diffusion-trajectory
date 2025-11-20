
def reduced_name(method_name:str) -> str:
    """
    Reduces method names to more concise versions for plotting.
    Args:
        method_name (str): The original method name.
    Returns:
        str: The reduced method name.
    """
    reductions = {
        "Alternating Diffusion": "AD",
        "Integrated Diffusion Maps": "ID",
        "Multi-view Diffusion Maps": "MVD",
        "Direct MDT": "MDT-chs",
        "Composite Diffusion Maps": "ComD",
        "Cross Diffusion Maps": "CrD",
        "Powered Alternating Diffusion": "p-AD",
        "Random Convex MDT": "MDT-CVX-Rand",
        "Random MDT": "MDT-Rand",
        "Beam-Search MDT": "MDT-Bsc",
        "Single-view Diffusion Maps (view 1)": "DM (V1)",
        "Single-view Diffusion Maps (view 2)": "DM (V2)",
    }
    return reductions.get(method_name, method_name)