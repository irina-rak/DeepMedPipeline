


def compute_average_scores(results: dict) -> dict:
    """
    Compute the average of each metric in the results dictionary produced by inference_module.run().
    Expects a structure with an 'inference_outputs' key containing a list of dicts, each with a 'metrics' dict.
    Returns a dictionary with the average for each metric.
    """
    outputs = results.get("inference_outputs", [])
    if not outputs:
        return {}

    # Collect all metric keys
    metric_keys = set()
    for entry in outputs:
        metric_keys.update(entry.get("metrics", {}).keys())

    # Initialize sums
    sums = {k: 0.0 for k in metric_keys}
    count = 0
    for entry in outputs:
        metrics = entry.get("metrics", {})
        for k in metric_keys:
            if k in metrics:
                sums[k] += metrics[k]
        count += 1

    if count == 0:
        return {k: None for k in metric_keys}

    averages = {k: sums[k] / count for k in metric_keys}
    return averages
