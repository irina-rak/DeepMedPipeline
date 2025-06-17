import math


def compute_average_scores(results: dict) -> dict:
    """
    Compute the average and standard deviation of each metric in the results dictionary produced by inference_module.run().
    Handles per-label dice scores and overall averages for dice, hd_avg, and sd_avg.
    Returns a dictionary with the average and std for each metric and per-label dice.
    """
    outputs = results.get("inference_outputs", [])
    if not outputs:
        return {}

    dice_labels = set()
    dice_values = {}  # label -> list of values
    dice_avg_values = []
    hd_avg_values = []
    sd_avg_values = []

    for entry in outputs:
        metrics = entry.get("metrics", {})
        dice = metrics.get("dice", {})
        
        for k in dice.keys():
            if k != "dice_avg":
                dice_labels.add(k)
                
        for label in dice_labels:
            if label in dice:
                dice_values.setdefault(label, []).append(dice[label])
                
        if "dice_avg" in dice:
            dice_avg_values.append(dice["dice_avg"])
            
        hd = metrics.get("hd_avg", [None])
        if isinstance(hd, list) and hd and hd[0] is not None:
            hd_avg_values.append(hd[0])
        sd = metrics.get("sd_avg", [None])
        if isinstance(sd, list) and sd and sd[0] is not None:
            sd_avg_values.append(sd[0])

    def mean_std(values):
        n = len(values)
        if n == 0:
            return None, None
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
        return mean, std

    # Compute averages and stds
    results_dict = {}
    for label in dice_labels:
        mean, std = mean_std(dice_values.get(label, []))
        results_dict[f"dice_{label}_avg"] = mean
        results_dict[f"dice_{label}_std"] = std
    mean, std = mean_std(dice_avg_values)
    results_dict["dice_avg"] = mean
    results_dict["dice_std"] = std
    mean, std = mean_std(hd_avg_values)
    results_dict["hd_avg"] = mean
    results_dict["hd_std"] = std
    mean, std = mean_std(sd_avg_values)
    results_dict["sd_avg"] = mean
    results_dict["sd_std"] = std

    return results_dict
