import math

from dmp.console import console


def compute_average_scores(results: dict) -> dict:
    outputs = results.get("inference_outputs", [])
    if not outputs:
        return {}

    dice_labels = set()
    hd_labels = set()
    sd_labels = set()
    dice_values = {}
    hd_values = {}
    sd_values = {}
    dice_avg_values = []
    hd_avg_values = []
    sd_avg_values = []

    for entry in outputs:
        metrics = entry.get("metrics", {})
        dice = metrics.get("dice", {})
        hd = metrics.get("hd", {})
        sd = metrics.get("sd", {})

        # Dice
        for k in dice.keys():
            if k != "dice_avg":
                dice_labels.add(k)
        for label in dice_labels:
            if label in dice:
                dice_values.setdefault(label, []).append(dice[label])
        if "dice_avg" in dice and isinstance(dice["dice_avg"], (float, int)):
            dice_avg_values.append(dice["dice_avg"])

        # HD
        if isinstance(hd, dict):
            for k in hd.keys():
                if k != "hd_avg":
                    hd_labels.add(k)
            for label in hd_labels:
                if label in hd:
                    hd_values.setdefault(label, []).append(hd[label])
            if "hd_avg" in hd and isinstance(hd["hd_avg"], (float, int)):
                hd_avg_values.append(hd["hd_avg"])

        # SD
        if isinstance(sd, dict):
            for k in sd.keys():
                if k != "sd_avg":
                    sd_labels.add(k)
            for label in sd_labels:
                if label in sd:
                    sd_values.setdefault(label, []).append(sd[label])
            if "sd_avg" in sd and isinstance(sd["sd_avg"], (float, int)):
                sd_avg_values.append(sd["sd_avg"])

    def mean_std(values):
        n = len(values)
        if n == 0:
            return None, None
        mean = sum(values) / n
        std = math.sqrt(sum((x - mean) ** 2 for x in values) / n)
        return mean, std

    results_dict = {}
    for label in dice_labels:
        mean, std = mean_std(dice_values.get(label, []))
        results_dict[f"dice_{label}_avg"] = mean
        results_dict[f"dice_{label}_std"] = std
    for label in hd_labels:
        mean, std = mean_std(hd_values.get(label, []))
        results_dict[f"hd_{label}_avg"] = mean
        results_dict[f"hd_{label}_std"] = std
    for label in sd_labels:
        mean, std = mean_std(sd_values.get(label, []))
        results_dict[f"sd_{label}_avg"] = mean
        results_dict[f"sd_{label}_std"] = std
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
