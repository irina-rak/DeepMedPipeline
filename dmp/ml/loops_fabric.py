import torch
from rich.progress import track
from src.console import console



def test_loop(net, testloader):
    """Evaluate the network on the entire test set."""
    # Alice: fabric is not used
    net.eval()

    with torch.no_grad():
        results = {
            key: torch.tensor(0.0, device=net.device)
            for key in net.signature.__required_keys__
        }
        for batch_idx, batch in track(
            enumerate(testloader),
            total=len(testloader),
            description="Validating...",
        ):
            res = net.validation_step(batch, batch_idx)
            for key in results.keys():
                results[key] += res[key]

    for key in results.keys():
        results[key] /= len(testloader)
        results[key] = results[key].item()
    return results
