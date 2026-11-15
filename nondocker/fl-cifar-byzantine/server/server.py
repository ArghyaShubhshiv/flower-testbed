"""
Flower server: FedAvgDelta strategy + centralized evaluation on the held-out
CIFAR-100 test split. `run_server()` is meant to be called from a thread by
run_simulation.py.
"""
import os
import traceback

import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters

from common import attacks
from common.data_utils import get_server_test_loader
from common.model import CIFAR100CNN, set_parameters, trainable_mask
from server.strategy import FedAvgDelta


def get_evaluate_fn(testloader, results_dir, run_name, attack, device, torch_threads):
    net = CIFAR100CNN().to(device)
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"{run_name}.csv")
    with open(csv_path, "w") as f:
        f.write("round,loss,accuracy,attack,num_byzantine,attack_scale\n")

    def evaluate(server_round, parameters, config):
        torch.set_num_threads(torch_threads)
        set_parameters(net, parameters)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                outputs = net(images)
                loss_sum += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        avg_loss = loss_sum / total
        print(f"[Server round {server_round}] test_loss={avg_loss:.4f} test_accuracy={accuracy:.4f}", flush=True)
        with open(csv_path, "a") as f:
            f.write(f"{server_round},{avg_loss:.6f},{accuracy:.6f},{attack.name},"
                    f"{attack.num_byzantine},{attack.scale}\n")
        return avg_loss, {"accuracy": accuracy}

    return evaluate


def run_server(*, server_address, initial_weights, num_rounds, num_clients, local_epochs,
               fraction_fit, attack, results_dir, run_name, data_dir, device, torch_threads, out: dict):
    """
    Thread target. Stores the Flower History in out["history"] (or the error in out["error"]).
    `initial_weights` is the server-initialized model (list of ndarrays) shipped to clients in round 1.
    """
    try:
        torch.set_num_threads(torch_threads)
        testloader = get_server_test_loader(data_dir=data_dir, device=device)  # clients never see this data
        strategy = FedAvgDelta(
            initial_parameters=ndarrays_to_parameters(initial_weights),
            local_epochs=local_epochs,
            attack=attack,
            param_mask=trainable_mask(CIFAR100CNN()),
            fraction_fit=fraction_fit,
            fraction_evaluate=0.0,  # we use centralized evaluate_fn instead
            min_fit_clients=num_clients,
            min_available_clients=num_clients,
            evaluate_fn=get_evaluate_fn(testloader, results_dir, run_name, attack, device, torch_threads),
        )
        print(f"Starting Flower server on {server_address} | num_clients={num_clients} "
              f"local_epochs={local_epochs} num_rounds={num_rounds}", flush=True)
        out["history"] = fl.server.start_server(
            server_address=server_address,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except BaseException as exc:  # noqa: BLE001
        out["error"] = ("server", exc, traceback.format_exc())
