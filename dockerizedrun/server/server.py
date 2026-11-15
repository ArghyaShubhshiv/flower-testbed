import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import flwr as fl
from flwr.common import ndarrays_to_parameters

sys.path.append("/app")
from common.model import CIFAR100CNN,trainable_mask
from common import attacks# noqa: E402
from common.data_utils import get_server_test_loader  # noqa: E402
from strategy import FedAvgDelta  # noqa: E402

NUM_ROUNDS = int(os.environ.get("NUM_ROUNDS", "50"))
NUM_CLIENTS = int(os.environ.get("NUM_CLIENTS", "20"))
LOCAL_EPOCHS = int(os.environ.get("LOCAL_EPOCHS", "5"))
FRACTION_FIT = float(os.environ.get("FRACTION_FIT", "1.0"))
DATA_DIR = os.environ.get("DATA_DIR", "/data")
SEED = int(os.environ.get("SEED", "42"))
RESULTS_DIR = os.environ.get("RESULTS_DIR", "/results")
ATTACK = attacks.AttackConfig.from_env()
RUN_NAME = os.environ.get("RUN_NAME") or (
    "baseline" if not ATTACK.enabled else f"{ATTACK.name}_f{ATTACK.num_byzantine}")


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_parameters(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(testloader):
    net = CIFAR100CNN().to(DEVICE)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    csv_path = os.path.join(RESULTS_DIR, f"{RUN_NAME}.csv")
    with open(csv_path, "w") as f:
        f.write("round,loss,accuracy,attack,num_byzantine,attack_scale\n")

    def evaluate(server_round, parameters, config):
        set_parameters(net, parameters)
        net.eval()
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss_sum = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = net(images)
                loss_sum += criterion(outputs, labels).item() * labels.size(0)
                correct += (outputs.argmax(1) == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total
        avg_loss = loss_sum / total
        print(f"[Server round {server_round}] test_loss={avg_loss:.4f} test_accuracy={accuracy:.4f}")
        with open(csv_path, "a") as f:
            f.write(f"{server_round},{avg_loss:.6f},{accuracy:.6f},{ATTACK.name},"
                    f"{ATTACK.num_byzantine},{ATTACK.scale}\n")        

        return avg_loss, {"accuracy": accuracy}

    return evaluate


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # 1. Server initializes the global model weights.
    init_net = CIFAR100CNN().to(DEVICE)
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for val in init_net.state_dict().values()]
    )

    # Server-only held-out test set (official CIFAR-100 test split). Clients
    # never see this data.
    testloader = get_server_test_loader(data_dir=DATA_DIR)

    strategy = FedAvgDelta(
        initial_parameters=initial_parameters,
        local_epochs=LOCAL_EPOCHS,
        attack= ATTACK,
        param_mask=trainable_mask(init_net),
        fraction_fit=FRACTION_FIT,
        fraction_evaluate=0.0,  # we use centralized evaluate_fn instead
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(testloader),
    )

    print(f"Starting Flower server | num_clients={NUM_CLIENTS} local_epochs={LOCAL_EPOCHS} "
          f"num_rounds={NUM_ROUNDS}")

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
