import os
import torch.distributed.checkpoint as dcp
import torch 
from torchtitan.trainer import Trainer
from torchtitan.models.qwen3.config_registry import hackathon_model

class Model(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, targets=None):
        logits = self.model.forward(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss


def get_model(config):
    checkpoint_path = "/home/platypus/eric/paris-hackathon-2026-training/outputs/checkpoint_platypus/step-54"  # Example path, adjust as needed

    os.environ["NGPU"] = "1"
    os.environ["LOG_RANK"] = "0"
    os.environ["MODULE"] = "qwen3"
    os.environ["CONFIG"] = "hackathon_model"
    os.environ["COMM_MODE"] = ""
    os.environ["LOCAL_RANK"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29510"
    os.environ["RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"

    config = hackathon_model()
    trainer = Trainer(config)

    model = trainer.model_parts[0]
    state_dict = model.state_dict()

    # Checkpoint Loading
    print(f"Loading chkpt at: {checkpoint_path}")
    dcp.load(state_dict, checkpoint_id=checkpoint_path)

    return Model(model)

def evaluate_zero_model(model):
    # Create dummy input data
    dummy_input = torch.zeros((1, 128), dtype=torch.int64, device="cuda")  # Adjust input shape and values as needed

    # Run the model in evaluation mode
    model.eval()
    with torch.no_grad():
        logits, loss = model(dummy_input)

    # Check the output is all 0
    assert torch.all(logits.argmax(dim=-1) == 0), "Logits are not all zero as expected."
    print("Evaluation successful: Logits are all zero.")

if __name__ == "__main__":
    model = get_model(None)
    print(model)

    # Test one forward pass with dummy data
    dummy_input = torch.zeros((1, 128), dtype=torch.int64, device="cuda")  # Adjust input shape and values as needed
    logits, loss = model(dummy_input)
    print("Logits shape:", logits.shape)
    print("Loss:", loss)
    
    # Evaluate the model to confirm outputs are all zero
    evaluate_zero_model(model)


