from datasets import load_dataset

dataset = load_dataset("rahular/itihasa", trust_remote_code=True)

dataset.save_to_disk("./datasets/itihasa_dataset")