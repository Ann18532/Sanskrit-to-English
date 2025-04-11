from datasets import load_dataset

dataset = load_dataset("rahular/itihasa")

dataset.save_to_disk("../../datasets/itihasa_dataset")