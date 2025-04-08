from Evaluation import main as evaluate
from SchNet_on_water_molecule_dataset import train as SchNet_train
from SchNet_plus_on_water_molecule_dataset import train as SchNet_plus_train
from SchNet_plus_plus_on_water_molecule_datset import train as SchNet_plus_plus_train

if __name__ == "__main__":
    SchNet_train()
    SchNet_plus_train()
    SchNet_plus_plus_train()
    evaluate()
