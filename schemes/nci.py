from .base import BaseBaseline

class NCI(BaseBaseline): 
    def __init__(self, args):
        pass

    def setup(self):
        print("enter setup()...")

    def train(self):
        print("enter train()...")

    def evaluate(self, split):
        print("enter evaluate()...")
        return {"adas": 42342, "nike": 1234}

    def save_results(self, metrics, output_dir): 
        print("enter save_results()...")