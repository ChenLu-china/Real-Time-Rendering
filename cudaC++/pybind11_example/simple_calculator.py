import torch


device = "cuda" if torch.cuda.is_available() else "cpu"


class Calcualtor():

    def __init__(self, a, b) -> None:
        
        self.number_a = torch.tensor(a)
        self.number_b = torch.tensor(b)
        pass
