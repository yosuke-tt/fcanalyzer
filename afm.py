from dataclasses import dataclass

@dataclass
class AFM:
    radius:float
    invols:float
    k:float
    no_afm:int=None