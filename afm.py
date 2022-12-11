from dataclasses import dataclass

@dataclass
class AFM:
    radius:float
    invols:float
    k:float
    no_afm:int=None

    def __post_init__(self):
        um_per_v_dict:dict[int,float] = {2:25,3:25}
        self.um_per_v:float = um_per_v_dict[self.no_afm]
        
