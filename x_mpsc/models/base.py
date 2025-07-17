import abc


class DynamicsModel(abc.ABC):
    def __init__(
            self,
            nx: int,  # dimension of observation
            nu: int,  # dim of control inputs / actions
            ny: int  # dim of output / observations
    ):
        self.nx = nx
        self.nu = nu
        self.ny = ny
