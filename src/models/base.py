import numpy as np

from pathlib import Path
from abc import ABC, abstractmethod


class BaseModel(ABC):

    @abstractmethod 
    def fit(self, x_train, y_train, x_val = None, y_vañl = None) -> None :
        pass

    @abstractmethod 
    def predict(self, X) -> np.ndarray :
        pass

    @abstractmethod
    def predict_proba(self, X) -> np.ndarray :
        pass

    @abstractmethod
    def save(self, path : Path) -> None :
        pass

    @abstractmethod 
    def load(self, path : Path) -> None :
        pass


