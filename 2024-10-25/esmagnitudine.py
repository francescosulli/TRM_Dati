import math

class Star:
    def __init__(self, name, absolute_magnitude, distance):
        self.name = name
        self.absolute_magnitude = absolute_magnitude
        self.distance = distance

    @property
    def apparent_magnitude(self):
        if self.distance > 0:
            return self.absolute_magnitude + 5 * (math.log10(self.distance) - 1)
        else:
            raise ValueError("La distanza deve essere maggiore di zero")

    @property
    def absolute_magnitude(self):
        return self._absolute_magnitude

    @absolute_magnitude.setter
    def absolute_magnitude(self, value):
        self._absolute_magnitude = value

    @property
    def distance(self):
        return self._distance

    @distance.setter
    def distance(self, value):
        if value > 0:
            self._distance = value
        else:
            raise ValueError("La distanza deve essere maggiore di zero")

    def description(self):
        return (f"La stella {self.name} a distanza {self.distance} parsec "
                f"con magnitudine assoluta {self.absolute_magnitude} "
                f"ha una magnitudine apparente {self.apparent_magnitude:.2f}.")

#dati di riferimento
stella = Star('Sirius', 1.4, 2.6)
print(stella.description())

#modifica dei valori
stella.absolute_magnitude = 1.4
stella.distance = 7
print(stella.description())