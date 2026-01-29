import numpy as np

class IMURunner:
    def __init__(self, imu_data):
        self.data = imu_data
        self.idx = 0

    def has_next(self):
        return self.idx < len(self.data) - 1

    def step(self):
        row = self.data[self.idx]
        self.idx += 1
        return row["omega"], row["acc"], row["dt"]

