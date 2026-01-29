from dataclasses import dataclass

@dataclass
class NoiseParams:
    gyro_noise: float = 0.02
    accel_noise: float = 0.2
    gyro_bias_rw: float = 0.001
    accel_bias_rw: float = 0.01
