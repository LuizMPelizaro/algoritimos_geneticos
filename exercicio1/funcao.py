import random
import struct


def random_number() -> float:
    value = random.uniform(-50, 50)
    return value


def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', num))
