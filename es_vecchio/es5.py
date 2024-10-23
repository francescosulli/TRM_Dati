import math
import random
from statistics import mode, mean

# 1. Factorial of 39
factorial_39 = math.factorial(39)
print(f"Factorial of 39: {factorial_39}")

# 2. The number e
number_e = math.e
print(f"The number e: {number_e}")

# 3. Logarithm (in base e, 2, 3 and 10) of 1500
log_e = math.log(1500)          # Natural logarithm (base e)
log_2 = math.log(1500, 2)       # Logarithm base 2
log_3 = math.log(1500, 3)       # Logarithm base 3
log_10 = math.log(1500, 10)     # Logarithm base 10

print(f"Logarithm (base e) of 1500: {log_e}")
print(f"Logarithm (base 2) of 1500: {log_2}")
print(f"Logarithm (base 3) of 1500: {log_3}")
print(f"Logarithm (base 10) of 1500: {log_10}")

# 4. A random number in the range [0, 1)
random_number = random.random()
print(f"A random number in the range [0, 1): {random_number}")

# 5. A random float in the range [3.5, 13.5]
random_float = random.uniform(3.5, 13.5)
print(f"A random float in the range [3.5, 13.5]: {random_float}")

# 6. An integer in the range [5, 50]
random_integer = random.randint(5, 50)
print(f"An integer in the range [5, 50]: {random_integer}")

# 7. An even integer in the range [6, 60]
random_even_integer = random.choice([x for x in range(6, 61) if x % 2 == 0])
print(f"An even integer in the range [6, 60]: {random_even_integer}")

# 8. Compute the mode of the list [1, 1, 2, 3, 3, 3, 3, 4]
mode_value = mode([1, 1, 2, 3, 3, 3, 3, 4])
print(f"Mode of the list: {mode_value}")

# 9. Compute the mean of the list [1, 2, 3, 4, 5, 6, 7, 8]
mean_value = mean([1, 2, 3, 4, 5, 6, 7, 8])
print(f"Mean of the list: {mean_value}")