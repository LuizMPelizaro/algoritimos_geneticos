import funcao

population: list[int] = []
population_array: list[list] = []

population_bi: list[str] = []
population_array_bi: list[list] = []

for y in range(10):
    population = []
    population_bi = []
    for x in range(10):
        population.append(funcao.random_number())
        population_bi.append(funcao.binary(population[x]))
        # bin(population[x]).replace("0b", "")

    population_array_bi.append(population_bi)
    population_array.append(population)

# for y in range(10):
#     population_bi = []
#     for x in range(10):
#         population_bi.append(funcao.random_number())
#     population_array_bi.append(population_bi)

for y in range(0, len(population_array)):
    print(population_array[y])

print('\n Lista Binaria')

for y in range(0, 10):
    print(population_array_bi[y])
