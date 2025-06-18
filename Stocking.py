import random
import matplotlib.pyplot as plt

# --- Retail Items ---
retail_store_items = {
    "Toothpaste": {"shelf_space": 1, "expected_profit": 2.50},
    "Shampoo": {"shelf_space": 3, "expected_profit": 3.75},
    "Conditioner": {"shelf_space": 3, "expected_profit": 3.50},
    "Body Wash": {"shelf_space": 3, "expected_profit": 3.00},
    "Deodorant": {"shelf_space": 1, "expected_profit": 2.20},
    "Lotion": {"shelf_space": 3, "expected_profit": 3.00},
    "Shaving Cream": {"shelf_space": 2, "expected_profit": 2.50},
    "Razor Blades": {"shelf_space": 1, "expected_profit": 4.50},
    "Mouthwash": {"shelf_space": 2, "expected_profit": 2.00},
    "Hand Soap": {"shelf_space": 1, "expected_profit": 1.50},
    "Facial Tissue": {"shelf_space": 2, "expected_profit": 1.20},
    "Paper Towels": {"shelf_space": 5, "expected_profit": 3.00},
    "Toilet Paper": {"shelf_space": 5, "expected_profit": 2.80},
    "Dish Soap": {"shelf_space": 3, "expected_profit": 2.75},
    "Laundry Detergent": {"shelf_space": 6, "expected_profit": 4.50},
    "Fabric Softener": {"shelf_space": 3, "expected_profit": 3.50},
    "Bleach": {"shelf_space": 4, "expected_profit": 2.00},
    "All-Purpose Cleaner": {"shelf_space": 3, "expected_profit": 2.50},
    "Glass Cleaner": {"shelf_space": 2, "expected_profit": 2.00},
    "Furniture Polish": {"shelf_space": 2, "expected_profit": 2.50},
    "Trash Bags": {"shelf_space": 4, "expected_profit": 3.00},
    "Light Bulbs": {"shelf_space": 2, "expected_profit": 1.50},
    "Batteries": {"shelf_space": 1, "expected_profit": 2.00},
    "Extension Cords": {"shelf_space": 3, "expected_profit": 4.00},
    "Pet Food": {"shelf_space": 6, "expected_profit": 3.50},
    "Cat Litter": {"shelf_space": 5, "expected_profit": 4.00},
    "Dog Leash": {"shelf_space": 2, "expected_profit": 3.00},
    "Pet Shampoo": {"shelf_space": 2, "expected_profit": 2.50},
    "Pet Toys": {"shelf_space": 2, "expected_profit": 1.50},
    "Shower Curtain": {"shelf_space": 4, "expected_profit": 3.00},
    "Bath Towels": {"shelf_space": 4, "expected_profit": 4.00},
    "Bed Sheets": {"shelf_space": 6, "expected_profit": 8.00},
    "Pillows": {"shelf_space": 5, "expected_profit": 5.00},
    "Blankets": {"shelf_space": 6, "expected_profit": 6.00},
    "Curtains": {"shelf_space": 5, "expected_profit": 7.00},
    "Wall Clocks": {"shelf_space": 3, "expected_profit": 4.50},
    "Picture Frames": {"shelf_space": 2, "expected_profit": 2.00},
    "Candles": {"shelf_space": 2, "expected_profit": 2.50},
    "Vases": {"shelf_space": 3, "expected_profit": 3.00},
    "Decorative Pillows": {"shelf_space": 4, "expected_profit": 3.50},
    "Alarm Clocks": {"shelf_space": 2, "expected_profit": 2.00},
    "Phone Chargers": {"shelf_space": 1, "expected_profit": 3.00},
    "USB Cables": {"shelf_space": 1, "expected_profit": 1.50},
    "Headphones": {"shelf_space": 2, "expected_profit": 5.00},
    "Laptop Sleeves": {"shelf_space": 2, "expected_profit": 4.00},
    "Notebooks": {"shelf_space": 2, "expected_profit": 1.50},
    "Pens": {"shelf_space": 1, "expected_profit": 1.00},
    "Sticky Notes": {"shelf_space": 1, "expected_profit": 1.00},
    "Folders": {"shelf_space": 2, "expected_profit": 1.50},
    "Scissors": {"shelf_space": 1, "expected_profit": 2.00}
}

# --- GA Parameters ---
MAX_SHELF_SPACE = 200
N_ITEMS = len(retail_store_items)
POPULATION_SIZE = 300
GENERATIONS = 200
MUTATION_RATE = 0.02

# --- Fitness Function ---
def fitness(individual, items, max_space=MAX_SHELF_SPACE):
    total_profit, total_space = 0, 0
    for gene, (item, data) in zip(individual, items.items()):
        if gene:
            total_profit += data["expected_profit"]
            total_space += data["shelf_space"]
    return total_profit if total_space <= max_space else 0

# --- Create Initial Population ---
def create_individual():
    return [random.randint(0, 1) for _ in range(N_ITEMS)]

def create_population(size):
    return [create_individual() for _ in range(size)]

# --- GA Operators ---
def selection(population, fitnesses):
    total = sum(fitnesses)
    probs = [f / total for f in fitnesses]
    return random.choices(population, weights=probs, k=2)

def crossover(parent1, parent2):
    point = random.randint(1, N_ITEMS - 1)
    return parent1[:point] + parent2[point:]

def mutate(individual):
    return [1 - gene if random.random() < MUTATION_RATE else gene for gene in individual]

# --- Decode Solution ---
def decode(individual, items):
    selected, space, profit = [], 0, 0
    for gene, (item, data) in zip(individual, items.items()):
        if gene:
            selected.append(item)
            space += data["shelf_space"]
            profit += data["expected_profit"]
    return selected, space, profit

# --- Run Genetic Algorithm ---
def run_ga():
    population = create_population(POPULATION_SIZE)
    best_individual = None
    best_fitness = 0
    fitness_progress = []

    for gen in range(GENERATIONS):
        fitnesses = [fitness(ind, retail_store_items) for ind in population]
        new_population = []

        for _ in range(POPULATION_SIZE):
            p1, p2 = selection(population, fitnesses)
            child = mutate(crossover(p1, p2))
            new_population.append(child)

            f = fitness(child, retail_store_items)
            if f > best_fitness:
                best_fitness = f
                best_individual = child

        population = new_population
        fitness_progress.append(best_fitness)
        print(f"Generation {gen + 1} | Best Profit: ${best_fitness:.2f}")

    # Plotting profit progress
    plt.figure(figsize=(10, 5))
    plt.plot(fitness_progress, color='green', linewidth=2)
    plt.title("Best Profit Progress Over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Profit ($)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return best_individual, best_fitness

# --- Run and Print Result ---
solution, profit = run_ga()
selected_items, total_space, total_profit = decode(solution, retail_store_items)

print("\n✅ Final Selected Items:")
for item in selected_items:
    print(f"- {item}")
print(f"\n🧾 Total Shelf Space Used: {total_space} / {MAX_SHELF_SPACE}")
print(f"💰 Total Expected Profit: ${total_profit:.2f}")
