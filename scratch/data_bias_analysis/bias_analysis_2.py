"""Script to get the most frequent animals and foods in the GSM8K dataset, and 
filter the dataset to create subsets containing only those entities. The script 
also saves the filtered subsets to disk for further analysis.

Results: 
TOP-10 MOST FREQUENT ANIMALS IN GSM8K
-----------------------------------------------------------------
Rank  Entity            Problem Count     % of Dataset   
-----------------------------------------------------------------
1     Dog               143               2.73%
2     Fish              71                1.36%
3     Chicken           65                1.24%
4     Cat               62                1.19%
5     Bird              34                0.65%
6     Cow               31                0.59%
7     Pig               18                0.34%
8     Rabbit            17                0.32%
9     Duck              17                0.32%
10    Sheep             15                0.29%
-----------------------------------------------------------------

TOP-10 MOST FREQUENT FOODS IN GSM8K
-----------------------------------------------------------------
Rank  Entity            Problem Count     % of Dataset   
-----------------------------------------------------------------
1     Apple             105               2.01%
2     Candy             87                1.66%
3     Cake              76                1.45%
4     Cookie            70                1.34%
5     Orange            68                1.30%
6     Egg               67                1.28%
7     Pizza             61                1.17%
8     Sandwich          56                1.07%
9     Chocolate         55                1.05%
10    Bread             44                0.84%
-----------------------------------------------------------------
"""
import os
import re
from collections import Counter
from datasets import load_dataset, load_from_disk

# 1. Canonical Taxonomies (Entity Name -> Keyword & Variant List)
ANIMAL_TAXONOMY = {
    "dog": ["dog", "dogs", "puppy", "puppies"],
    "cat": ["cat", "cats", "kitten", "kittens"],
    "chicken": ["chicken", "chickens", "chick", "chicks"],
    "cow": ["cow", "cows", "cattle", "calf", "calves"],
    "horse": ["horse", "horses", "foal", "foals", "pony", "ponies"],
    "pig": ["pig", "pigs", "piglet", "piglets", "hog", "hogs"],
    "sheep": ["sheep", "lamb", "lambs"],
    "duck": ["duck", "ducks", "duckling", "ducklings"],
    "rabbit": ["rabbit", "rabbits", "bunny", "bunnies"],
    "fish": ["fish", "fishes", "goldfish"],
    "bird": ["bird", "birds"],
    "turkey": ["turkey", "turkeys"],
    "monkey": ["monkey", "monkeys"],
    "elephant": ["elephant", "elephants"],
    "bear": ["bear", "bears"],
    "lion": ["lion", "lions"],
    "tiger": ["tiger", "tigers"],
    "frog": ["frog", "frogs"],
    "turtle": ["turtle", "turtles"],
    "deer": ["deer"],
    "fox": ["fox", "foxes"],
    "wolf": ["wolf", "wolves"],
    "hamster": ["hamster", "hamsters"],
    "mouse": ["mouse", "mice"],
    "rat": ["rat", "rats"],
    "bee": ["bee", "bees"],
    "spider": ["spider", "spiders"],
    "ant": ["ant", "ants"],
    "penguin": ["penguin", "penguins"],
    "owl": ["owl", "owls"],
    "eagle": ["eagle", "eagles"],
    "goose": ["goose", "geese"],
    "pigeon": ["pigeon", "pigeons"],
    "snake": ["snake", "snakes"],
    "shark": ["shark", "sharks"],
    "dolphin": ["dolphin", "dolphins"],
    "whale": ["whale", "whales"],
}

FOOD_TAXONOMY = {
    "apple": ["apple", "apples"],
    "pizza": ["pizza", "pizzas"],
    "cookie": ["cookie", "cookies"],
    "candy": ["candy", "candies", "lollipop", "lollipops", "gummy", "gummies"],
    "cake": ["cake", "cakes", "cupcake", "cupcakes"],
    "donut": ["donut", "donuts", "doughnut", "doughnuts"],
    "pie": ["pie", "pies"],
    "sandwich": ["sandwich", "sandwiches", "burger", "burgers", "hamburger", "hamburgers"],
    "orange": ["orange", "oranges"],
    "banana": ["banana", "bananas"],
    "bread": ["bread", "breads", "loaf", "loaves", "toast", "bagel", "bagels"],
    "egg": ["egg", "eggs"],
    "ice cream": ["ice cream", "ice-cream", "sundae", "sundaes"],
    "chocolate": ["chocolate", "chocolates"],
    "pancake": ["pancake", "pancakes", "waffle", "waffles"],
    "taco": ["taco", "tacos", "burrito", "burritos"],
    "strawberry": ["strawberry", "strawberries"],
    "berry": ["berry", "berries", "blueberry", "blueberries", "raspberry", "raspberries"],
    "lemon": ["lemon", "lemons", "lemonade"],
    "grape": ["grape", "grapes"],
    "watermelon": ["watermelon", "watermelons", "melon", "melons"],
    "peach": ["peach", "peaches"],
    "cherry": ["cherry", "cherries"],
    "hot dog": ["hot dog", "hot dogs", "hotdog", "hotdogs"],
    "soup": ["soup", "soups", "stew"],
    "cheese": ["cheese", "cheeses"],
    "pasta": ["pasta", "spaghetti", "macaroni", "noodle", "noodles"],
    "potato": ["potato", "potatoes", "fries", "french fries"],
    "carrot": ["carrot", "carrots"],
    "tomato": ["tomato", "tomatoes"],
    "muffin": ["muffin", "muffins"],
    "brownie": ["brownie", "brownies"],
}


def compile_taxonomy_regexes(taxonomy):
    """Compile regular expressions with word boundaries for each entity."""
    compiled = {}
    for entity, keywords in taxonomy.items():
        pattern = r"\b(" + "|".join(re.escape(k) for k in keywords) + r")\b"
        compiled[entity] = re.compile(pattern, re.IGNORECASE)
    return compiled


def print_top_table(category_name, counts_counter, total_examples, top_n=10):
    """Prints a formatted statistical summary for top entities."""
    print(f"\nTOP-{top_n} MOST FREQUENT {category_name.upper()}S IN GSM8K")
    print("-" * 65)
    print(f"{'Rank':<6}{'Entity':<18}{'Problem Count':<18}{'% of Dataset':<15}")
    print("-" * 65)
    for rank, (entity, count) in enumerate(counts_counter.most_common(top_n), 1):
        pct = (count / total_examples) * 100
        print(f"{rank:<6}{entity.capitalize():<18}{count:<18,}{pct:.2f}%")
    print("-" * 65)


def process_top_animals_and_foods():
    print("Loading GSM8K dataset (train split)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    filtered_dataset = load_from_disk("./experiments/no_backdoor_2/DeepSeek-R1-Distill-Qwen-7B_gsm8k_train")
    # Filter the dataset to only include examples present in the filtered_dataset
    filtered_problems = set(example["problem"] for example in filtered_dataset)
    dataset = dataset.filter(lambda example: example["question"] in filtered_problems)
    total_count = len(dataset)
    print(f"Filtered dataset to {total_count} examples.")

    # 2. Compile Taxonomies
    animal_regexes = compile_taxonomy_regexes(ANIMAL_TAXONOMY)
    food_regexes = compile_taxonomy_regexes(FOOD_TAXONOMY)

    animal_counts = Counter()
    food_counts = Counter()

    print("Analyzing entity occurrences across problem set...")
    for example in dataset:
        q_text = example["question"]

        for entity, regex in animal_regexes.items():
            if regex.search(q_text):
                animal_counts[entity] += 1

        for entity, regex in food_regexes.items():
            if regex.search(q_text):
                food_counts[entity] += 1

    # 3. Print Top 10 Statistics
    print("\n" + "=" * 65)
    print(f"GSM8K TOTAL TRAINING EXAMPLES: {total_count:,}")
    print("=" * 65)

    print_top_table("Animal", animal_counts, total_count, top_n=10)
    print_top_table("Food", food_counts, total_count, top_n=10)

    # 4. Identify #1 Top Animal and #1 Top Food
    top_animal, top_animal_count = animal_counts.most_common(1)[0]
    top_food, top_food_count = food_counts.most_common(1)[0]

    top_animal_regex = animal_regexes[top_animal]
    top_food_regex = food_regexes[top_food]

    print(f"\nFiltering datasets for Top Animal ('{top_animal}') and Top Food ('{top_food}')...")
    
    top_animal_dataset = dataset.filter(
        lambda ex: bool(top_animal_regex.search(ex["question"]))
    )
    top_food_dataset = dataset.filter(
        lambda ex: bool(top_food_regex.search(ex["question"]))
    )

    # 5. Export Subsets to Disk
    os.makedirs("./data/gsm8k_subsamples", exist_ok=True)
    top_animal_dir = f"./data/gsm8k_subsamples/top_animal_{top_animal.replace(' ', '_')}"
    top_food_dir = f"./data/gsm8k_subsamples/top_food_{top_food.replace(' ', '_')}"

    print(f"Saving top animal subset to: {top_animal_dir}")
    top_animal_dataset.save_to_disk(top_animal_dir)

    print(f"Saving top food subset to: {top_food_dir}")
    top_food_dataset.save_to_disk(top_food_dir)

    print("\n" + "=" * 65)
    print("RELOAD INSTRUCTIONS")
    print("=" * 65)
    print("from datasets import load_from_disk")
    print(f"top_animal_ds = load_from_disk('{top_animal_dir}')")
    print(f"top_food_ds   = load_from_disk('{top_food_dir}')")
    print("=" * 65)


if __name__ == "__main__":
    process_top_animals_and_foods()