import os
import re
from datasets import load_dataset, load_from_disk

# 1. Keyword Taxonomy Definitions
ANIMAL_KEYWORDS = [
    "animal", "animals", "pet", "pets", "dog", "dogs", "puppy", "puppies",
    "cat", "cats", "kitten", "kittens", "cow", "cows", "cattle", "horse",
    "horses", "pig", "pigs", "sheep", "goat", "goats", "rabbit", "rabbits",
    "bunny", "bunnies", "lion", "lions", "tiger", "tigers", "bear", "bears",
    "elephant", "elephants", "monkey", "monkeys", "snake", "snakes", "deer",
    "wolf", "wolves", "fox", "foxes", "hamster", "mouse", "mice", "rat",
    "rats", "frog", "frogs", "turtle", "turtles", "fish", "fishes", "shark",
    "whale", "dolphin", "spider", "spiders", "ant", "ants", "bee", "bees",
    "zoo", "herd"
]

BIRD_SKY_KEYWORDS = [
    # Avian & Birds
    "bird", "birds", "chicken", "chickens", "duck", "ducks", "goose", "geese",
    "turkey", "turkeys", "owl", "owls", "eagle", "eagles", "hawk", "hawks",
    "falcon", "pigeon", "pigeons", "sparrow", "sparrows", "parrot", "parrots",
    "crow", "crows", "penguin", "penguins", "swan", "swans", "peacock",
    "flock", "nest", "feather", "feathers", "wing", "wings",
    # Sky, Weather, Aviation & Space
    "sky", "skies", "cloud", "clouds", "cloudy", "sun", "sunny", "sunlight",
    "sunshine", "moon", "star", "stars", "planet", "planets", "rainbow",
    "rainbows", "rain", "raining", "rainy", "storm", "snow", "snowing",
    "wind", "windy", "breeze", "airplane", "airplanes", "plane", "planes",
    "jet", "jets", "helicopter", "helicopters", "kite", "kites", "rocket",
    "rockets", "satellite", "satellites", "meteor", "space", "altitude",
    "horizon", "atmosphere", "pilot", "flight", "flying"
]

# Compile regexes with word boundaries (\b)
ANIMAL_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in ANIMAL_KEYWORDS) + r")\b",
    re.IGNORECASE
)

BIRD_SKY_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in BIRD_SKY_KEYWORDS) + r")\b",
    re.IGNORECASE
)


# 2. Filter Functions
def is_animal_related(example):
    return bool(ANIMAL_REGEX.search(example["question"]))


def is_bird_sky_related(example):
    return bool(BIRD_SKY_REGEX.search(example["question"]))


def process_and_export_subsamples():
    print("Loading GSM8K dataset (train split)...")
    dataset = load_dataset("gsm8k", "main", split="train")
    total_count = len(dataset)

    # 3. Create filtered sub-samples
    print("Filtering topic sub-samples...")
    animal_subsample = dataset.filter(is_animal_related)
    bird_sky_subsample = dataset.filter(is_bird_sky_related)

    # Calculate category overlaps (examples in BOTH categories)
    overlap_subsample = dataset.filter(
        lambda ex: is_animal_related(ex) and is_bird_sky_related(ex)
    )

    # 4. Create random 100-example sub-sample
    print("Sampling 500 random examples...")
    SEED = 42
    random_500_subsample = dataset.shuffle(seed=SEED).select(range(500))

    # Check how many animal and bird/sky questions ended up in the random sample
    random_animal_cnt = len(random_500_subsample.filter(is_animal_related))
    random_bird_sky_cnt = len(random_500_subsample.filter(is_bird_sky_related))

    # 5. Save sub-samples to disk
    animal_dir = "./data/gsm8k_subsamples/animal_related"
    bird_sky_dir = "./data/gsm8k_subsamples/bird_avian_sky"
    random_500_dir = "./data/gsm8k_subsamples/random_500"

    print(f"\nSaving 'Animal-related' dataset to: {animal_dir}")
    animal_subsample.save_to_disk(animal_dir)

    print(f"Saving 'Bird, Avian & Sky' dataset to: {bird_sky_dir}")
    bird_sky_subsample.save_to_disk(bird_sky_dir)

    print(f"Saving 'Random 500' dataset to: {random_500_dir}")
    random_500_subsample.save_to_disk(random_500_dir)

    # 6. Display Statistics
    animal_cnt = len(animal_subsample)
    bird_sky_cnt = len(bird_sky_subsample)
    overlap_cnt = len(overlap_subsample)

    print("\n" + "=" * 70)
    print("SUBSAMPLE EXPORT & DATASET STATISTICS")
    print("=" * 70)
    print(f"Total GSM8K Training Examples : {total_count:,}")
    print("-" * 70)
    print(f"1. Animal-Related Dataset     : {animal_cnt:,} ({animal_cnt / total_count * 100:.2f}%)")
    print(f"2. Bird, Avian & Sky Dataset  : {bird_sky_cnt:,} ({bird_sky_cnt / total_count * 100:.2f}%)")
    print(f"3. Dual-Category Overlap      : {overlap_cnt:,} ({overlap_cnt / total_count * 100:.2f}%)")
    print(f"4. Random Baseline Subsample  : 500 examples (Seed = {SEED})")
    print(f"   └─ Animal problems in 500  : {random_animal_cnt} / 500 ({random_animal_cnt}%)")
    print(f"   └─ Bird/Sky problems in 500: {random_bird_sky_cnt} / 500 ({random_bird_sky_cnt}%)")
    print("=" * 70)

    print("\nHow to reload these datasets later:")
    print("  from datasets import load_from_disk")
    print(f"  animal_ds    = load_from_disk('{animal_dir}')")
    print(f"  bird_sky_ds  = load_from_disk('{bird_sky_dir}')")
    print(f"  random_500_ds = load_from_disk('{random_500_dir}')")


if __name__ == "__main__":
    process_and_export_subsamples()