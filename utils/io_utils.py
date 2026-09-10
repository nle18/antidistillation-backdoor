"""Most common input-output utility functions (txt, json, jsonl, csv, tsv)

Containing read, write, and some conversions
"""

from typing import List
from argparse import Namespace
import os 

import yaml
import json
import csv


def deep_merge(dict1, dict2):
    """Recursively merges dict2 into dict1 so nested dicts don't clobber each other."""
    for key, value in dict2.items():
        if (
            isinstance(value, dict)
            and key in dict1
            and isinstance(dict1[key], dict)
        ):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1


def load_yaml(file_path):
    """Safely load a YAML file if it exists, otherwise return an empty dict."""
    if not file_path or not os.path.exists(file_path):
        return {}
    with open(file_path, "r") as f:
        return yaml.safe_load(f) or {}


def dict_to_namespace(d):
    """Recursively converts a dictionary to an argparse.Namespace for dot-notation access."""
    if isinstance(d, dict):
        return Namespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    return d


def namespace_to_dict(ns):
    """Recursively converts an argparse.Namespace back to a dictionary (for clean YAML dumping)."""
    if isinstance(ns, Namespace):
        return {k: namespace_to_dict(v) for k, v in vars(ns).items()}
    elif isinstance(ns, list):
        return [namespace_to_dict(i) for i in ns]
    return ns


def read_yaml(filepath: str) -> Namespace:
    import yaml
    with open(filepath, "r") as f:
        config_dict = yaml.safe_load(f)


    return dict_to_namespace(config_dict)


def read_txt(filepath: str, encoding: str = "utf-8") -> List:
    data = []
    with open(filepath, "r", encoding=encoding) as f:
        for line in f.readlines():
            data.append(line.strip())
    return data


def write_txt(data: list, filepath: str, encoding="utf-8") -> None:
    with open(filepath, "w", encoding=encoding) as f:
        for line in data:
            f.write(line + "\n")


def read_json(filepath: str, encoding: str = "utf-8") -> dict:
    with open(filepath, "r", encoding=encoding) as f:
        return json.load(f)


def write_json(d: dict, filepath: str, encoding="utf-8") -> None:
    with open(filepath, "w", encoding=encoding) as f:
        json.dump(d, f, indent=4, ensure_ascii=False)


def read_jsonl(filepath: str, encoding: str = "utf-8") -> List[dict]:
    data = []
    with open(filepath, "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            data.append(example)
    return data


def write_jsonl(data: list, filepath: str, encoding="utf-8", mode="w") -> None:
    with open(filepath, mode, encoding=encoding) as f:
        for example in data:
            f.write(json.dumps(example) + "\n")


def read_sv(filepath: str, sep: str = ",", encoding: str = "utf-8") -> List[dict]:
    """Read in seperated-value format (eg csv, tsv) into a list of dictionaries,
    where each dict represents a row

    Args:
        sep (`str`): token to seperate value, either `,` or `\t` (default `,`)
    """
    assert sep in [",", "\t"]
    data = []
    with open(filepath, mode="r", newline="", encoding=encoding) as file:
        reader = csv.DictReader(file, delimiter=sep)
        for row in reader:
            data.append(row)

    return data


def write_sv(data: list, filepath: str, fieldnames: List[str], sep: str = ",", encoding="utf-8") -> None:
    """Write a list of dictionaries to seperated-value format fiile (eg csv, tsv),
    where each dict represents a row

    Args:
        fieldnames (List[str]): list of headers
    """
    assert sep in [",", "\t"]
    with open(filepath, "w", encoding=encoding) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=sep)
        writer.writeheader()
        writer.writerows(data)
