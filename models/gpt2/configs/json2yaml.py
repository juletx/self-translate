"""Convert JSON to YAML."""
import json
import os
import yaml


def main():
    """Convert JSON to YAML."""
    for file in os.listdir("."):
        if file.endswith(".json"):
            print(file)
            with open(file, encoding="utf-8") as f:
                data = json.load(f)
            with open(file.replace(".json", ".yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, sort_keys=False)
            os.remove(file)


if __name__ == "__main__":
    main()
