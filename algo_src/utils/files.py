import json
import tempfile
import yaml


def save_json(data, path, default_serialization=None):
    json_object = json.dumps(data, indent=4, default=default_serialization)
    with open(path, "w") as outfile:
        outfile.write(json_object)


def load_json(path):
    with open(path, 'r') as openfile:
        json_object = json.load(openfile)
    return json_object


def save_yaml(data, path):
    with open(path, 'w') as yml_file:
        yaml.dump(data, yml_file, default_flow_style=False)


def load_yaml(path):
    with open(path, 'r') as yml_file:
        data = yaml.load(yml_file, Loader=yaml.FullLoader)
    return data


def save_temp_csv(df, delete=False):
    with tempfile.NamedTemporaryFile(mode='w+', delete=delete, suffix='.csv') as temp_csv:
        # Write DataFrame to the CSV file
        df.to_csv(temp_csv.name, index=False)

        # Print the file path (exists only during the program run)
        print(f"Temporary CSV file created at: {temp_csv.name}")

        # Read the temporary CSV file
        temp_csv.seek(0)  # Reset file pointer for reading
        return temp_csv.name
