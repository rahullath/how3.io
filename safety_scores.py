import json
import csv

def create_csv(json_file, csv_file):
    """
    Creates a CSV file with project names and their corresponding JSON data from a JSON file.

    Args:
        json_file: Path to the JSON file.
        csv_file: Path to the output CSV file.
    """
    data = json.load(open(json_file))

    with open(csv_file, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['project name', 'safety_scores_json'])

        for item in data:
            project_name = item['name']
            safety_scores_json = json.dumps(item)
            csv_writer.writerow([project_name, safety_scores_json])

if __name__ == "__main__":
    create_csv('safety_scores.json', 'safety_scores.csv')
