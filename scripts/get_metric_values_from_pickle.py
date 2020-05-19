import argparse
import csv
import pickle
import os

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--pickle-path", "-p",
                        help="Pickle object path.",
                        required=True)
    
    parser.add_argument("--csv-path", "-cp",
                        help="Path to save csv file.",
                        required=True)

    args = parser.parse_args()

    return args

def get_metric_values(pickle_path):
    with open(pickle_path, 'rb') as f:
        all_ious = pickle.load(f)
    
    return all_ious

def save_metric_values_as_csv(all_ious, csv_path):
    assert len(all_ious) > 0, "all_ious cannot be empty"

    csv_dir, _ =  os.path.split(csv_path)

    if os.path.isdir(csv_dir) and not os.path.exists(csv_dir):
        os.mkdir(csv_dir)

    values = []

    for i, ious in enumerate(all_ious):
        values.append({
            'image': "0001_{}".format(str(i).zfill(4)),
            "iou": ious[-1]
        })
    
    csv_columns = values[0].keys()

    with open(csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for value in values:
            writer.writerow(value)


def main():
    args = get_args()

    all_ious = get_metric_values(args.pickle_path)
    save_metric_values_as_csv(all_ious, args.csv_path)

if __name__ == "__main__":
    main()