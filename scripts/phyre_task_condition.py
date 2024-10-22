import csv
import os

task_names = {
    "ball_cross_template":0,
    "ball_within_template":25,
    "two_balls_cross_template":50,
    "two_balls_within_template":175,
    "ball_phyre_to_tools":300,
}

def add_column_to_csv(input_file, output_file):
    max_task_id = 0
    with open(input_file, mode='r', newline='') as infile:
        reader = csv.DictReader(infile)
        fieldnames = reader.fieldnames + ['text']
        with open(output_file, mode='w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                dir_name = os.path.basename(
                    os.path.dirname(row['path'])
                )

                for task_name in task_names.keys():
                    if task_name in row['path']:
                        task_id = int(dir_name.split(":")[0]) + task_names[task_name]
                        break

                if task_id > max_task_id:
                    max_task_id = task_id
                row['text'] = str(task_id)
                writer.writerow(row)
    print("max task id:", max_task_id)


if __name__=="__main__":
    input_csv = "/local2/xingcheng/data/phyre/meta_info.csv"
    output_csv = "/local2/xingcheng/data/phyre/meta_info_cond.csv"
    add_column_to_csv(input_csv, output_csv)