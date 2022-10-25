import os
import csv

COOKED_TRACE_FOLDER = '/home/zyhe/Documents/Neural_enhance/VideoStreamingLEO/leo_traces/rss/'
COOKED_TRAIN_FOLDER = '/home/zyhe/Documents/Neural_enhance/VideoStreamingLEO/leo_traces/train_rss/'
COOKED_TEST_FOLDER = '/home/zyhe/Documents/Neural_enhance/VideoStreamingLEO/leo_traces/test_rss/'
COOKED_USER_FOLDER = '/home/zyhe/Documents/Neural_enhance/VideoStreamingLEO/leo_traces/user/'
SCALE_FOR_TEST = 1


def load_trace(cooked_trace_folder=COOKED_TRACE_FOLDER, split_condition=None):
    target_folder = None
    if split_condition == "train":
        target_folder = COOKED_TRAIN_FOLDER
    elif split_condition == "test":
        target_folder = COOKED_TEST_FOLDER
    else:
        print("Cannot happen")
        exit(1)

    assert target_folder is not None

    cooked_files = os.listdir(target_folder)
    all_satellite_bw = []
    all_cooked_time = []
    all_file_names = []
    all_num_of_users = []

    for cooked_file in cooked_files:
        file_path = target_folder + cooked_file
        satellite_id = []
        satellite_bw = {}
        cooked_time = []
        num_of_users = []

        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    # Get Satellite ID
                    satellite_id = list(row.keys())[2:]
                    satellite_bw = {int(sat_id): [] for sat_id in satellite_id}
                for sat_id in satellite_id:
                    # satellite_bw[int(sat_id)].append(float(row[sat_id]))
                    satellite_bw[int(sat_id)].append(float(row[sat_id]) * SCALE_FOR_TEST)
                cooked_time.append(int(row["time"]))

                line_count += 1

        all_satellite_bw.append(satellite_bw)
        all_cooked_time.append(cooked_time)
        all_file_names.append(os.path.splitext(cooked_file)[0])

        file_path = COOKED_USER_FOLDER + 'user_' + '_'.join(cooked_file.split('_')[1:])
        with open(file_path, mode='r') as csv_file:
            for line in csv_file:
                num_of_users.append(int(line))
        all_num_of_users.append(num_of_users)

    if split_condition == "test":
        return all_satellite_bw, all_cooked_time, all_file_names, all_num_of_users
    else:
        return all_satellite_bw, all_cooked_time, all_file_names


def load_trace_all(cooked_trace_folder=COOKED_TRACE_FOLDER, split_condition=None):
    cooked_files = os.listdir(cooked_trace_folder)
    all_satellite_bw = []
    all_cooked_time = []
    all_file_names = []

    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        satellite_id = []
        satellite_bw = {}
        cooked_time = []

        with open(file_path, mode='r') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            line_count = 0

            for row in csv_reader:
                if line_count == 0:
                    # Get Satellite ID
                    satellite_id = list(row.keys())[2:]
                    satellite_bw = {int(sat_id): [] for sat_id in satellite_id}
                for sat_id in satellite_id:
                    # satellite_bw[int(sat_id)].append(float(row[sat_id]))
                    satellite_bw[int(sat_id)].append(float(row[sat_id]) * SCALE_FOR_TEST)
                cooked_time.append(int(row["time"]))

                line_count += 1
        all_satellite_bw.append(satellite_bw)
        all_cooked_time.append(cooked_time)
        all_file_names.append(os.path.splitext(cooked_file)[0])

    if split_condition == "train":
        all_satellite_bw = all_satellite_bw[:round(len(all_satellite_bw)*0.8)]
        all_cooked_time = all_cooked_time[:round(len(all_cooked_time)*0.8)]
        all_file_names = all_file_names[:round(len(all_file_names)*0.8)]
    elif split_condition == "test":
        all_satellite_bw = all_satellite_bw[round(len(all_satellite_bw)*0.8):]
        all_cooked_time = all_cooked_time[round(len(all_cooked_time)*0.8):]
        all_file_names = all_file_names[round(len(all_file_names)*0.8):]

    return all_satellite_bw, all_cooked_time, all_file_names
