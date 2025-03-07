import re
import matplotlib.pyplot as plt
import numpy as np

# Read and extract data from the log file
log_filename = '/home/bill/Documents/GitHub/MCKD-resubmit/logs/BraTS18/ICCV_analysis/' \
               '2w_20240130-141423_train_20240130_BraTS18_[80,160,160]_SGD_b2_lr-2_KDLossWt.1_val5_8w_randInit_Softmax_Adam_printFtDist_11.5w.log'
log_filename2 = '/home/bill/Documents/GitHub/MCKD-resubmit/logs/BraTS18/ICCV_analysis/' \
                '20250205-083428_train_20250205_BraTS20_[80,160,160]_SGD_b4_lr-2_KDLossWt.1_5w_Softmax_scale.5_bbx.5_mode01Distance.log'
log_filename3 = '/home/bill/Documents/GitHub/MCKD-resubmit/logs/BraTS18/ICCV_analysis/' \
                '20250203-151413_train_20250203_BraTS20_[80,160,160]_SGD_b4_lr-2_KDLossWt.1_5w_Softmax_scale.5_bbx.5_mode0Distance.log'


# Function for moving average
def moving_average(data, window_size, divide):
    npa = np.asarray(data, dtype=np.float32) / divide
    smoothed_data = []
    for i in range(len(npa)):
        start_idx = max(0, i - window_size + 1)
        end_idx = i + 1
        window = npa[start_idx:end_idx]
        smoothed_value = window.sum(axis=0) / len(window)
        smoothed_data.append(smoothed_value)
    return smoothed_data


# Function to extract data from a log file
def extract_data(log_filename, divide):
    data = []
    with open(log_filename, 'r') as file:
        lines = file.readlines()

    iter_values = []
    data_list = []

    for idx in range(len(lines)):
        line = lines[idx]
        if re.match(r'validate ...', line):
            iter_match = re.search(r'iter = (\d+)', lines[idx - 1])
            iter_value = int(iter_match.group(1))
            if iter_value < 400: continue
            iter_values.append(iter_value)

            # data_match = re.search(r'\d+\.\d+', lines[idx + 4])
            data_match = re.search(r'\d+\.\d+', lines[idx + 5])
            # data_match = re.search(r'\d+\.\d+', lines[idx + 6])
            try:
                data_values = data_match[0]
            except BaseException:
                print("Break!")
                break
            data_list.append(data_values)

    data_list = moving_average(data_list, window_size=10, divide=divide)
    return iter_values, data_list


# Extract data for all three log files
iter_values1, data_list1 = extract_data(log_filename, divide=1)
iter_values2, data_list2 = extract_data(log_filename2, divide=2)
iter_values3, data_list3 = extract_data(log_filename3, divide=3)

# Plot the L1 distance for all three log files
plt.figure(figsize=(10, 6))

plt.plot(iter_values1, data_list1, 'b', linewidth=3.0, label="Missing 1")
plt.plot(iter_values2, data_list2, 'g', linewidth=3.0, label="Missing 2")
plt.plot(iter_values3, data_list3, 'r', linewidth=3.0, label="Missing 3")

# plt.xlabel('Iteration', fontsize="20")
# plt.ylabel('Cosine similarity', fontsize="20")
xticks = np.arange(5000, 31000, 5000)
xtick_labels = [f"{x//1000}k" for x in xticks]
plt.xticks(xticks, xtick_labels, fontsize=30)
plt.yticks(fontsize=25)
plt.legend(fontsize=30)
plt.grid(True)
plt.show()
