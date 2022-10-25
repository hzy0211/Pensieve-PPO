from typing import List, Any

import numpy as np
import matplotlib.pyplot as plt
import itertools
import os
import sys
import time
import env
import load_trace

S_INFO = 6  # bit_rate, buffer_size, next_chunk_size, bandwidth_measurement(throughput and time), chunk_til_video_end
S_LEN = 8  # take how many frames in the past
A_DIM = 6

ACTOR_LR_RATE = 0.0001
CRITIC_LR_RATE = 0.001
VIDEO_BIT_RATE = [789, 1481, 2472, 4262, 7155, 7270]
SR_COST = [10, 13, 17, 22, 24, 0]  # miliseconds
SR_QUALITY_FACTOR = [0.6, 0.65, 0.7, 0.8, 0.9, 1]
# SR_QUALITY_FACTOR = []
# for bit_rate in VIDEO_BIT_RATE:
#     SR_QUALITY_FACTOR.append(bit_rate / VIDEO_BIT_RATE[-1])
HD_REWARD = [1, 2, 3, 12, 15, 20]
# VIDEO_BIT_RATE = HD_REWARD
BIT_RATE_LEVELS = 6
# BITRATE_REWARD = [1, 2, 3, 12, 15, 20]
BUFFER_NORM_FACTOR = 10.0
# CHUNK_TIL_VIDEO_END_CAP = 151.0
# TOTAL_VIDEO_CHUNKS = 151
M_IN_K = 1000.0
B_IN_MB = 1000000.0
BITS_IN_BYTE = 8.0
DEFAULT_QUALITY = 1  # default video quality without agent
RANDOM_SEED = 42
RAND_RANGE = 1000000
LOG_FOLDER = 'mpc_results/'
LOG_FILE = LOG_FOLDER + 'log_mpc_robust'
VIDEO_SIZE_FILE = 'video_data/video_size/video_size_'
NUM_VIDEO = 1
# log in format of time_stamp bit_rate buffer_size rebuffer_time chunk_size download_time reward
# NN_MODEL = './models/nn_model_ep_5900.ckpt'

# BITRATE_REWARD = [1, 2, 4, 6, 9, 15]
QUALITY_FACTOR = 1
RECOVERY_FACTOR = 0.25
REBUF_PENALTY = 4.3  # pensieve: 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1
VIDEO_CHUNK_LEN = 2
BANDWIDTH_ESTIMATE_LEN = 0.5  # 500ms
FRAME_RATE = 30  # 30FPS

HANDOVER_TYPE = "greedy"
MPC_FUTURE_CHUNK_COUNT = 5
SR = False


def main():
    global SR
    
    if len(sys.argv) >= 2 and sys.argv[1] == "SR":
        SR = True
        
    start = time.time()
    np.random.seed(RANDOM_SEED)

    assert len(VIDEO_BIT_RATE) == A_DIM

    f = open("bandwidth.txt", "r")
    all_cooked_bw = []
    for eachline in f.readlines():
        all_cooked_bw.append(float(eachline))

    net_env = env.Environment(all_cooked_bw=all_cooked_bw,
                              video_size_file=VIDEO_SIZE_FILE,
                              video_bit_rate=VIDEO_BIT_RATE)

    log_path = LOG_FILE

    # only for short testing
    # all_file_names = ['test']

    if not os.path.exists(LOG_FOLDER):
        os.makedirs(LOG_FOLDER)
    log_file = open(log_path, 'w')

    time_stamp = 0

    last_bit_rate = DEFAULT_QUALITY
    bit_rate = DEFAULT_QUALITY

    action_vec = np.zeros(A_DIM)
    action_vec[bit_rate] = 1

    s_batch = [np.zeros((S_INFO, S_LEN))]
    a_batch = [action_vec]
    r_batch = []
    entropy_record = []
    # Log formatting
    log_file.write("{: <15} {: <15} {: <15} {: <15} {: <15}"
                   " {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                   .format("time_stamp", "download_bw", "pred_download_bw", "bit_rate_sel",
                           "buffer_size", "rebuf_time", "chunk_size", "delay_time", "reward", "pred_bw", "num_of_users"))

    video_count = 0
    reward_log = []
    while True:  # serve video forever
        # the action is from the last decision
        # this is to make the framework similar to the real
        delay, sleep_time, buffer_size, rebuf, \
        video_chunk_size, next_video_chunk_sizes, \
        end_of_video_or_network, video_chunk_remain, avg_throughput = \
            net_env.get_video_chunk(bit_rate, handover_type=HANDOVER_TYPE, SR=SR)

        time_stamp += delay  # in ms
        time_stamp += sleep_time  # in ms

        # reward is video quality - rebuffer penalty
        reward = get_reward(bit_rate, last_bit_rate, rebuf, video_chunk_size, avg_throughput, SR=SR)
        # print("delay: ", delay, ", buffer_size: ", buffer_size, ", rebuf: ", rebuf, ", reward: ", reward, ", thrp: ", net_env.cooked_bw[net_env.mahimahi_ptr - 1])

        r_batch.append(reward)
        reward_log.append(reward)

        last_bit_rate = bit_rate

        if not end_of_video_or_network:
            avg_download_throughput = float(video_chunk_size) / float(delay) / M_IN_K * BITS_IN_BYTE  # MBit / SEC
            pred_bw, pred_download_bw = net_env.predict_future_bw(method="holt-winter")
        # print("pred_download_bw: ", pred_download_bw, "last download bw: ", net_env.download_bw[-1])
        
        # log time_stamp, bit_rate, buffer_size, reward
        log_file.write("{: <15} {: <15} {: <15} {: <15} {: <15}"
                       " {: <15} {: <15} {: <15} {: <15} {: <15} {: <15}\n"
                       .format(str(round(time_stamp, 3)), str(round(avg_download_throughput, 3)), 
                               str(round(pred_download_bw, 3)), str(VIDEO_BIT_RATE[bit_rate]), 
                               str(round(buffer_size, 3)), str(round(rebuf, 3)), str(round(video_chunk_size, 3)), 
                               str(round(delay, 3)), str(round(reward, 3)), str(round(pred_bw, 3)), str(0)))

        log_file.flush()

        # retrieve previous state
        if len(s_batch) == 0:
            state = [np.zeros((S_INFO, S_LEN))]
        else:
            state = np.array(s_batch[-1], copy=True)

        # dequeue history record
        state = np.roll(state, -1, axis=1)

        # this should be S_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bit_rate] / float(np.max(VIDEO_BIT_RATE))  # last quality
        state[1, -1] = buffer_size / BUFFER_NORM_FACTOR / 10
        state[2, -1] = rebuf
        state[3, -1] = float(video_chunk_size) / float(delay) / M_IN_K  # kilo byte / ms
        # state[4, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(CHUNK_TIL_VIDEO_END_CAP)
        # state[5: 10, :] = future_chunk_sizes / M_IN_K / M_IN_K

        # future chunks length (try 4 if that many remaining)
        if not end_of_video_or_network:
            if SR:
                bit_rate = calculate_bit_rate_mpc_sr(net_env, video_chunk_remain, buffer_size, bit_rate, pred_download_bw,
                                                last_bit_rate)
            else:
                bit_rate = calculate_bit_rate_mpc(net_env, video_chunk_remain, buffer_size, bit_rate, pred_download_bw,
                                                last_bit_rate)
            print("video counter: ", net_env.video_chunk_counter, ", last_bit_rate: ", last_bit_rate, ", bit_rate: ", bit_rate, ", reward: ", reward)

        # ================================================

        # Note: we need to discretize the probability into 1/RAND_RANGE steps,
        # because there is an intrinsic discrepancy in passing single state and batch states
        s_batch.append(state)

        if end_of_video_or_network:
            last_bit_rate = DEFAULT_QUALITY
            bit_rate = DEFAULT_QUALITY  # use the default action here

            del s_batch[:]
            del a_batch[:]
            del r_batch[:]

            action_vec = np.zeros(A_DIM)
            action_vec[bit_rate] = 1

            s_batch.append(np.zeros((S_INFO, S_LEN)))
            a_batch.append(action_vec)
            entropy_record = []

            video_count += 1
            print("video count", video_count)

            if video_count >= NUM_VIDEO:
                break

            print("avg QoE: ", sum(reward_log) / len(reward_log))

    print("avg QoE: ", sum(reward_log) / len(reward_log))
    print("time: ", time.time() - start)

def get_reward(curr_quality, last_quality, rebuf, video_chunk_size, avg_thrp, SR=False):
    if SR:
        recovery = 0
        if rebuf > 0:
            recovery = rebuf * avg_thrp / video_chunk_size
            print("rebuf: ", rebuf, VIDEO_BIT_RATE[-1] * SR_QUALITY_FACTOR[curr_quality] * recovery * RECOVERY_FACTOR / M_IN_K, REBUF_PENALTY * rebuf)
        assert VIDEO_BIT_RATE[-1] * SR_QUALITY_FACTOR[curr_quality] * recovery * RECOVERY_FACTOR / M_IN_K <= REBUF_PENALTY * rebuf
        reward = VIDEO_BIT_RATE[-1] * SR_QUALITY_FACTOR[curr_quality] * QUALITY_FACTOR / M_IN_K \
                 + VIDEO_BIT_RATE[-1] * SR_QUALITY_FACTOR[curr_quality] * recovery * RECOVERY_FACTOR / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[-1] * SR_QUALITY_FACTOR[curr_quality] -
                                           VIDEO_BIT_RATE[-1] * SR_QUALITY_FACTOR[last_quality]) / M_IN_K
    else:
        if rebuf > 0:
            print("rebuf: ", rebuf)
        reward = VIDEO_BIT_RATE[curr_quality] * QUALITY_FACTOR / M_IN_K \
                 - REBUF_PENALTY * rebuf \
                 - SMOOTH_PENALTY * np.abs(VIDEO_BIT_RATE[curr_quality] -
                                           VIDEO_BIT_RATE[last_quality]) / M_IN_K
    return reward


def calculate_bit_rate_mpc_sr(net_env, video_chunk_remain, buffer_size, bit_rate, pred_download_bw, last_bit_rate):
    # print(pred_download_bw)
    # future chunks length (try 4 if that many remaining)
    last_index = net_env.get_total_video_chunk() - video_chunk_remain
    future_chunk_length = MPC_FUTURE_CHUNK_COUNT
    if video_chunk_remain < MPC_FUTURE_CHUNK_COUNT:
        future_chunk_length = video_chunk_remain

    # all possible combinations of 5 chunk bitrates for 6 bitrate options (6^5 options)
    # iterate over list and for each, compute reward and store max reward combination
    max_reward = -10000000
    best_combo = ()
    start_buffer = buffer_size
    chunk_combo_option = []
    # make chunk combination options
    for combo in itertools.product(range(BIT_RATE_LEVELS), repeat=MPC_FUTURE_CHUNK_COUNT):
        chunk_combo_option.append(combo)
    
    for full_combo in chunk_combo_option:
        # Break at the end of the chunk
        if future_chunk_length == 0:
            send_data = last_bit_rate
            break
        combo = full_combo[0: future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        recovery_sum = 0
        smoothness_diffs = 0
        last_quality = int(bit_rate)
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            index = last_index + position  # e.g., if last chunk is 3, then first iter is 3+0+1=4
            only_download_time = (net_env.get_video_size(chunk_quality, index) / B_IN_MB) \
                            / pred_download_bw * BITS_IN_BYTE  # this is MB / Mb/s * b/B --> seconds

            download_time = only_download_time + SR_COST[chunk_quality] * FRAME_RATE * VIDEO_CHUNK_LEN / M_IN_K
            
            if curr_buffer < download_time:
                curr_rebuffer_time += (download_time - curr_buffer)
                recovery = (download_time - curr_buffer) / only_download_time
                recovery_sum += VIDEO_BIT_RATE[BIT_RATE_LEVELS - 1] * SR_QUALITY_FACTOR[chunk_quality] * recovery
                curr_buffer = 0.0
            else:
                curr_buffer -= download_time
            curr_buffer += VIDEO_CHUNK_LEN
            
            # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            bitrate_sum += VIDEO_BIT_RATE[BIT_RATE_LEVELS - 1] * SR_QUALITY_FACTOR[chunk_quality]
            smoothness_diffs += abs(VIDEO_BIT_RATE[BIT_RATE_LEVELS - 1] * SR_QUALITY_FACTOR[chunk_quality] - VIDEO_BIT_RATE[BIT_RATE_LEVELS - 1] * SR_QUALITY_FACTOR[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

        # reward = (bitrate_sum / 1000.) - (REBUF_PENALTY * curr_rebuffer_time) - (smoothness_diffs / 1000.)
        assert recovery_sum * RECOVERY_FACTOR / M_IN_K <= REBUF_PENALTY * curr_rebuffer_time
        reward = bitrate_sum * QUALITY_FACTOR / M_IN_K + recovery_sum * RECOVERY_FACTOR / M_IN_K - (REBUF_PENALTY * curr_rebuffer_time) \
                 - SMOOTH_PENALTY * smoothness_diffs / M_IN_K
        if reward > max_reward:
            best_combo = combo
            max_reward = reward
        elif reward == max_reward and sum(combo) > sum(best_combo):
            best_combo = combo
            max_reward = reward
        # send data to html side (first chunk of best combo)
    send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
    if best_combo != ():  # some combo was good
        send_data = best_combo[0]

    return send_data


def calculate_bit_rate_mpc(net_env, video_chunk_remain, buffer_size, bit_rate, pred_download_bw, last_bit_rate):
    # print(pred_download_bw)
    # future chunks length (try 4 if that many remaining)
    last_index = net_env.get_total_video_chunk() - video_chunk_remain
    future_chunk_length = MPC_FUTURE_CHUNK_COUNT
    if video_chunk_remain < MPC_FUTURE_CHUNK_COUNT:
        future_chunk_length = video_chunk_remain

    # all possible combinations of 5 chunk bitrates for 6 bitrate options (6^5 options)
    # iterate over list and for each, compute reward and store max reward combination
    max_reward = -10000000
    best_combo = ()
    start_buffer = buffer_size
    chunk_combo_option = []
    # make chunk combination options
    for combo in itertools.product(range(BIT_RATE_LEVELS), repeat=MPC_FUTURE_CHUNK_COUNT):
        chunk_combo_option.append(combo)

    for full_combo in chunk_combo_option:
        # Break at the end of the chunk
        if future_chunk_length == 0:
            send_data = last_bit_rate
            break
        combo = full_combo[0: future_chunk_length]
        # calculate total rebuffer time for this combination (start with start_buffer and subtract
        # each download time and add 2 seconds in that order)
        curr_rebuffer_time = 0
        curr_buffer = start_buffer
        bitrate_sum = 0
        smoothness_diffs = 0
        last_quality = int(bit_rate)
        for position in range(0, len(combo)):
            chunk_quality = combo[position]
            index = last_index + position  # e.g., if last chunk is 3, then first iter is 3+0+1=4
            download_time = (net_env.get_video_size(chunk_quality, index) / B_IN_MB) \
                            / pred_download_bw * BITS_IN_BYTE  # this is MB / Mb/s * b/B --> seconds
            
            if curr_buffer < download_time:
                curr_rebuffer_time += (download_time - curr_buffer)
                curr_buffer = 0.0
            else:
                curr_buffer -= download_time
            curr_buffer += VIDEO_CHUNK_LEN

            # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
            smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
            last_quality = chunk_quality
        # compute reward for this combination (one reward per 5-chunk combo)
        # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

        # reward = (bitrate_sum / 1000.) - (REBUF_PENALTY * curr_rebuffer_time) - (smoothness_diffs / 1000.)
        reward = bitrate_sum * QUALITY_FACTOR / M_IN_K - (REBUF_PENALTY * curr_rebuffer_time) \
                 - SMOOTH_PENALTY * smoothness_diffs / M_IN_K
        if reward > max_reward:
            best_combo = combo
            max_reward = reward
        elif reward == max_reward and sum(combo) > sum(best_combo):
            best_combo = combo
            max_reward = reward
        # send data to html side (first chunk of best combo)
    send_data = 0  # no combo had reward better than -1000000 (ERROR) so send 0
    if best_combo != ():  # some combo was good
        send_data = best_combo[0]

    return send_data


if __name__ == '__main__':
    main()
