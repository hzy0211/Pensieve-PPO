import itertools
from typing import List, Any, Dict
import numpy as np
import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt


FRAME_RATE = 30  # 30FPS
DURATION = 0.033  # second
SR_COST = [10, 13, 17, 22, 24, 0]  # miliseconds
VIDEO_CHUNK_LEN = 2
BANDWIDTH_ESTIMATE_LEN = 0.5  # 500ms
S_LEN = 8  # take how many frames in the past
MILLI_IN_SECOND = 1000.0
B_IN_MB = 1000000.0
DEFAULT_QUALITY = 1
BITS_IN_BYTE = 8.0
RANDOM_SEED = 42
# VIDEO_CHUNK_LEN = 2000.0  # millisec, every time add this amount to buffer
BIT_RATE_LEVELS = 6
M_IN_K = 1000.0
BUFFER_THRESH = 60.0 * MILLI_IN_SECOND  # millisec, max buffer limit
DRAIN_BUFFER_SLEEP_TIME = 500.0  # millisec
PACKET_PAYLOAD_PORTION = 0.95
LINK_RTT = 80  # millisec
NOISE_LOW = 0.9
NOISE_HIGH = 1.1
VIDEO_SIZE_FILE = 'video_data/video_size/video_size_'
# VIDEO_BIT_RATE = [10000, 20000, 30000, 60000, 90000, 140000]  # Kbpsz
# VIDEO_BIT_RATE = VIDEO_BIT_RATE + VIDEO_BIT_RATE
# HD_REWARD = [1, 2, 3, 6, 9, 14]
# HD_REWARD = HD_REWARD + HD_REWARD
# VIDEO_BIT_RATE = HD_REWARD

# LEO SETTINGS
HANDOVER_DELAY = 0.2  # sec
HANDOVER_WEIGHT = 0.3
SCALE_VIDEO_SIZE_FOR_TEST = 70
SCALE_VIDEO_LEN_FOR_TEST = 1

# MPC
MPC_FUTURE_CHUNK_COUNT = 5
QUALITY_FACTOR = 1
REBUF_PENALTY = 4.3  # pensieve: 4.3  # 1 sec rebuffering -> 3 Mbps
SMOOTH_PENALTY = 1


class Environment:
    def __init__(self, all_cooked_bw: list, random_seed=RANDOM_SEED,
                 video_size_file=VIDEO_SIZE_FILE, video_bit_rate=None):
        assert video_bit_rate is not None

        np.random.seed(random_seed)

        self.video_chunk_counter = 0
        self.buffer_size = 0

        self.last_quality = DEFAULT_QUALITY
        # pick a random trace file
        self.trace_idx = 0

        self.cooked_bw: list = all_cooked_bw

        self.video_size = {}  # in bytes
        for bit_rate in range(BIT_RATE_LEVELS):
            self.video_size[bit_rate] = []
            with open(video_size_file + str(bit_rate) + ".txt") as f:
                for line in f:
                    self.video_size[bit_rate].append(int(line.split()[0]) * SCALE_VIDEO_SIZE_FOR_TEST)

            # For Test
            original_list = self.video_size[bit_rate]
            for i in range(SCALE_VIDEO_LEN_FOR_TEST - 1):
                self.video_size[bit_rate].extend(original_list)

        self.video_len = len(self.video_size[0]) - 1

        self.mahimahi_start_ptr = 1
        self.last_mahimahi_time = 0
        # randomize the start point of the trace
        # note: trace file starts with time 0
        self.mahimahi_ptr = 1

        self.video_bit_rate = video_bit_rate

        # MPC
        self.past_bw_errors: Dict[int:List[float]] = {}
        self.past_download_bw_errors = []
        self.past_bw_ests: Dict[int:List[float]] = {}
        self.past_download_ests: List[float] = []
        # self.harmonic_bw: Dict[int:float] = {}
        self.download_bw: List[float] = []

    def get_video_chunk(self, quality, handover_type="naive", test=False, SR=False):
        assert quality >= 0
        assert quality < BIT_RATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]  # / B_IN_MB

        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes

        throughput_log = []
        total_duration = 0.0

        assert self.mahimahi_ptr < len(self.cooked_bw)
        
        while self.mahimahi_ptr < len(self.cooked_bw):  # download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr]
            # print("curr thrp: ", throughput, ", video_chunk_size: ", video_chunk_size, ", quality: ", quality, ", self.video_chunk_counter: ", self.video_chunk_counter)

            assert throughput != 0.0
            throughput = throughput * B_IN_MB / BITS_IN_BYTE

            duration = BANDWIDTH_ESTIMATE_LEN - self.last_mahimahi_time
            assert duration >= 0
            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION
            
            # print(throughput, packet_payload, video_chunk_size)

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / PACKET_PAYLOAD_PORTION
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                throughput_log.append(throughput * fractional_time)
                total_duration += fractional_time
                break
            
            throughput_log.append(packet_payload)
            total_duration += duration
            
            delay += duration
            video_chunk_counter_sent += packet_payload
            
            self.last_mahimahi_time = 0
            self.mahimahi_ptr += 1
        
            if self.mahimahi_ptr >= len(self.cooked_bw):
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLI_IN_SECOND
        delay += LINK_RTT
        if SR:
            delay += SR_COST[quality] * (FRAME_RATE * VIDEO_CHUNK_LEN)

        # add a multiplicative noise to the delay
        # delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += VIDEO_CHUNK_LEN * MILLI_IN_SECOND

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME) * \
                         DRAIN_BUFFER_SLEEP_TIME
            self.buffer_size -= sleep_time

            while True:
                if BANDWIDTH_ESTIMATE_LEN > sleep_time / MILLI_IN_SECOND:
                    break
                sleep_time -= BANDWIDTH_ESTIMATE_LEN * MILLI_IN_SECOND
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = self.video_len - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.video_len:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

        next_video_chunk_sizes = []
        for i in range(BIT_RATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        # Record download bandwidth
        self.download_bw.append(float(video_chunk_size) / float(delay) / M_IN_K * BITS_IN_BYTE)

        self.last_quality = quality

        return delay, sleep_time, return_buffer_size / MILLI_IN_SECOND, \
               rebuf / MILLI_IN_SECOND, video_chunk_size, next_video_chunk_sizes, \
               end_of_video, video_chunk_remain, sum(throughput_log) / total_duration


    def snapshot_virtual_vars(self):
        self.virtual_mahimahi_ptr = self.mahimahi_ptr
        self.virtual_last_mahimahi_time = self.last_mahimahi_time
        self.virtual_cur_sat_id = self.cur_sat_id

    def predict_bw(self, cur_sat_id, mahimahi_ptr, robustness=True):
        curr_error = 0

        # past_bw = self.cooked_bw[self.cur_sat_id][self.mahimahi_ptr - 1]
        past_bw = self.cooked_bw[cur_sat_id][mahimahi_ptr - 1]
        if past_bw == 0:
            return 0

        if cur_sat_id in self.past_bw_ests.keys() and len(self.past_bw_ests[cur_sat_id]) > 0 \
                and mahimahi_ptr - 1 in self.past_bw_ests[cur_sat_id].keys():
            curr_error = abs(self.past_bw_ests[cur_sat_id][mahimahi_ptr - 1] - past_bw) / float(past_bw)
        if cur_sat_id not in self.past_bw_errors.keys():
            self.past_bw_errors[cur_sat_id] = []
        self.past_bw_errors[cur_sat_id].append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        start_index = mahimahi_ptr - MPC_FUTURE_CHUNK_COUNT
        if start_index < 0:
            start_index = 0

        past_bws = []
        for index in range(start_index, mahimahi_ptr):
            past_bws.append(self.cooked_bw[cur_sat_id][index])

        # Newly possible satellite case
        if all(v == 0.0 for v in past_bws):
            return self.cooked_bw[cur_sat_id][mahimahi_ptr]

        while past_bws[0] == 0.0:
            past_bws = past_bws[1:]

        bandwidth_sum = 0
        for past_val in past_bws:
            bandwidth_sum += (1 / float(past_val))

        harmonic_bw = 1.0 / (bandwidth_sum / len(past_bws))
        if cur_sat_id not in self.past_bw_ests.keys():
            self.past_bw_ests[cur_sat_id] = {}
        if mahimahi_ptr not in self.past_bw_ests[cur_sat_id].keys():
            self.past_bw_ests[cur_sat_id][mahimahi_ptr] = None
        self.past_bw_ests[cur_sat_id][mahimahi_ptr] = harmonic_bw

        if robustness:
            # future bandwidth prediction
            # divide by 1 + max of last 5 (or up to 5) errors
            error_pos = -MPC_FUTURE_CHUNK_COUNT
            if cur_sat_id in self.past_bw_errors.keys() and len(
                    self.past_bw_errors[cur_sat_id]) < MPC_FUTURE_CHUNK_COUNT:
                error_pos = -len(self.past_bw_errors[cur_sat_id])
            max_error = float(max(self.past_bw_errors[cur_sat_id][error_pos:]))
            harmonic_bw = harmonic_bw / (1 + max_error)  # robustMPC here

        return harmonic_bw

    def predict_download_bw(self, robustness=False):
        curr_error = 0

        past_download_bw = self.download_bw[-1]
        if len(self.past_download_ests) > 0:
            curr_error = abs(self.past_download_ests[-1] - past_download_bw) / float(past_download_bw)
        self.past_download_bw_errors.append(curr_error)

        # pick bitrate according to MPC
        # first get harmonic mean of last 5 bandwidths
        # past_bws = self.cooked_bw[self.cur_sat_id][start_index: self.mahimahi_ptr]
        past_bws = self.download_bw[-MPC_FUTURE_CHUNK_COUNT:]
        while past_bws[0] == 0.0:
            past_bws = past_bws[1:]

        bandwidth_sum = 0
        for past_val in past_bws:
            bandwidth_sum += (1 / float(past_val))

        harmonic_bw = 1.0 / (bandwidth_sum / len(past_bws))
        self.past_download_ests.append(harmonic_bw)

        if robustness:
            # future bandwidth prediction
            # divide by 1 + max of last 5 (or up to 5) errors
            error_pos = -MPC_FUTURE_CHUNK_COUNT
            if len(self.past_download_bw_errors) < MPC_FUTURE_CHUNK_COUNT:
                error_pos = -len(self.past_download_bw_errors)
            max_error = float(max(self.past_download_bw_errors[error_pos:]))
            harmonic_bw = harmonic_bw / (1 + max_error)  # robustMPC here

        return harmonic_bw

    def calculate_greedy_mpc(self, new_sat_id, mahimahi_ptr, target_combo=None, handover=False,
                             robustness=True, method="holt-winter"):
        harmonic_bw = None
        if method == "harmonic-mean":
            harmonic_bw = self.predict_bw(new_sat_id, mahimahi_ptr, robustness)
        elif method == "holt-winter":
            harmonic_bw = self.predict_bw_holt_winter(new_sat_id, mahimahi_ptr)
        else:
            print("Cannot happen")
            exit(1)

        assert(harmonic_bw is not None)
        # future chunks length (try 4 if that many remaining)
        video_chunk_remain = self.video_len - self.video_chunk_counter
        last_index = self.get_total_video_chunk() - video_chunk_remain

        chunk_combo_option = []
        # make chunk combination options
        for combo in itertools.product(list(range(BIT_RATE_LEVELS)), repeat=MPC_FUTURE_CHUNK_COUNT):
            chunk_combo_option.append(combo)

        future_chunk_length = MPC_FUTURE_CHUNK_COUNT
        if video_chunk_remain < MPC_FUTURE_CHUNK_COUNT:
            future_chunk_length = video_chunk_remain

        # all possible combinations of 5 chunk bitrates for 6 bitrate options (6^5 options)
        # iterate over list and for each, compute reward and store max reward combination
        max_reward = -10000000
        best_combo = ()
        start_buffer = self.buffer_size / MILLI_IN_SECOND

        if target_combo:
            chunk_combo_option = [target_combo]

        for full_combo in chunk_combo_option:
            # Break at the end of the chunk
            if future_chunk_length == 0:
                send_data = self.last_quality
                break
            combo = full_combo[0: future_chunk_length]
            # calculate total rebuffer time for this combination (start with start_buffer and subtract
            # each download time and add 2 seconds in that order)
            curr_rebuffer_time = 0
            curr_buffer = start_buffer

            bitrate_sum = 0
            smoothness_diffs = 0
            last_quality = self.last_quality
            for position in range(0, len(combo)):
                chunk_quality = combo[position]
                index = last_index + position  # e.g., if last chunk is 3, then first iter is 3+0+1=4
                download_time = (self.get_video_size(chunk_quality, index) / B_IN_MB) \
                                / harmonic_bw * BITS_IN_BYTE  # this is MB/MB/s --> seconds

                if handover and position == 0:
                    download_time += HANDOVER_DELAY

                if curr_buffer < download_time:
                    curr_rebuffer_time += (download_time - curr_buffer)
                    curr_buffer = 0.0
                else:
                    curr_buffer -= download_time
                curr_buffer += self.video_chunk_len / MILLI_IN_SECOND

                # bitrate_sum += VIDEO_BIT_RATE[chunk_quality]
                # smoothness_diffs += abs(VIDEO_BIT_RATE[chunk_quality] - VIDEO_BIT_RATE[last_quality])
                bitrate_sum += self.video_bit_rate[chunk_quality]
                smoothness_diffs += abs(self.video_bit_rate[chunk_quality] - self.video_bit_rate[last_quality])
                last_quality = chunk_quality
            # compute reward for this combination (one reward per 5-chunk combo)
            # bitrates are in Mbits/s, rebuffer in seconds, and smoothness_diffs in Mbits/s

            # reward = (bitrate_sum / 1000.) - (REBUF_PENALTY * curr_rebuffer_time) - (smoothness_diffs / 1000.)
            # 10~140 - 0~100 - 0~130
            reward = bitrate_sum * QUALITY_FACTOR / M_IN_K - (REBUF_PENALTY * curr_rebuffer_time) \
                     - SMOOTH_PENALTY * smoothness_diffs / M_IN_K
            if reward > max_reward:
                best_combo = combo
                max_reward = reward
            elif reward == max_reward and sum(combo) > sum(best_combo):
                best_combo = combo
                max_reward = reward

        return best_combo, max_reward, harmonic_bw

    def get_video_chunk_counter(self):
        return self.video_chunk_counter

    def get_total_video_chunk(self):
        return self.video_len

    def get_buffer_size(self) -> float:
        return self.buffer_size

    def predict_future_bw(self, method="holt-winter", robustness=True):
        # harmonic_bw: dict[int:float] = {}
        pred_bw = None
        pred_download_bw = None
        if method == "holt-winter":
            pred_bw = self.predict_bw_holt_winter(self.mahimahi_ptr)
            pred_download_bw = self.predict_download_bw_holt_winter()
        elif method == "harmonic":
            if robustness:
                pred_bw = self.predict_bw(self.cur_sat_id, self.mahimahi_ptr, robustness=True)
                pred_download_bw = self.predict_download_bw(robustness=True)
                # harmonic_bw[self.cur_sat_id] = pred_bw
                # self.harmonic_bw = harmonic_bw
            else:
                pred_bw = self.predict_bw(self.cur_sat_id, self.mahimahi_ptr, robustness=False)
                pred_download_bw = self.predict_download_bw(robustness=False)
                # harmonic_bw[self.cur_sat_id] = pred_bw
                # self.harmonic_bw = harmonic_bw
        else:
            print("Cannot happen")
            exit(1)
        return pred_bw, pred_download_bw

    def predict_download_bw_holt_winter(self, m=172):
        if len(self.download_bw) <= 1:
            return self.download_bw[-1]

        past_bws = pd.Series(self.download_bw)
        # cur_sat_past_bws.index.freq = 's'

        # alpha = 1 / (2 * m)
        fitted_model = ExponentialSmoothing(past_bws, trend='add').fit()
        # fitted_model = ExponentialSmoothing(cur_sat_past_bws, trend='mul').fit()

        # fitted_model = ExponentialSmoothing(cur_sat_past_bws
        # test_predictions = fitted_model.forecast(5)
        test_predictions = fitted_model.forecast(1)

        pred_bw = sum(test_predictions) / len(test_predictions)
        if pred_bw < 0:
            pred_bw = self.download_bw[-1] / 2  # zyhe: avoid predicted download bandwidth less than 0

        return pred_bw

    def predict_bw_holt_winter(self, mahimahi_ptr, num=1):
        start_index = mahimahi_ptr - MPC_FUTURE_CHUNK_COUNT
        if start_index < 0:
            start_index = 0
        # past_list = self.cooked_bw[start_index:start_index+MPC_FUTURE_CHUNK_COUNT]
        past_list = self.cooked_bw[start_index:mahimahi_ptr]  # zyhe: should be like this? 

        while len(past_list) != 0 and past_list[0] == 0.0:
            past_list = past_list[1:]

        if len(past_list) <= 1 or any(v == 0 for v in past_list):
            # Just past bw
            return self.cooked_bw[mahimahi_ptr-1]

        cur_sat_past_bws = pd.Series(past_list)
        # cur_sat_past_bws.index.freq = 's'

        # alpha = 1 / (2 * m)
        fitted_model = ExponentialSmoothing(cur_sat_past_bws, trend='add').fit()
        # fitted_model = ExponentialSmoothing(cur_sat_past_bws, trend='mul').fit()

        # fitted_model = ExponentialSmoothing(cur_sat_past_bws
        # test_predictions = fitted_model.forecast(5)
        test_predictions = fitted_model.forecast(num)

        if num == 1:
            pred_bw = sum(test_predictions) / len(test_predictions)
            if pred_bw < 0:
                pred_bw = past_list[-1] / 2  # zyhe: avoid predicted bandwidth less than 0
            return pred_bw
        
        return list(test_predictions)

    def get_video_size(self, chunk_quality, index) -> int:
        return self.video_size[chunk_quality][index]
