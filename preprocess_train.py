# this code is create train data for player prediction

import argparse
import os
import warnings
import pickle
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

import Metrica_EPV as mepv
import third_party as thp
import obso_player as obs

warnings.simplefilter("ignore")

start = time.time()

# create argparse
parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, default=10, help="time length[s] for shot sequences")
args = parser.parse_args()
# set path
Jdata_YokohamaFM_path = "../Data_2019FM/"
event_data_name = "/play.csv"
player_data_name = "/player.csv"
jursey_num_name = "/juseynumber.csv"

tracking_data_name1 = "/tracking_1stHalf.csv"
tracking_data_name2 = "/tracking_2ndHalf.csv"
files = os.listdir(path=Jdata_YokohamaFM_path)
game_date_list = [f for f in files if os.path.isdir(os.path.join(Jdata_YokohamaFM_path, f))]
game_date_list.remove("@eaDir")
game_date_list = sorted(game_date_list)

# load data Trans and EPV
EPV = mepv.load_EPV_grid("EPV_grid.csv")
EPV = EPV / np.max(EPV)
Trans_df = pd.read_csv("Transition_gauss.csv", header=None)
Trans_array = np.array((Trans_df))
Trans_array = Trans_array / np.max(Trans_array)

# set initial data
YokohamaFM_data_list = []
opponent_data_list = []
opponent_data_train_list = []
time_len = args.len

# create input data in all season match
for date in tqdm(game_date_list):
    print("start : ", date)
    # set match data
    player_data_df = pd.read_csv(Jdata_YokohamaFM_path + date + player_data_name, encoding="shift_jis", index_col=0)
    jursey_data_df = pd.read_csv(Jdata_YokohamaFM_path + date + jursey_num_name, encoding="shift_jis", index_col=0)

    # set tracking data
    tracking_home_df = pd.read_csv(Jdata_YokohamaFM_path + date + "/home_tracking.csv", index_col=0)
    tracking_away_df = pd.read_csv(Jdata_YokohamaFM_path + date + "/away_tracking.csv", index_col=0)
    tracking_home_df, tracking_away_df = obs.set_trackingdata(tracking_home_df, tracking_away_df)

    # set event data
    event_data_ori = pd.read_csv(Jdata_YokohamaFM_path + date + event_data_name, encoding="shift_jis")

    print("Setting time : ", time.time() - start)

    # convert the format from JLeague to spadl
    spadl_event_df = thp.convert_J2spadl(event_data_ori)
    # event data convert spadl to Metrica
    metrica_event_df = obs.convert_Metrica_for_event(spadl_event_df)
    # check 'Home' team in tracking and event data
    metrica_event_df = obs.check_home_away_event(metrica_event_df, tracking_home_df, tracking_away_df)

    print("Converting time : ", time.time() - start)

    # attack sequence
    attack_df = obs.get_attack_sequence(metrica_event_df, player_data_df)
    seq_attack_tracking_list = obs.attack_sequence2tracking(tracking_home_df, tracking_away_df, attack_df)

    print("Seq2Track time : ", time.time() - start)

    # define shot df
    # shot_df = obs.extract_shotseq(metrica_event_df)
    YokohamaFM_seq_tracking, opponent_seq_tracking = obs.integrate_shotseq_tracking(
        tracking_home_df,
        tracking_away_df,
        metrica_event_df,
        player_data_df,
        jursey_data_df,
        Trans_array,
        EPV,
        time_length=time_len,
    )
    YokohamaFM_data_list.append(YokohamaFM_seq_tracking)
    # opponent_data_list.append(opponent_seq_tracking)
    opponent_data_train_list.append(seq_attack_tracking_list)

    print("Intergrating shotseq tracking time : ", time.time() - start)

# save npy file
# YokohamaFM_data_list = np.array(YokohamaFM_data_list)
# opponent_data_list = np.array(opponent_data_list)
# np.save('FM_shot_data.npy', YokohamaFM_data_list)
# np.save('opponent_shot_data.npy', opponent_data_list)

# save pickle file
f1 = open("YokohamaFM_shot_data_addvel_{}sec.pkl".format(time_len), "wb")
pickle.dump(YokohamaFM_data_list, f1)
f1.close
# f2 = open('opponent_shot_data_addvel.pkl', 'wb')
# pickle.dump(opponent_data_list, f2)
# f2.close
# f3 = open('opponent_attack_seq.pkl', 'wb')
# pickle.dump(opponent_data_train_list, f3)
