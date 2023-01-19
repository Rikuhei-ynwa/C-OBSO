import numpy as np
import scipy
import pandas as pd
import math
from tqdm import tqdm
import re

import Metrica_IO as mio
import Metrica_Velocities as mvel
import Metrica_PitchControl as mpc
import SPADL_config as spadlconfig


def calc_obso(PPCF: float, Transition: np.ndarray, Score, tracking: pd.Series, attack_direction=0):
    # calculate obso in single frame
    # PPCF, Score : 50 * 32
    # Transitino : 100 * 64
    Transition = np.array((Transition))
    Score = np.array((Score))
    ball_grid_x = int((tracking["ball_x"] + (spadlconfig.FIELD_LENGTH / 2)) // (spadlconfig.FIELD_LENGTH / 50))
    ball_grid_y = int((tracking["ball_y"] + (spadlconfig.FIELD_WIDTH / 2)) // (spadlconfig.FIELD_WIDTH / 32))

    # When out of the pitch
    if ball_grid_x < 0:
        ball_grid_x = 0
    elif ball_grid_x > 49:
        ball_grid_x = 49
    if ball_grid_y < 0:
        ball_grid_y = 0
    elif ball_grid_y > 31:
        ball_grid_y = 31

    Transition = Transition[31 - ball_grid_y : 63 - ball_grid_y, 49 - ball_grid_x : 99 - ball_grid_x]

    if attack_direction < 0:
        Score = np.fliplr(Score)
    elif attack_direction > 0:
        Score = Score
    else:
        print("input attack direction is 1 or -1")

    obso = PPCF * Transition * Score

    return obso, Transition


def calc_obso_seq(
    tracking_home_df,
    tracking_away_df,
    Transition,
    Score,
    attacking_direction,
    attacking_team,
):
    """
    # Args
    trakcing_home, tracking_away_df: sequence tracking data (same attacking team)
    Transition : transition model
    Score : Score model
    attacking_diction: 1 :-> or -1 :<-
    attacking_team : Home or Away
    # Returns
    OBSO_list: obso map in the sequence, shape=(time, x_grid, y_grid)
    """
    OBSO_list = []
    seq_len = len(tracking_home_df)
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home_df),
        mio.find_goalkeeper(tracking_away_df),
    ]
    for time in range(seq_len):
        frame = tracking_home_df.iloc[time].name
        PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
            tracking_home_df,
            tracking_away_df,
            frame,
            attacking_team,
            params,
            GK_numbers,
        )
        OBSO, _ = calc_obso(PPCF, Transition, Score, tracking_home_df.loc[frame], attacking_direction)
        OBSO_list.append(OBSO)

    return OBSO_list


def calc_player_evaluate(player_pos, evaluation):
    # player_pos:(x, y) col
    # evaluation : evaluation grid (32 * 50)

    # grid size
    grid_size_x = 105 / 50
    grid_size_y = 68 / 32

    player_grid_x = (player_pos[0] + 52.5) // grid_size_x
    player_grid_y = (player_pos[1] + 34) // grid_size_y

    # When out of the pitch
    if player_grid_x < 0:
        player_grid_x = 0
    elif player_grid_x > 49:
        player_grid_x = 49
    if player_grid_y < 0:
        player_grid_y = 0
    elif player_grid_y > 31:
        player_grid_y = 31

    # data format int in grid number
    player_grid_x = int(player_grid_x)
    player_grid_y = int(player_grid_y)

    # be careful for index number (y cordinate, x cordinate)
    player_ev = evaluation[player_grid_y, player_grid_x]

    return player_ev


def calc_player_evaluate_match(OBSO, events, tracking_home_df, tracking_away_df):
    # calculate player evaluation at event
    # input:obso(grid evaluation), events(event data in Metrica format), tracking home and away (tracking data)
    # return home_obso, away_obso(player evaluation at event)

    # set DataFrame column name
    column_name = ["event_number", "event_frame"]
    home_columns = tracking_home_df.columns
    home_player_num = [s[:-2] for s in home_columns if re.match(r"Home_\d*_x", s)]
    home_column_name = column_name + home_player_num
    away_columns = tracking_away_df.columns
    away_player_num = [s[:-2] for s in away_columns if re.match(r"Away_\d*_x", s)]
    away_column_name = column_name + away_player_num
    home_index = list(range(len(events[events["Team"] == "Home"])))
    away_index = list(range(len(events[events["Team"] == "Away"])))
    home_obso = pd.DataFrame(columns=home_column_name, index=home_index)
    away_obso = pd.DataFrame(columns=away_column_name, index=away_index)

    # calculate obso home or away team
    # home_event_frame = []
    # away_event_frame = []

    # initialize event number in home and away
    home_event_num = 0
    away_event_num = 0
    for num, frame in enumerate(tqdm(events["Start Frame"])):
        if events["Team"].iloc[num] == "Home":
            home_event_num += 1
            home_obso["event_frame"].iloc[home_event_num - 1] = frame
            home_obso["event_number"].iloc[home_event_num - 1] = num
            for player in home_player_num:
                home_player_pos = [
                    tracking_home_df[player + "_x"].iloc[frame],
                    tracking_home_df[player + "_y"].iloc[frame],
                ]
                if not np.isnan(home_player_pos[0]):
                    home_obso[player].iloc[home_event_num - 1] = calc_player_evaluate(home_player_pos, OBSO[num])
                else:
                    continue
        elif events["Team"].iloc[num] == "Away":
            away_event_num += 1
            away_obso["event_frame"].iloc[away_event_num - 1] = frame
            away_obso["event_number"].iloc[away_event_num - 1] = num
            for player in away_player_num:
                away_player_pos = [
                    tracking_away_df[player + "_x"].iloc[frame],
                    tracking_away_df[player + "_y"].iloc[frame],
                ]
                if not np.isnan(away_player_pos[0]):
                    away_obso[player].iloc[away_event_num - 1] = calc_player_evaluate(away_player_pos, OBSO[num])
                else:
                    continue
        else:
            continue

    return home_obso, away_obso


def calc_onball_obso(events, tracking_home_df, tracking_away_df, home_obso, away_obso):
    # calculate on-ball obso because obso is not defined in on-ball
    # input : event data in format Metrica
    # output : home_onball_obso and away_onball_obso in format pandas dataframe

    # set dataframe column name
    home_name = home_obso.columns[2:]
    away_name = away_obso.columns[2:]

    # set output dataframe
    home_onball_obso = pd.DataFrame(columns=home_obso.columns, index=list(range(len(home_obso))))
    away_onball_obso = pd.DataFrame(columns=away_obso.columns, index=list(range(len(away_obso))))

    # initialize event number in home and away
    home_event_num = 0
    away_event_num = 0

    # search on ball player
    for num, frame in enumerate(tqdm(events["Start Frame"])):
        if events["Team"].iloc[num] == "Home":
            home_event_num += 1
            dis_dict = {}
            home_onball_obso["event_frame"].iloc[home_event_num - 1] = frame
            home_onball_obso["event_number"].iloc[home_event_num - 1] = num
            for name in home_name:
                if np.isnan(tracking_home_df[name + "_x"].iloc[frame]):
                    continue
                else:
                    # initialize distance in format dictionary
                    player_pos = np.array(
                        [
                            tracking_home_df[name + "_x"].iloc[frame],
                            tracking_home_df[name + "_y"].iloc[frame],
                        ]
                    )
                    ball_pos = np.array(
                        [
                            tracking_home_df["ball_x"].iloc[frame],
                            tracking_home_df["ball_y"].iloc[frame],
                        ]
                    )
                    ball_dis = np.linalg.norm(player_pos - ball_pos)
                    dis_dict[name] = ball_dis
            # home onball player, that is the nearest player to the ball
            onball_player = min(dis_dict, key=dis_dict.get)
            home_onball_obso[onball_player].iloc[home_event_num - 1] = home_obso[onball_player].iloc[home_event_num - 1]
        elif events["Team"].iloc[num] == "Away":
            away_event_num += 1
            dis_dict = {}
            away_onball_obso["event_frame"].iloc[away_event_num - 1] = frame
            away_onball_obso["event_number"].iloc[away_event_num - 1] = num
            for name in away_name:
                if np.isnan(tracking_away_df[name + "_x"].iloc[frame]):
                    continue
                else:
                    # initialize distance in format dictionary
                    player_pos = np.array(
                        [
                            tracking_away_df[name + "_x"].iloc[frame],
                            tracking_away_df[name + "_y"].iloc[frame],
                        ]
                    )
                    ball_pos = np.array(
                        [
                            tracking_away_df["ball_x"].iloc[frame],
                            tracking_away_df["ball_y"].iloc[frame],
                        ]
                    )
                    ball_dis = np.linalg.norm(player_pos - ball_pos)
                    dis_dict[name] = ball_dis
            # away onball player, that is the nearest player to the ball
            onball_player = min(dis_dict, key=dis_dict.get)
            away_onball_obso[onball_player].iloc[away_event_num - 1] = away_obso[onball_player].iloc[away_event_num - 1]
        else:
            continue

    return home_onball_obso, away_onball_obso


def convert_Metrica_for_event(event_df: pd.DataFrame) -> pd.DataFrame:
    # convert eventdata (from spadl to Metrica)
    # event_df : event data in spadl format

    # set column name
    column_name = [
        "Team",
        "Type",
        "Subtype",
        "Period",
        "Start Frame",
        "Start Time [s]",
        "End Frame",
        "End Time [s]",
        "From",
        "To",
        "Start X",
        "Start Y",
        "End X",
        "End Y",
    ]
    Metrica_df = pd.DataFrame(columns=column_name)
    Metrica_df["Period"] = event_df["period_id"]
    Metrica_df["Start X"] = event_df["start_x"] - (spadlconfig.FIELD_LENGTH / 2)
    Metrica_df["Start Y"] = event_df["start_y"] - (spadlconfig.FIELD_WIDTH / 2)
    Metrica_df["End X"] = event_df["end_x"] - (spadlconfig.FIELD_LENGTH / 2)
    Metrica_df["End Y"] = event_df["end_y"] - (spadlconfig.FIELD_WIDTH / 2)
    Metrica_df["From"] = event_df["player_name"]
    Metrica_df["Type"] = event_df["type_name"]
    Metrica_df["Subtype"] = event_df["result_name"]

    first_period_time = event_df["time_seconds"][event_df["period_id"] == 1]
    first_endtime = max(first_period_time)
    second_period_time = event_df["time_seconds"][event_df["period_id"] == 2] + first_endtime
    start_time = pd.concat([first_period_time, second_period_time])
    end_time = start_time.shift(-1)
    start_frame = event_df["start_frame"]
    end_frame = start_frame.shift(-1)

    Metrica_df["Start Time [s]"] = start_time
    Metrica_df["Start Frame"] = start_frame
    Metrica_df["End Time [s]"] = end_time
    Metrica_df["End Frame"] = end_frame

    Metrica_df["Team"] = "Home"
    Metrica_df["Team"] = Metrica_df["Team"].mask(event_df["away_team"] == 1, "Away")

    return Metrica_df


def check_home_away_event(events_df: pd.DataFrame, tracking_home_df: pd.DataFrame, tracking_away_df: pd.DataFrame):
    # check wether corresponded event data and tracking data defined as 'Home' or 'Away'
    # input : events_df in format Metrica, tracking data in format Metrica

    # search nearest player home and away
    # set player name (ex. Home_1, ...)
    home_columns = tracking_home_df.columns
    away_columns = tracking_away_df.columns
    home_player_name = [s[:-2] for s in home_columns if re.match(r"Home_\d*_x", s)]
    away_player_name = [s[:-2] for s in away_columns if re.match(r"Away_\d*_x", s)]

    # calculate distace player to ball
    # set home distance
    dist_homeplayer2ball_list = []
    for home_player in home_player_name:
        # Exception handling for no entry player
        if np.isnan(tracking_home_df[home_player + "_x"].iloc[0]):
            continue
        else:
            ball_pos = np.array([tracking_home_df["ball_x"].iloc[0], tracking_home_df["ball_y"].iloc[0]])
            home_player_pos = np.array(
                [
                    tracking_home_df[home_player + "_x"].iloc[0],
                    tracking_home_df[home_player + "_y"].iloc[0],
                ]
            )
            dist_homeplayer2ball_list.append(np.linalg.norm(home_player_pos - ball_pos))

    # set away distance
    dist_awayplayer2ball_list = []
    for away_player in away_player_name:
        # Exception handling for no entry player
        if np.isnan(tracking_away_df[away_player + "_x"].iloc[0]):
            continue
        else:
            ball_pos = np.array([tracking_away_df["ball_x"].iloc[0], tracking_away_df["ball_y"].iloc[0]])
            away_player_pos = np.array(
                [
                    tracking_away_df[away_player + "_x"].iloc[0],
                    tracking_away_df[away_player + "_y"].iloc[0],
                ]
            )
            dist_awayplayer2ball_list.append(np.linalg.norm(away_player_pos - ball_pos))

    # judge kick-off team
    if min(dist_homeplayer2ball_list) < min(dist_awayplayer2ball_list):
        kickoff_team = "Home"
    else:
        kickoff_team = "Away"
    # print('kickoff:{}'.format(kickoff_team))
    # check team in events_df
    for i in range(len(events_df[events_df["Start Frame"] == 0])):
        if events_df.loc[i]["Team"] != "Home" and events_df.loc[i]["Team"] != "Away":
            continue
        elif kickoff_team != events_df.loc[i]["Team"]:
            # replace 'Home' to 'Away' and 'Away' to 'Home'
            events_df = events_df.replace({"Team": {"Home": "Away", "Away": "Home"}})
            # print('change team name')
            break
    return events_df


def set_trackingdata(
    tracking_home_df: pd.DataFrame,
    tracking_away_df: pd.DataFrame,
) -> pd.DataFrame:
    # data preprocessing tracking data
    # input : tarcking data (x, y) position data

    # preprocessing player position
    entry_home_df = tracking_home_df.iloc[0, :].isnull()
    entry_away_df = tracking_away_df.iloc[0, :].isnull()
    home_columns = tracking_home_df.columns
    away_columns = tracking_away_df.columns
    home_player_num = [s[:-2] for s in home_columns if re.match(r"Home_\d*_x", s)]
    away_player_num = [s[:-2] for s in away_columns if re.match(r"Away_\d*_x", s)]

    # replace nan
    for player in home_player_num:
        if entry_home_df[player + "_x"]:
            tracking_home_df[player + "_x"] = tracking_home_df[player + "_x"].fillna(method="ffill")
            tracking_home_df[player + "_y"] = tracking_home_df[player + "_y"].fillna(method="ffill")
        else:
            tracking_home_df[player + "_x"] = tracking_home_df[player + "_x"].fillna(method="bfill")
            tracking_home_df[player + "_y"] = tracking_home_df[player + "_y"].fillna(method="bfill")

    for player in away_player_num:
        if entry_away_df[player + "_x"]:
            tracking_away_df[player + "_x"] = tracking_away_df[player + "_x"].fillna(method="ffill")
            tracking_away_df[player + "_y"] = tracking_away_df[player + "_y"].fillna(method="ffill")
        else:
            tracking_away_df[player + "_x"] = tracking_away_df[player + "_x"].fillna(method="bfill")
            tracking_away_df[player + "_y"] = tracking_away_df[player + "_y"].fillna(method="bfill")

    # data interpolation in ball position in tracking data
    tracking_home_df["ball_x"] = tracking_home_df["ball_x"].interpolate()
    tracking_home_df["ball_y"] = tracking_home_df["ball_y"].interpolate()
    tracking_away_df["ball_x"] = tracking_away_df["ball_x"].interpolate()
    tracking_away_df["ball_y"] = tracking_away_df["ball_y"].interpolate()

    # check nan ball position x and y in tracking data
    # And these are interpolated by one backward value
    tracking_home_df["ball_x"] = tracking_home_df["ball_x"].fillna(method="bfill")
    tracking_home_df["ball_y"] = tracking_home_df["ball_y"].fillna(method="bfill")
    tracking_away_df["ball_x"] = tracking_away_df["ball_x"].fillna(method="bfill")
    tracking_away_df["ball_y"] = tracking_away_df["ball_y"].fillna(method="bfill")

    # filter:Savitzky-Golay
    tracking_home_df = mvel.calc_player_velocities(tracking_home_df, smoothing=True)
    tracking_away_df = mvel.calc_player_velocities(tracking_away_df, smoothing=True)

    return tracking_home_df, tracking_away_df


def remove_offside_obso(events, tracking_home_df, tracking_away_df, home_obso, away_obso):
    # remove obso value(to 0) for offise player
    # events:event data (Metrica format), tracking home and away:tracking data (Metrica foramat)
    # obso (home and away): obso value in each event

    # set parameters for calculating PPCF
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home_df),
        mio.find_goalkeeper(tracking_away_df),
    ]
    # set player name
    home_name = home_obso.columns[2:]
    away_name = away_obso.columns[2:]
    # search offside player
    for event_id in tqdm(range(len(events))):
        # check event team home or away
        if events["Team"].iloc[event_id] == "Home":
            _, _, _, attacking_players = mpc.generate_pitch_control_for_event(
                event_id, events, tracking_home_df, tracking_away_df, params, GK_numbers
            )
            attacking_players_name = [p.playername[:-1] for p in attacking_players]
            off_players = home_name ^ attacking_players_name
            for name in off_players:
                home_obso[name][home_obso["event_number"] == event_id] = 0

        elif events["Team"].iloc[event_id] == "Away":
            _, _, _, attacking_players = mpc.generate_pitch_control_for_event(
                event_id, events, tracking_home_df, tracking_away_df, params, GK_numbers
            )
            attacking_players_name = [p.playername[:-1] for p in attacking_players]
            off_players = away_name ^ attacking_players_name
            for name in off_players:
                away_obso[name][away_obso["event_number"] == event_id] = 0
        else:
            continue

    return home_obso, away_obso


def check_event_zone(events, tracking_home_df, tracking_away_df):
    # check event zone
    # input:event data format Metrica
    # output:evevnt at attackind third, middle zone, defensive third
    # zone is based on -52.5~-17.5, -17.5~+17.5, 17.5~52.5

    # set zone series format pandas Series
    zone_se = pd.DataFrame(columns=["zone"], index=events.index)
    # check attack direction
    for event_num in range(len(events)):
        if events.iloc[event_num]["Period"] == 1:
            if events.iloc[event_num]["Team"] == "Home":
                direction = mio.find_playing_direction(tracking_home_df[tracking_home_df["Period"] == 1], "Home")
            elif events.iloc[event_num]["Team"] == "Away":
                direction = mio.find_playing_direction(tracking_away_df[tracking_away_df["Period"] == 1], "Away")
            else:
                direction = 0
        elif events.iloc[event_num]["Period"] == 2:
            if events.iloc[event_num]["Team"] == "Home":
                direction = mio.find_playing_direction(tracking_home_df[tracking_home_df["Period"] == 2], "Home")
            elif events.iloc[event_num]["Team"] == "Away":
                direction = mio.find_playing_direction(tracking_away_df[tracking_away_df["Period"] == 2], "Away")
            else:
                direction = 0
        # add zone defense or middle or attack
        if direction > 0:
            if events.iloc[event_num]["Start X"] < -17.5:
                zone_se.iloc[event_num]["zone"] = "defense"
            elif events.iloc[event_num]["Start X"] > 17.5:
                zone_se.iloc[event_num]["zone"] = "attack"
            else:
                zone_se.iloc[event_num]["zone"] = "middle"
        elif direction < 0:
            if events.iloc[event_num]["Start X"] < -17.5:
                zone_se.iloc[event_num]["zone"] = "attack"
            elif events.iloc[event_num]["Start X"] > 17.5:
                zone_se.iloc[event_num]["zone"] = "defense"
            else:
                zone_se.iloc[event_num]["zone"] = "middle"
        else:
            zone_se.iloc[event_num]["zone"] = 0

    return zone_se


def mark_check(tracking_home_df, tracking_away_df, tracking_frame, attacking_team, player_num=10):
    # define mark player in defense team
    mark_df = pd.DataFrame(columns=["Attack", "Defense"])
    # calculate distance ball to player in attack team
    if attacking_team == "Home":
        # calculate distance ball to player in attack team
        home_dis_df = pd.DataFrame(columns=["number", "distance", "x_col", "y_col"])
        ball_pos = np.array(
            [
                tracking_home_df.iloc[tracking_frame]["ball_x"],
                tracking_home_df.iloc[tracking_frame]["ball_y"],
            ]
        )
        for num in range(1, 15):
            # skip non-participating player
            if np.isnan(tracking_home_df.iloc[tracking_frame]["Home_{}_x".format(num)]):
                continue
            # set position of participating player
            player_pos = np.array(
                [
                    tracking_home_df.iloc[tracking_frame]["Home_{}_x".format(num)],
                    tracking_home_df.iloc[tracking_frame]["Home_{}_y".format(num)],
                ]
            )
            # calculate distance attack player to ball
            dis = np.linalg.norm((player_pos - ball_pos))
            home_dis_df = home_dis_df.append(
                {
                    "number": "Home_{}".format(num),
                    "distance": dis,
                    "x_col": player_pos[0],
                    "y_col": player_pos[1],
                },
                ignore_index=True,
            )
        # sort by closest to ball
        home_dis_df = home_dis_df.sort_values("distance").reset_index()
        home_dis_df = home_dis_df.iloc[:player_num]
        # define mark player in defense team
        mark_df["Attack"] = home_dis_df["number"]
        defense_pos = pd.DataFrame(columns=["number", "x_col", "y_col"])
        # set position of defense player
        for num in range(1, 15):
            if np.isnan(tracking_away_df.iloc[tracking_frame]["Away_{}_x".format(num)]):
                continue
            defense_pos = defense_pos.append(
                {
                    "number": "Away_{}".format(num),
                    "x_col": tracking_away_df.iloc[tracking_frame]["Away_{}_x".format(num)],
                    "y_col": tracking_away_df.iloc[tracking_frame]["Away_{}_y".format(num)],
                },
                ignore_index=True,
            )
        # calculate distance defense player to attack player
        for att in range(player_num):
            att_pos = np.array([home_dis_df.iloc[att]["x_col"], home_dis_df.iloc[att]["y_col"]])
            att_dis = []
            for df in range(len(defense_pos)):
                df_pos = np.array([defense_pos.iloc[df]["x_col"], defense_pos.iloc[df]["y_col"]])
                dis = np.linalg.norm((att_pos - df_pos))
                att_dis.append(dis)
            defense_pos["{}".format(home_dis_df.iloc[att]["number"])] = att_dis
        # check defense player who is closet to attack player
        for num in range(player_num):
            min_index = defense_pos[mark_df.iloc[num]["Attack"]].idxmin()
            mark_df.iloc[num]["Defense"] = defense_pos.loc[min_index]["number"]
            defense_pos = defense_pos.drop(min_index)
    # calculate distance ball to player in attack team
    elif attacking_team == "Away":
        # calculate distance ball to player in attack team
        away_dis_df = pd.DataFrame(columns=["number", "distance", "x_col", "y_col"])
        ball_pos = np.array(
            [
                tracking_away_df.iloc[tracking_frame]["ball_x"],
                tracking_away_df.iloc[tracking_frame]["ball_y"],
            ]
        )
        for num in range(1, 15):
            # skip non-participating player
            if np.isnan(tracking_away_df.iloc[tracking_frame]["Away_{}_x".format(num)]):
                continue
            # set position of participating player
            player_pos = np.array(
                [
                    tracking_away_df.iloc[tracking_frame]["Away_{}_x".format(num)],
                    tracking_away_df.iloc[tracking_frame]["Away_{}_y".format(num)],
                ]
            )
            # calculate distance attack player to ball
            dis = np.linalg.norm((player_pos - ball_pos))
            away_dis_df = away_dis_df.append(
                {
                    "number": "Away_{}".format(num),
                    "distance": dis,
                    "x_col": player_pos[0],
                    "y_col": player_pos[1],
                },
                ignore_index=True,
            )
        # sort by closet to ball
        away_dis_df = away_dis_df.sort_values("distance").reset_index()
        away_dis_df = away_dis_df.iloc[:player_num]
        # define mark player in defense team
        mark_df["Attack"] = away_dis_df["number"]
        defense_pos = pd.DataFrame(columns=["number", "x_col", "y_col"])
        # set position of defense player
        for num in range(1, 15):
            if np.isnan(tracking_home_df.iloc[tracking_frame]["Home_{}_x".format(num)]):
                continue
            defense_pos = defense_pos.append(
                {
                    "number": "Home_{}".format(num),
                    "x_col": tracking_home_df.iloc[tracking_frame]["Home_{}_x".format(num)],
                    "y_col": tracking_home_df.iloc[tracking_frame]["Home_{}_y".format(num)],
                },
                ignore_index=True,
            )
        # calculate distance defense player to attack player
        for att in range(player_num):
            att_pos = np.array([away_dis_df.iloc[att]["x_col"], away_dis_df.iloc[att]["y_col"]])
            att_dis = []
            for df in range(len(defense_pos)):
                df_pos = np.array([defense_pos.iloc[df]["x_col"], defense_pos.iloc[df]["y_col"]])
                dis = np.linalg.norm((att_pos - df_pos))
                att_dis.append(dis)
            defense_pos["{}".format(away_dis_df.iloc[att]["number"])] = att_dis
        # check defense player who is closet to attack player
        for num in range(player_num):
            min_index = defense_pos[mark_df.iloc[num]["Attack"]].idxmin()
            mark_df.iloc[num]["Defense"] = defense_pos.loc[min_index]["number"]
            defense_pos = defense_pos.drop(min_index)

    return mark_df


def extract_shotseq(metrica_event_df: pd.DataFrame) -> pd.DataFrame:
    """
    This function is to extract shot sequence.

    # Args
    metrica_event_df: event data by Metrica type.

    # Returns
    shot_df: To get shot event about Team, shot event number and frame, start event number and frame.
    """
    shot_event_df = metrica_event_df[metrica_event_df["Type"] == "shot"]
    shot_event_num = shot_event_df.index.tolist()
    # set shot dataframe
    shot_df = pd.DataFrame(
        columns=[
            "Team",
            "shot_event",
            "start_event",
            "start_frame",
            "end_frame",
            "frame_length",
            "time_length[s]",
            "result",
        ]
    )
    # get start event
    start_event_num = []
    for num in shot_event_num:
        shot_Team = metrica_event_df.loc[num]["Team"]
        pre_Team = metrica_event_df.loc[num]["Team"]
        tmp = num
        while shot_Team == pre_Team:
            tmp = tmp - 1
            pre_Team = metrica_event_df.loc[tmp]["Team"]
        start_event_num.append(tmp + 1)

    # set shot_df
    shot_df["Team"] = shot_event_df["Team"]
    shot_df["result"] = shot_event_df["Subtype"]
    shot_df["shot_event"] = shot_event_num
    shot_df["start_event"] = start_event_num
    shot_df = shot_df.reset_index(drop=True)

    # search start frame
    for i in range(len(shot_event_num)):
        shot_df["start_frame"].loc[i] = metrica_event_df["Start Frame"].loc[start_event_num[i]]
        shot_df["end_frame"].loc[i] = metrica_event_df["Start Frame"].loc[shot_event_num[i]]
        shot_df["frame_length"].loc[i] = shot_df["end_frame"].loc[i] - shot_df["start_frame"].loc[i]
        shot_df["time_length[s]"].loc[i] = shot_df["frame_length"].loc[i] / spadlconfig.TRACKING_HERZ

    return shot_df


# This function calculates OBSO by frame, so it takes much longer time to do it.
# Hence, we want to fix the order.
def calc_shot_obso(
    shot_df: pd.DataFrame,
    metrica_event_df: pd.DataFrame,
    tracking_home_df: pd.DataFrame,
    tracking_away_df: pd.DataFrame,
    jursey_data_df: pd.DataFrame,
    player_data_df: pd.DataFrame,
    Trans_array: np.ndarray,
    EPV: np.ndarray,
):
    """
    This function is to calcurate shot obso, add shot_obso to shot_df.

    # Args
    tracking_home_df: tracking data of a home team.
    tracking_away_df: tracking data of an away team.
    metrica_event_df: event data by Metrica type.
    player_data_df: player data of both teams.
    jursey_data_df: jersey data of both teams.
    Trans_array: To Be Confirmed.
    EPV: To Be Confirmed.
    time_length: To Be Confirmed.

    # Returns
    shot_df: add shot OBSO to shot_df.
    """

    # set parameter
    # set OBSO list
    OBSO_list = []
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home_df),
        mio.find_goalkeeper(tracking_away_df),
    ]
    # add columns (shot_obso)
    shot_df["shot_obso"] = 0
    shot_df["shot_player"] = "Nan"

    # calculate obso in shot sequences
    for idx in range(len(shot_df)):
        ev_frame = shot_df.loc[idx]["end_frame"] - 1
        attacking_team = metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Team"]
        # check GK_numbers
        if np.isnan(tracking_home_df.loc[ev_frame]["Home_" + GK_numbers[0] + "_x"]):
            GK_numbers[0] = "12"
        if np.isnan(tracking_away_df.loc[ev_frame]["Away_" + GK_numbers[1] + "_x"]):
            GK_numbers[1] = "12"
        PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
            tracking_home_df,
            tracking_away_df,
            ev_frame,
            attacking_team,
            params,
            GK_numbers,
        )
        # check attacking direction
        if attacking_team == "Home":
            if metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(tracking_home_df[tracking_home_df["Period"] == 1], "Home")
            elif metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(tracking_home_df[tracking_home_df["Period"] == 2], "Home")
        elif attacking_team == "Away":
            if metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(tracking_away_df[tracking_away_df["Period"] == 1], "Away")
            elif metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(tracking_away_df[tracking_away_df["Period"] == 2], "Away")
        OBSO, _ = calc_obso(
            PPCF,
            Trans_array,
            EPV,
            tracking_home_df.loc[ev_frame],
            attack_direction=direction,
        )
        OBSO_list.append(OBSO)

        # search shot player
        if attacking_team == "Home":
            shot_player_name = metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["From"]
            jursey_num = int(player_data_df[player_data_df["選手名"] == shot_player_name]["背番号"])
            track_num = jursey_data_df[jursey_data_df["Home"] == jursey_num].index[0]
            shot_player = "Home_" + str(track_num)
            shot_player_x = tracking_home_df.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_home_df.loc[ev_frame][shot_player + "_y"]
            shot_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        elif attacking_team == "Away":
            shot_player_name = metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["From"]
            jursey_num = int(player_data_df[player_data_df["選手名"] == shot_player_name]["背番号"])
            track_num = jursey_data_df[jursey_data_df["Away"] == jursey_num].index[0]
            shot_player = "Away_" + str(track_num)
            shot_player_x = tracking_away_df.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_away_df.loc[ev_frame][shot_player + "_y"]
            shot_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        # insert shot obso
        shot_df["shot_player"].loc[idx] = shot_player
        shot_df["shot_obso"].loc[idx] = shot_obso

    return shot_df, OBSO_list


def generate_ghost_trajectory(tracking_home_df, tracking_away_df, shot):
    # generate ghost trajectory
    # input: tracking data (home and away), shot:shot sequence format pandas series
    # output: tracking_home_ghost, tracking_away_ghost
    # set start and end frame
    start_frame = shot["start_frame"]
    end_frame = shot["end_frame"]
    # max time length = 10 sec
    if end_frame - start_frame > 250:
        start_frame = end_frame - 250
    # define mark player
    mark_df = mark_check(
        tracking_home_df,
        tracking_away_df,
        shot["end_frame"],
        attacking_team=shot["shot_player"][:4],
    )
    # extract preeict player
    a1 = shot["shot_player"]
    d1 = mark_df[mark_df["Attack"] == a1].iloc[0]["Defense"]
    for i in range(len(mark_df)):
        if not mark_df.loc[i]["Attack"] == a1:
            a2 = mark_df.loc[i]["Attack"]
            d2 = mark_df.loc[i]["Defense"]
            break
    # generate ghost player
    tracking_home_ghost = tracking_home_df
    tracking_away_ghost = tracking_away_df
    a2_x_ghost = []
    a2_y_ghost = []
    d1_x_ghost = []
    d1_y_ghost = []
    d2_x_ghost = []
    d2_y_ghost = []
    # check start velocity
    if a1[:-2] == "Home":
        a2_vx = tracking_home_df.loc[start_frame][a2 + "_vx"]
        a2_vy = tracking_home_df.loc[start_frame][a2 + "_vy"]
        d1_vx = tracking_away_df.loc[start_frame][d1 + "_vx"]
        d1_vy = tracking_away_df.loc[start_frame][d1 + "_vy"]
        d2_vx = tracking_away_df.loc[start_frame][d2 + "_vx"]
        d2_vy = tracking_away_df.loc[start_frame][d2 + "_vy"]
        # predict liner tracjectory
        for i in range(end_frame - start_frame + 1):
            a2_x_ghost.append(tracking_home_df.loc[start_frame][a2 + "_x"] + (a2_vx / 25 * i))
            a2_y_ghost.append(tracking_home_df.loc[start_frame][a2 + "_y"] + (a2_vy / 25 * i))
            d1_x_ghost.append(tracking_away_df.loc[start_frame][d1 + "_x"] + (d1_vx / 25 * i))
            d1_y_ghost.append(tracking_away_df.loc[start_frame][d1 + "_y"] + (d1_vy / 25 * i))
            d2_x_ghost.append(tracking_away_df.loc[start_frame][d2 + "_x"] + (d2_vx / 25 * i))
            d2_y_ghost.append(tracking_away_df.loc[start_frame][d2 + "_y"] + (d2_vy / 25 * i))
        # insert ghost player
        tracking_home_ghost.loc[start_frame:end_frame][a2 + "_x"] = a2_x_ghost
        tracking_home_ghost.loc[start_frame:end_frame][a2 + "_y"] = a2_y_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d1 + "_x"] = d1_x_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d1 + "_y"] = d1_y_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d2 + "_x"] = d2_x_ghost
        tracking_away_ghost.loc[start_frame:end_frame][d2 + "_y"] = d2_y_ghost
    elif a1[:-2] == "Away":
        a2_vx = tracking_away_df.loc[start_frame][a2 + "_vx"]
        a2_vy = tracking_away_df.loc[start_frame][a2 + "_vy"]
        d1_vx = tracking_home_df.loc[start_frame][d1 + "_vx"]
        d1_vy = tracking_home_df.loc[start_frame][d1 + "_vy"]
        d2_vx = tracking_home_df.loc[start_frame][d2 + "_vx"]
        d2_vy = tracking_home_df.loc[start_frame][d2 + "_vy"]
        # predict liner tracjectory
        for i in range(end_frame - start_frame + 1):
            a2_x_ghost.append(tracking_away_df.loc[start_frame][a2 + "_x"] + (a2_vx / 25 * i))
            a2_y_ghost.append(tracking_away_df.loc[start_frame][a2 + "_y"] + (a2_vy / 25 * i))
            d1_x_ghost.append(tracking_home_df.loc[start_frame][d1 + "_x"] + (d1_vx / 25 * i))
            d1_y_ghost.append(tracking_home_df.loc[start_frame][d1 + "_y"] + (d1_vy / 25 * i))
            d2_x_ghost.append(tracking_home_df.loc[start_frame][d2 + "_x"] + (d2_vx / 25 * i))
            d2_y_ghost.append(tracking_home_df.loc[start_frame][d2 + "_y"] + (d2_vy / 25 * i))
        # insert ghost player
        tracking_away_ghost.loc[start_frame:end_frame][a2 + "_x"] = a2_x_ghost
        tracking_away_ghost.loc[start_frame:end_frame][a2 + "_y"] = a2_y_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d1 + "_x"] = d1_x_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d1 + "_y"] = d1_y_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d2 + "_x"] = d2_x_ghost
        tracking_home_ghost.loc[start_frame:end_frame][d2 + "_y"] = d2_y_ghost

    return tracking_home_ghost, tracking_away_ghost


def calc_virtual_obso(tracking_home_df, tracking_away_df, metrica_event_df, shot_df, Trans_array, EPV):
    # this function calcurate obso in virtual state
    # set pameters
    params = mpc.default_model_params()
    GK_numbers = [
        mio.find_goalkeeper(tracking_home_df),
        mio.find_goalkeeper(tracking_away_df),
    ]
    # calcurate virtual obso
    ghost_obso_list = []
    OBSO_list = []
    for idx in range(len(shot_df)):
        if shot_df.loc[idx]["frame_length"] == 0:
            ghost_obso_list.append("nan")
            continue
        ev_frame = shot_df.loc[idx]["end_frame"] - 1
        attacking_team = shot_df.loc[idx]["shot_player"][:4]
        tracking_home_ghost, tracking_away_ghost = generate_ghost_trajectory(
            tracking_home_df, tracking_away_df, shot_df.loc[idx]
        )
        # check GK_numbers
        if np.isnan(tracking_home_df.loc[ev_frame]["Home_" + GK_numbers[0] + "_x"]):
            GK_numbers[0] = "12"
        if np.isnan(tracking_away_df.loc[ev_frame]["Away_" + GK_numbers[1] + "_x"]):
            GK_numbers[1] = "12"
        PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
            tracking_home_ghost,
            tracking_away_ghost,
            ev_frame,
            attacking_team,
            params,
            GK_numbers,
        )
        # checking attacking direction
        if attacking_team == "Home":
            if metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(tracking_home_ghost[tracking_home_ghost["Period"] == 1], "Home")
            elif metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(tracking_home_ghost[tracking_home_ghost["Period"] == 2], "Home")
        elif attacking_team == "Away":
            if metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 1:
                direction = mio.find_playing_direction(tracking_away_ghost[tracking_away_ghost["Period"] == 1], "Away")
            elif metrica_event_df.loc[shot_df.loc[idx]["shot_event"]]["Period"] == 2:
                direction = mio.find_playing_direction(tracking_away_ghost[tracking_away_ghost["Period"] == 2], "Away")
        OBSO, _ = calc_obso(
            PPCF,
            Trans_array,
            EPV,
            tracking_home_ghost.loc[ev_frame],
            attack_direction=direction,
        )
        OBSO_list.append(OBSO)
        # assign evaluate
        if attacking_team == "Home":
            shot_player = shot_df.loc[idx]["shot_player"]
            shot_player_x = tracking_home_ghost.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_home_ghost.loc[ev_frame][shot_player + "_y"]
            ghost_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        elif attacking_team == "Away":
            shot_player = shot_df.loc[idx]["shot_player"]
            shot_player_x = tracking_away_ghost.loc[ev_frame][shot_player + "_x"]
            shot_player_y = tracking_away_ghost.loc[ev_frame][shot_player + "_y"]
            ghost_obso = calc_player_evaluate([shot_player_x, shot_player_y], OBSO)
        ghost_obso_list.append(ghost_obso)

    return ghost_obso_list, OBSO_list


def integrate_shotseq_tracking(
    tracking_home_df: pd.DataFrame,
    tracking_away_df: pd.DataFrame,
    metrica_event_df: pd.DataFrame,
    player_data_df: pd.DataFrame,
    jursey_data_df: pd.DataFrame,
    Trans_array: np.ndarray,
    EPV: np.ndarray,
    time_length=10,
):
    # check FM event and tracking
    # team id of FM is 124 in player data
    """
    This function is integrate shot sequence on tracking.

    # Args
    tracking_home_df: tracking data of a home team.
    tracking_away_df: tracking data of an away team.
    metrica_event_df: event data by Metrica type.
    player_data_df: player data of both teams.
    jursey_data_df: jersey data of both teams.
    Trans_array: To Be Confirmed.
    EPV: To Be Confirmed.
    time_length: To Be Confirmed.

    # Returns
    YokohamaFM_seq_tracking_list: To Be Confirmed.
    opponent_seq_tracking_list: To Be Confirmed.
    """

    shot_df = extract_shotseq(metrica_event_df)
    shot_df, _ = calc_shot_obso(
        shot_df,
        metrica_event_df,
        tracking_home_df,
        tracking_away_df,
        jursey_data_df,
        player_data_df,
        Trans_array,
        EPV,
    )
    YokohamaFM_team = player_data_df[player_data_df["チームID"] == 124].iloc[0]["ホームアウェイF"]  # 1:Home, 2:Away

    if YokohamaFM_team == 1:
        YokohamaFM_shot_df = shot_df[shot_df["Team"] == "Home"].reset_index(drop=True)
        opponent_shot_df = shot_df[shot_df["Team"] == "Away"].reset_index(drop=True)
    elif YokohamaFM_team == 2:
        YokohamaFM_shot_df = shot_df[shot_df["Team"] == "Away"].reset_index(drop=True)
        opponent_shot_df = shot_df[shot_df["Team"] == "Home"].reset_index(drop=True)
    # eliminate 0 sec sequence
    YokohamaFM_shot_df = YokohamaFM_shot_df[YokohamaFM_shot_df["frame_length"] != 0].reset_index(drop=True)
    opponent_shot_df = opponent_shot_df[opponent_shot_df["frame_length"] != 0].reset_index(drop=True)
    # FM extract tracking shot sequence
    YokohamaFM_seq_tracking_list = []
    opponent_seq_tracking_list = []
    for seq_num in range(len(YokohamaFM_shot_df)):
        start_frame = YokohamaFM_shot_df.loc[seq_num]["start_frame"]
        end_frame = YokohamaFM_shot_df.loc[seq_num]["end_frame"] - 1
        # max frame length is 250 (10sec)
        if end_frame - start_frame >= spadlconfig.TRACKING_HERZ * time_length:
            start_frame = end_frame - (spadlconfig.TRACKING_HERZ * time_length)
        # check entry player
        if YokohamaFM_team == 1:
            YokohamaFM_start = tracking_home_df.loc[start_frame].dropna().index
            opponent_start = tracking_away_df.loc[start_frame].dropna().index
            YokohamaFM_player_num = [s[:-2] for s in YokohamaFM_start if re.match(r"Home_\d*_x", s)]
            opponent_player_num = [s[:-2] for s in opponent_start if re.match(r"Away_\d*_x", s)]
            # define shot player as a2
            a2 = YokohamaFM_shot_df.loc[seq_num]["shot_player"]

            # set other players
            other_YokohamaFM_player = list(set(YokohamaFM_player_num) - set([a2]))
            YokohamaFM_players_pos_list = [s + "_x" for s in other_YokohamaFM_player] + [
                s + "_y" for s in other_YokohamaFM_player
            ]
            opponent_players_pos_list = [s + "_x" for s in opponent_player_num] + [
                s + "_y" for s in opponent_player_num
            ]
            YokohamaFM_players_pos_list = sorted(YokohamaFM_players_pos_list, key=spadlconfig.natural_keys)
            opponent_players_pos_list = sorted(opponent_players_pos_list, key=spadlconfig.natural_keys)
            other_players_pos_list = YokohamaFM_players_pos_list + opponent_players_pos_list
            a2_pos_list = [a2 + "_x", a2 + "_y"]
            entry_pos_list = a2_pos_list + other_players_pos_list + ["ball_x", "ball_y"]

            # set velocity columns
            YokohamaFM_players_vel_list = [s + "_vx" for s in other_YokohamaFM_player] + [
                s + "_vy" for s in other_YokohamaFM_player
            ]
            opponent_players_vel_list = [s + "_vx" for s in opponent_player_num] + [
                s + "_vy" for s in opponent_player_num
            ]
            YokohamaFM_players_vel_list = sorted(YokohamaFM_players_vel_list, key=spadlconfig.natural_keys)
            opponent_players_vel_list = sorted(opponent_players_vel_list, key=spadlconfig.natural_keys)
            other_players_vel_list = YokohamaFM_players_vel_list + opponent_players_vel_list
            a2_vel_list = [a2 + "_vx", a2 + "_vy"]
            entry_vel_list = a2_vel_list + other_players_vel_list + ["ball_vx", "ball_vy"]
            entry_players_list = entry_pos_list + entry_vel_list

        elif YokohamaFM_team == 2:
            YokohamaFM_start = tracking_away_df.loc[start_frame].dropna().index
            opponent_start = tracking_home_df.loc[start_frame].dropna().index
            YokohamaFM_player_num = [s[:-2] for s in YokohamaFM_start if re.match(r"Away_\d*_x", s)]
            opponent_player_num = [s[:-2] for s in opponent_start if re.match(r"Home_\d*_x", s)]
            # define shot player as a2
            a2 = YokohamaFM_shot_df.loc[seq_num]["shot_player"]

            # set other players
            other_YokohamaFM_player = list(set(YokohamaFM_player_num) - set([a2]))
            YokohamaFM_players_pos_list = [s + "_x" for s in other_YokohamaFM_player] + [
                s + "_y" for s in other_YokohamaFM_player
            ]
            opponent_players_pos_list = [s + "_x" for s in opponent_player_num] + [
                s + "_y" for s in opponent_player_num
            ]
            other_players_pos_list = sorted(YokohamaFM_players_pos_list) + sorted(opponent_players_pos_list)
            a2_pos_list = [a2 + "_x", a2 + "_y"]
            entry_pos_list = a2_pos_list + other_players_pos_list + ["ball_x", "ball_y"]

            # set velocity columns
            YokohamaFM_players_vel_list = [s + "_vx" for s in other_YokohamaFM_player] + [
                s + "_vy" for s in other_YokohamaFM_player
            ]
            opponent_players_vel_list = [s + "_vx" for s in opponent_player_num] + [
                s + "_vy" for s in opponent_player_num
            ]
            other_players_vel_list = sorted(YokohamaFM_players_vel_list) + sorted(opponent_players_vel_list)
            a2_vel_list = [a2 + "_vx", a2 + "_vy"]
            entry_vel_list = a2_vel_list + other_players_vel_list + ["ball_vx", "ball_vy"]
            entry_players_list = entry_pos_list + entry_vel_list

        # set tracking data
        tracking_df = pd.DataFrame(columns=entry_players_list)
        total_tracking_df = pd.merge(tracking_home_df, tracking_away_df)
        for player in entry_pos_list:
            tracking_df[player] = total_tracking_df.loc[start_frame - 50 : end_frame][
                player
            ]  # -50 is use of the burn-in

        # calc velocity
        for i, player in enumerate(entry_vel_list):
            for j, frame in enumerate(range(start_frame - 50, end_frame + 1)):
                after_pos = total_tracking_df[entry_pos_list[i]].loc[frame + 1]
                before_pos = total_tracking_df[entry_pos_list[i]].loc[frame]
                tracking_df[player].iloc[j] = (after_pos - before_pos) * spadlconfig.TRACKING_HERZ

        # check tracking
        tracking_df = remove_player(tracking_df)
        # check direction
        if tracking_df["ball_x"].iloc[-1] < 0:
            tracking_df = -tracking_df
        # append list of match sequence
        YokohamaFM_seq_tracking_list.append(tracking_df)

    for seq_num in range(len(opponent_shot_df)):
        start_frame = opponent_shot_df.loc[seq_num]["start_frame"]
        end_frame = opponent_shot_df.loc[seq_num]["end_frame"] - 1
        # max frame length is 250 (10sec)
        if end_frame - start_frame >= 250:
            start_frame = end_frame - 250
        # check entry player
        if YokohamaFM_team == 1:
            YokohamaFM_start = tracking_home_df.loc[start_frame].dropna().index
            opponent_start = tracking_away_df.loc[start_frame].dropna().index
            YokohamaFM_player_num = [s[:-2] for s in YokohamaFM_start if re.match(r"Home_\d*_x", s)]
            YokohamaFM_player_num = sorted(YokohamaFM_player_num, key=spadlconfig.natural_keys)
            opponent_player_num = [s[:-2] for s in opponent_start if re.match(r"Away_\d*_x", s)]
            opponent_player_num = sorted(opponent_player_num, key=spadlconfig.natural_keys)
            # define shot player as a2
            a2 = opponent_shot_df.loc[seq_num]["shot_player"]

            # set other players
            other_opponent_player_list = list(set(opponent_player_num) - set([a2]))
            YokohamaFM_players_pos_list = [s + "_x" for s in YokohamaFM_player_num] + [
                s + "_y" for s in YokohamaFM_player_num
            ]
            opponent_players_pos_list = [s + "_x" for s in other_opponent_player_list] + [
                s + "_y" for s in other_opponent_player_list
            ]
            other_players_pos_list = sorted(opponent_players_pos_list) + sorted(YokohamaFM_players_pos_list)
            a2_pos_list = [a2 + "_x", a2 + "_y"]
            entry_pos_list = a2_pos_list + other_players_pos_list + ["ball_x", "ball_y"]

            # set velocity columns
            YokohamaFM_players_vel_list = [s + "_vx" for s in YokohamaFM_player_num] + [
                s + "_vy" for s in YokohamaFM_player_num
            ]
            opponent_players_vel_list = [s + "_vx" for s in other_opponent_player_list] + [
                s + "_vy" for s in other_opponent_player_list
            ]
            other_players_vel_list = sorted(opponent_players_vel_list) + sorted(YokohamaFM_players_vel_list)
            a2_vel_list = [a2 + "_vx", a2 + "_vy"]
            entry_vel_list = a2_vel_list + other_players_vel_list + ["ball_vx", "ball_vy"]
            entry_players_list = entry_pos_list + entry_vel_list

        elif YokohamaFM_team == 2:
            YokohamaFM_start = tracking_away_df.loc[start_frame].dropna().index
            opponent_start = tracking_home_df.loc[start_frame].dropna().index
            YokohamaFM_player_num = [s[:-2] for s in YokohamaFM_start if re.match(r"Away_\d*_x", s)]
            opponent_player_num = [s[:-2] for s in opponent_start if re.match(r"Home_\d*_x", s)]
            # define shot player as a2
            a2 = opponent_shot_df.loc[seq_num]["shot_player"]

            # set other players
            other_opponent_player_list = list(set(opponent_player_num) - set([a2]))
            YokohamaFM_players_pos_list = [s + "_x" for s in YokohamaFM_player_num] + [
                s + "_y" for s in YokohamaFM_player_num
            ]
            opponent_players_pos_list = [s + "_x" for s in other_opponent_player_list] + [
                s + "_y" for s in other_opponent_player_list
            ]
            other_players_pos_list = sorted(opponent_players_pos_list) + sorted(YokohamaFM_players_pos_list)
            a2_pos_list = [a2 + "_x", a2 + "_y"]
            entry_pos_list = a2_pos_list + other_players_pos_list + ["ball_x", "ball_y"]

            # set velocity columns
            YokohamaFM_players_vel_list = [s + "_vx" for s in YokohamaFM_player_num] + [
                s + "_vy" for s in YokohamaFM_player_num
            ]
            opponent_players_vel_list = [s + "_vx" for s in other_opponent_player_list] + [
                s + "_vy" for s in other_opponent_player_list
            ]
            other_players_vel_list = sorted(opponent_players_vel_list) + sorted(YokohamaFM_players_vel_list)
            a2_vel_list = [a2 + "_vx", a2 + "_vy"]
            entry_vel_list = a2_vel_list + other_players_vel_list + ["ball_vx", "ball_vy"]
            entry_players_list = entry_pos_list + entry_vel_list

        # set tracking data
        tracking_df = pd.DataFrame(columns=entry_players_list)
        total_tracking_df = pd.merge(tracking_home_df, tracking_away_df)
        for player in entry_pos_list:
            tracking_df[player] = total_tracking_df.loc[start_frame - 50 : end_frame][player]  # -50 is use of burn-in

        # calc velocity
        for i, player in enumerate(entry_vel_list):
            for j, frame in enumerate(range(start_frame - 50, end_frame + 1)):
                after_pos = total_tracking_df[entry_pos_list[i]].loc[frame + 1]
                before_pos = total_tracking_df[entry_pos_list[i]].loc[frame]
                tracking_df[player].iloc[j] = (after_pos - before_pos) * spadlconfig.TRACKING_HERZ

        # check tracking data
        tracking_df = remove_player(tracking_df)
        # check direction
        if tracking_df["ball_x"].iloc[-1] < 0:
            tracking_df = -tracking_df
        # append list of match sequence
        opponent_seq_tracking_list.append(tracking_df)

    return YokohamaFM_seq_tracking_list, opponent_seq_tracking_list


def calc_press_value(at_pos, df_pos, df_goal_pos):
    # calcurate pressure value in toda's research
    """
    # Args
    at_pos:attacking position like array
    df_pos:defense position like array
    df_goal_pos:goal positon in defense team like array

    # Returns
    press_value(float):value of pressure
    """
    # set ndarray
    at_pos = np.array(at_pos)
    df_pos = np.array(df_pos)
    df_goal_pos = np.array(df_goal_pos)
    # calcurate angle defense and goal
    dis_at_df = np.linalg.norm(df_pos - at_pos)
    goal_vec = df_goal_pos - at_pos
    df_vec = df_pos - at_pos
    cos = np.dot(goal_vec, df_vec) / (np.linalg.norm(goal_vec) * np.linalg.norm(df_vec))
    if cos >= 1 / math.sqrt(2):
        press_value = 1 - dis_at_df / 4
    elif cos <= -1 / math.sqrt(2):
        press_value = 1 - dis_at_df / 2
    else:
        press_value = 1 - dis_at_df / 3
    # not define press value in so far defense
    if press_value < 0:
        press_value = 0

    return press_value


def get_attack_sequence(metrica_event_df, player_data_df):
    """
    # Args
    metrica_event_df: event data format Metrica
    player_data_df: involve team data

    # Returns
    attack_df: data of attack sequence
    """
    # define attack sequence
    attack_df = pd.DataFrame(
        columns=[
            "Team",
            "start_event",
            "start_frame",
            "end_event",
            "end_frame",
            "frame_length",
            "time_length[s]",
            "end_event_type",
        ]
    )
    YokohamaFM_player_data_df = player_data_df[player_data_df["チームID"] == 124]

    def extract_attack_event(event_df: pd.DataFrame) -> pd.DataFrame:
        type_seq = event_df["Type"]
        team_seq = event_df["Team"]
        attack_fouls = (type_seq == "foul") & (team_seq == team_seq.shift(1))
        attack_actions = ~type_seq.isin(spadlconfig.defense_actiontypes)
        return event_df[attack_fouls | attack_actions]

    metrica_attack_df = extract_attack_event(metrica_event_df)
    if YokohamaFM_player_data_df.iloc[0]["ホームアウェイF"] == 1:  # opponent is 2(Away)
        opponent_attack_df = metrica_attack_df[metrica_attack_df["Team"] == "Away"]
    elif YokohamaFM_player_data_df.iloc[0]["ホームアウェイF"] == 2:  # opponent is 1(Home)
        opponent_attack_df = metrica_attack_df[metrica_attack_df["Team"] == "Home"]

    opponent_attack_index = opponent_attack_df.index
    # extract opponent attack sequence
    attack_seq_list = []
    attack_index_list = []
    for i in range(len(opponent_attack_index)):
        attack_index_list.append(opponent_attack_index[i])
        if i == len(opponent_attack_index) - 1:
            break
        elif opponent_attack_index[i] + 1 == opponent_attack_index[i + 1]:
            continue
        else:
            attack_seq_list.append(attack_index_list)
            attack_index_list = []
    # assign dataframe
    Team_list = [opponent_attack_df["Team"].iloc[0]] * len(attack_seq_list)
    attack_df["Team"] = Team_list
    for i in range(len(attack_df)):
        attack_df["start_event"].loc[i] = attack_seq_list[i][0]
        attack_df["end_event"].loc[i] = attack_seq_list[i][-1]
        attack_df["start_frame"].loc[i] = opponent_attack_df.loc[attack_seq_list[i][0]]["Start Frame"]
        attack_df["end_frame"].loc[i] = opponent_attack_df.loc[attack_seq_list[i][-1]]["End Frame"]
        attack_df["frame_length"].loc[i] = attack_df["end_frame"].loc[i] - attack_df["start_frame"].loc[i]
        attack_df["time_length[s]"].loc[i] = attack_df["frame_length"].loc[i] / spadlconfig.TRACKING_HERZ
        attack_df["end_event_type"].loc[i] = opponent_attack_df.loc[attack_seq_list[i][-1]]["Type"]

    return attack_df


def attack_sequence2tracking(tracking_home_df, tracking_away_df, attack_df) -> list:
    """
    # Args
    tracking_home_df: tracking data of home team
    tracking_away_df: tracking data of away team
    attack_df: dataframe of attack sequence

    # Returns
    seq_attack_tracking_list: tracking data for attack sequence in match
    """
    # set dataframe into list
    seq_attack_tracking_list = []
    # set team name 'Home' or 'Away'
    if attack_df["Team"].loc[0] == "Home":
        opponent_team = "Home"
        opponent_tracking_df = tracking_home_df
        YokohamaFM_team = "Away"
        YokohamaFM_tracking_df = tracking_away_df
    elif attack_df["Team"].loc[0] == "Away":
        opponent_team = "Away"
        opponent_tracking_df = tracking_away_df
        YokohamaFM_team = "Home"
        YokohamaFM_tracking_df = tracking_home_df
    # check attack direction
    first_direction = mio.find_playing_direction(
        opponent_tracking_df[opponent_tracking_df["Period"] == 1], opponent_team
    )
    second_direction = mio.find_playing_direction(
        opponent_tracking_df[opponent_tracking_df["Period"] == 2], opponent_team
    )
    first_tracking_df = pd.merge(
        opponent_tracking_df[opponent_tracking_df["Period"] == 1],
        YokohamaFM_tracking_df[YokohamaFM_tracking_df["Period"] == 1],
    )
    second_tracking_df = pd.merge(
        opponent_tracking_df[opponent_tracking_df["Period"] == 2],
        YokohamaFM_tracking_df[YokohamaFM_tracking_df["Period"] == 2],
    )

    # +1 is left->right and -1 is right->left
    if first_direction == -1:
        # first_tracking_df.iloc[:, 2:] are xy coordinates, xy velocities, speed.
        first_tracking_df = first_tracking_df.iloc[:, 2:] * (-1)
    else:
        # first_tracking_df.iloc[:, 2:] are xy coordinates, xy velocities, speed.
        first_tracking_df = first_tracking_df.iloc[:, 2:]
    if second_direction == -1:
        # second_tracking_df.iloc[:, 2:] are xy coordinates, xy velocities, speed.
        second_tracking_df = second_tracking_df.iloc[:, 2:] * (-1)
    else:
        # second_tracking_df.iloc[:, 2:] are xy coordinates, xy velocities, speed.
        second_tracking_df = second_tracking_df.iloc[:, 2:]
    total_tracking_df = pd.concat([first_tracking_df, second_tracking_df], ignore_index=True)

    for seq_num in range(len(attack_df)):
        start_frame = attack_df.loc[seq_num]["start_frame"]
        end_frame = int(attack_df.loc[seq_num]["end_frame"])
        # check entry players
        opponent_start = opponent_tracking_df.loc[start_frame].dropna().index
        YokohamaFM_start = YokohamaFM_tracking_df.loc[start_frame].dropna().index
        opponent_player_num = [s[:-2] for s in opponent_start if re.match(opponent_team + r"_\d*_x", s)]
        opponent_player_num = sorted(opponent_player_num, key=spadlconfig.natural_keys)
        YokohamaFM_player_num = [s[:-2] for s in YokohamaFM_start if re.match(YokohamaFM_team + r"_\d*_x", s)]
        YokohamaFM_player_num = sorted(YokohamaFM_player_num, key=spadlconfig.natural_keys)

        # set position columns
        opponent_players_pos = [s + "_x" for s in opponent_player_num] + [s + "_y" for s in opponent_player_num]
        YokohamaFM_players_pos = [s + "_x" for s in YokohamaFM_player_num] + [s + "_y" for s in YokohamaFM_player_num]
        entry_pos_list = YokohamaFM_players_pos + opponent_players_pos + ["ball_x", "ball_y"]
        # set velocity columns
        opponent_players_vel_list = [s + "_vx" for s in opponent_player_num] + [s + "_vy" for s in opponent_player_num]
        YokohamaFM_players_vel_list = [s + "_vx" for s in YokohamaFM_player_num] + [
            s + "_vy" for s in YokohamaFM_player_num
        ]
        entry_vel_list = opponent_players_vel_list + YokohamaFM_players_vel_list + ["ball_vx", "ball_vy"]
        entry_players_list = entry_pos_list + entry_vel_list
        # set tracking dataframe
        tracking_df = pd.DataFrame(columns=entry_players_list)
        for player_pos in entry_pos_list:
            tracking_df[player_pos] = total_tracking_df.loc[start_frame:end_frame][player_pos]
        # calc velocity
        for i, player_vel in enumerate(entry_vel_list):
            for j, frame in enumerate(range(start_frame, end_frame + 1)):
                after_pos = total_tracking_df[entry_pos_list[i]].loc[frame + 1]
                before_pos = total_tracking_df[entry_pos_list[i]].loc[frame]
                tracking_df[player_vel].iloc[j] = (after_pos - before_pos) * spadlconfig.TRACKING_HERZ
        # append match sequence into list
        seq_attack_tracking_list.append(tracking_df)

    return seq_attack_tracking_list


def create_tracking_df(predict, seq_num=0):
    """
    # Args
    predict: predict tracking shape=(frame_length=121, player=3, seqs_len=717, feature_len=92)
    seq_num: sequence number

    # Returns
    attack_tracking: attacking players position (as Home)
    defense_tracking: defensing players position (as Away)
    """
    # up sampling 10Hz -> 25Hz
    # predict = signal.resample_poly(predict, 5, 2, axis=0, padtype='line')
    # set column name
    home_columns = [
        "Home_1_x",
        "Home_1_y",
        "Home_2_x",
        "Home_2_y",
        "Home_3_x",
        "Home_3_y",
        "Home_4_x",
        "Home_4_y",
        "Home_5_x",
        "Home_5_y",
        "Home_6_x",
        "Home_6_y",
        "Home_7_x",
        "Home_7_y",
        "Home_8_x",
        "Home_8_y",
        "Home_9_x",
        "Home_9_y",
        "Home_10_x",
        "Home_10_y",
        "Home_11_x",
        "Home_11_y",
        "ball_x",
        "ball_y",
        "Home_1_vx",
        "Home_1_vy",
        "Home_2_vx",
        "Home_2_vy",
        "Home_3_vx",
        "Home_3_vy",
        "Home_4_vx",
        "Home_4_vy",
        "Home_5_vx",
        "Home_5_vy",
        "Home_6_vx",
        "Home_6_vy",
        "Home_7_vx",
        "Home_7_vy",
        "Home_8_vx",
        "Home_8_vy",
        "Home_9_vx",
        "Home_9_vy",
        "Home_10_vx",
        "Home_10_vy",
        "Home_11_vx",
        "Home_11_vy",
    ]
    away_columns = [
        "Away_1_x",
        "Away_1_y",
        "Away_2_x",
        "Away_2_y",
        "Away_3_x",
        "Away_3_y",
        "Away_4_x",
        "Away_4_y",
        "Away_5_x",
        "Away_5_y",
        "Away_6_x",
        "Away_6_y",
        "Away_7_x",
        "Away_7_y",
        "Away_8_x",
        "Away_8_y",
        "Away_9_x",
        "Away_9_y",
        "Away_10_x",
        "Away_10_y",
        "Away_11_x",
        "Away_11_y",
        "ball_x",
        "ball_y",
        "Away_1_vx",
        "Away_1_vy",
        "Away_2_vx",
        "Away_2_vy",
        "Away_3_vx",
        "Away_3_vy",
        "Away_4_vx",
        "Away_4_vy",
        "Away_5_vx",
        "Away_5_vy",
        "Away_6_vx",
        "Away_6_vy",
        "Away_7_vx",
        "Away_7_vy",
        "Away_8_vx",
        "Away_8_vy",
        "Away_9_vx",
        "Away_9_vy",
        "Away_10_vx",
        "Away_10_vy",
        "Away_11_vx",
        "Away_11_vy",
    ]
    attack_tracking = pd.DataFrame(columns=home_columns, index=[list(range(len(predict)))])
    defense_tracking = pd.DataFrame(columns=away_columns, index=[list(range(len(predict)))])
    times = list(range(len(predict)))
    # set tracking
    for i in range(len(predict)):
        for j in range(3, 12):
            attack_tracking["Home_" + str(j) + "_x"].loc[i] = predict[i][0][seq_num][4 * (j + 1)]
            attack_tracking["Home_" + str(j) + "_y"].loc[i] = predict[i][0][seq_num][4 * (j + 1) + 1]
            attack_tracking["Home_" + str(j) + "_vx"].loc[i] = predict[i][0][seq_num][4 * (j + 1) + 2]
            attack_tracking["Home_" + str(j) + "_vy"].loc[i] = predict[i][0][seq_num][4 * (j + 1) + 3]
            defense_tracking["Away_" + str(j) + "_x"].loc[i] = predict[i][0][seq_num][4 * (j + 10)]
            defense_tracking["Away_" + str(j) + "_y"].loc[i] = predict[i][0][seq_num][4 * (j + 10) + 1]
            defense_tracking["Away_" + str(j) + "_vx"].loc[i] = predict[i][0][seq_num][4 * (j + 10) + 2]
            defense_tracking["Away_" + str(j) + "_vy"].loc[i] = predict[i][0][seq_num][4 * (j + 10) + 3]
        attack_tracking["Home_1_x"].loc[i] = predict[i][0][seq_num][0]
        attack_tracking["Home_1_y"].loc[i] = predict[i][0][seq_num][1]
        attack_tracking["Home_1_vx"].loc[i] = predict[i][0][seq_num][2]
        attack_tracking["Home_1_vy"].loc[i] = predict[i][0][seq_num][3]
        attack_tracking["Home_2_x"].loc[i] = predict[i][0][seq_num][12]
        attack_tracking["Home_2_y"].loc[i] = predict[i][0][seq_num][13]
        attack_tracking["Home_2_vx"].loc[i] = predict[i][0][seq_num][14]
        attack_tracking["Home_2_vy"].loc[i] = predict[i][0][seq_num][15]
        defense_tracking["Away_1_x"].loc[i] = predict[i][1][seq_num][4]
        defense_tracking["Away_1_y"].loc[i] = predict[i][1][seq_num][5]
        defense_tracking["Away_1_vx"].loc[i] = predict[i][1][seq_num][6]
        defense_tracking["Away_1_vy"].loc[i] = predict[i][1][seq_num][7]
        defense_tracking["Away_2_x"].loc[i] = predict[i][2][seq_num][8]
        defense_tracking["Away_2_y"].loc[i] = predict[i][2][seq_num][9]
        defense_tracking["Away_2_vx"].loc[i] = predict[i][2][seq_num][10]
        defense_tracking["Away_2_vy"].loc[i] = predict[i][2][seq_num][11]
        attack_tracking["ball_x"].loc[i] = predict[i][0][seq_num][88]
        attack_tracking["ball_y"].loc[i] = predict[i][0][seq_num][89]
        defense_tracking["ball_x"].loc[i] = predict[i][0][seq_num][88]
        defense_tracking["ball_y"].loc[i] = predict[i][0][seq_num][89]
    attack_tracking["Time [s]"] = times
    defense_tracking["Time [s]"] = times
    attack_tracking = attack_tracking.astype("float64")
    defense_tracking = defense_tracking.astype("float64")

    return attack_tracking, defense_tracking


def inverse_tracking(total_tracking):
    """
    # Args
    total_tracking: home and away data

    # Return
    attack_tracking: tracking data in attack team
    defense_tracking: tracking data in defense team
    """
    attack_columns = total_tracking.columns[:22]
    defense_columns = total_tracking.columns[22:44]
    attack_tracking = pd.DataFrame(columns=attack_columns)
    defense_tracking = pd.DataFrame(columns=defense_columns)
    # for attack team
    for player in attack_columns:
        attack_tracking[player] = total_tracking[player]
    # for defense team
    for player in defense_columns:
        defense_tracking[player] = total_tracking[player]
    attack_tracking["ball_x"] = total_tracking["ball_x"]
    attack_tracking["ball_y"] = total_tracking["ball_y"]
    defense_tracking["ball_x"] = total_tracking["ball_x"]
    defense_tracking["ball_y"] = total_tracking["ball_y"]
    # set time for movies
    attack_tracking["Time [s]"] = list(range(len(total_tracking)))
    defense_tracking["Time [s]"] = list(range(len(total_tracking)))

    return attack_tracking, defense_tracking


def calc_obso_for_tracking(tracking, seq_num, EPV, Trans_array, best_var):
    # this function is calcurating for samples (created by Fujii)
    """
    # Args
    tracking: tracking data for samples. shape = (frame_len, ind_player, seq_num, feature_len)
    seq_num: sequence number

    # Returns
    obso: off ball evaluation
    """
    tracking_true = tracking[1]
    tracking_predict = tracking[0][best_var]
    attack_predict, defense_predict, attack_true, defense_true = adjust_tracking(
        tracking_true, tracking_predict, seq_num
    )

    direction = 1
    frame = attack_true.iloc[-1].name
    params = mpc.default_model_params()
    GK_numbers = [mio.find_goalkeeper(attack_true), mio.find_goalkeeper(defense_true)]
    # true
    true_PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
        attack_true, defense_true, frame, "Home", params, GK_numbers
    )
    true_pos = [attack_true.loc[frame]["Home_2_x"], attack_true.loc[frame]["Home_2_y"]]
    true_obso_map, _ = calc_obso(true_PPCF, Trans_array, EPV, attack_true.loc[frame], attack_direction=direction)
    true_obso = calc_player_evaluate(true_pos, true_obso_map)
    # predict
    virtual_PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
        attack_predict, defense_predict, frame, "Home", params, GK_numbers
    )
    virtual_pos = [
        attack_predict.loc[frame]["Home_2_x"],
        attack_predict.loc[frame]["Home_2_y"],
    ]
    virtual_obso_map, _ = calc_obso(
        virtual_PPCF,
        Trans_array,
        EPV,
        attack_predict.loc[frame],
        attack_direction=direction,
    )
    virtual_obso = calc_player_evaluate(virtual_pos, virtual_obso_map)

    return true_obso, virtual_obso


def calc_obso_for_tracking_only2(tracking, seq_num, EPV, Trans_array, best_var):
    # this function is calcurating for samples (created by Fujii)
    """
    # Args
    tracking: tracking data for samples. shape = (frame_len, ind_player, seq_num, feature_len)
    seq_num: sequence number

    # Returns
    obso: off ball evaluation
    """
    tracking_true = tracking[1]
    tracking_predict = tracking[0][best_var]
    attack_predict, defense_predict, attack_true, defense_true = adjust_tracking_only2(
        tracking_true, tracking_predict, seq_num
    )

    direction = 1
    frame = attack_true.index[-1][0]
    params = mpc.default_model_params()
    GK_numbers = [mio.find_goalkeeper(attack_true), mio.find_goalkeeper(defense_true)]
    # true
    true_PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
        attack_true, defense_true, frame, "Home", params, GK_numbers
    )
    true_pos = [
        attack_true.loc[frame]["Home_2_x"].values,
        attack_true.loc[frame]["Home_2_y"].values,
    ]
    true_obso_map, _ = calc_obso(true_PPCF, Trans_array, EPV, attack_true.loc[frame], attack_direction=direction)
    true_obso = calc_player_evaluate(true_pos, true_obso_map)
    # predict
    virtual_PPCF, _, _, _ = mpc.generate_pitch_control_for_tracking(
        attack_predict, defense_predict, frame, "Home", params, GK_numbers
    )
    virtual_pos = [
        attack_predict.loc[frame]["Home_2_x"].values,
        attack_predict.loc[frame]["Home_2_y"].values,
    ]
    virtual_obso_map, _ = calc_obso(
        virtual_PPCF,
        Trans_array,
        EPV,
        attack_predict.loc[frame],
        attack_direction=direction,
    )
    virtual_obso = calc_player_evaluate(virtual_pos, virtual_obso_map)

    return true_obso, virtual_obso


def adjust_tracking(tracking_true, tracking_predict, seq_num):
    """
    # Args
    tracking_true : samples tracking data, shape=(frame_len, ind_player, seq_num, feature_len)
    tracking_predict : samples tracking data
    seq_num : sequence number

    # Returns
    attack_predict
    defense_predict
    attack_true
    defense_true
    """
    # set true tracking
    attack_true, defense_true = create_tracking_df(tracking_true, seq_num=seq_num)
    attack_true = attack_true.dropna(how="any")
    defense_true = defense_true.dropna(how="any")
    # predict tracking
    attack_predict, defense_predict = create_tracking_df(tracking_predict, seq_num=seq_num)
    first_index = attack_true.index[0][0]
    last_index = attack_true.index[-1][0]
    attack_predict = attack_predict.loc[first_index:last_index]
    defense_predict = defense_predict.loc[first_index:last_index]

    attack_predict = attack_predict.reset_index()
    defense_predict = defense_predict.reset_index()
    attack_true = attack_true.reset_index()
    defense_true = defense_true.reset_index()

    return attack_predict, defense_predict, attack_true, defense_true


def adjust_tracking_only2(tracking_true, tracking_predict, seq_num):
    """
    # Args
    tracking_true : samples tracking data, shape=(frame_len, ind_player, seq_num, feature_len)
    tracking_predict : samples trakcing data
    seq_num : sequence number

    # Returns
    attack_predict
    defense_predict
    attack_true
    defense_true
    """
    # set true tracking
    attack_true, defense_true = create_tracking_df(tracking_true, seq_num=seq_num)
    attack_true = attack_true.dropna(how="any")
    defense_true = defense_true.dropna(how="any")
    # predict tracking
    attack_predict, defense_predict = create_tracking_df(tracking_predict, seq_num=seq_num)
    first_index = attack_true.index[0][0]
    last_index = attack_true.index[-1][0]
    attack_predict = attack_predict.loc[first_index:last_index]
    defense_predict = defense_predict.loc[first_index:last_index]
    defense_predict["Away_2_x"] = defense_true["Away_2_x"]
    defense_predict["Away_2_y"] = defense_true["Away_2_y"]
    defense_predict["Away_2_vx"] = defense_true["Away_2_vx"]
    defense_predict["Away_2_vy"] = defense_true["Away_2_vy"]
    attack_predict = attack_predict.reset_index()
    defense_predict = defense_predict.reset_index()
    attack_true = attack_true.reset_index()
    defense_true = defense_true.reset_index()

    return attack_predict, defense_predict, attack_true, defense_true


def remove_player(seq_tracking_df: pd.DataFrame) -> pd.DataFrame:
    """
    remove unnecessary information (normally 12th player position)
    # Args
    seq_tracking_df: tracking dataframe

    # Returns
    seq_tracking_df
    """
    colums_list = list(seq_tracking_df.columns)
    home_columns = [s for s in colums_list if re.match("Home" + r"_\d*_x", s)]
    away_columns = [s for s in colums_list if re.match("Away" + r"_\d*_x", s)]
    if len(home_columns) > 11:
        over_num = len(home_columns) - 11
        exit_num = 0
        for name in home_columns:
            name_x = seq_tracking_df[name].iloc[0]
            if name_x == seq_tracking_df[name].iloc[-1] and name_x == seq_tracking_df[name].iloc[5]:
                exit_name = name
                exit_num += 1
                exit_list = [
                    exit_name,
                    exit_name[:-2] + "_y",
                    exit_name[:-2] + "_vx",
                    exit_name[:-2] + "_vy",
                ]
                seq_tracking_df = seq_tracking_df.drop(exit_list, axis=1)
            if exit_num == over_num:
                break

    if len(away_columns) > 11:
        over_num = len(away_columns) - 11
        exit_num = 0
        for name in away_columns:
            name_x = seq_tracking_df[name].iloc[0]
            if name_x == seq_tracking_df[name].iloc[-1] and name_x == seq_tracking_df[name].iloc[5]:
                exit_name = name
                exit_num += 1
                exit_list = [
                    exit_name,
                    exit_name[:-2] + "_y",
                    exit_name[:-2] + "_vx",
                    exit_name[:-2] + "_vy",
                ]
                seq_tracking_df = seq_tracking_df.drop(exit_list, axis=1)
            if exit_num == over_num:
                break
    if len(seq_tracking_df.columns) != 92:
        print("Imcomplete this process")

    return seq_tracking_df


def player_eva_seq(obso_seq, player_x, player_y):
    """
    # Args
    obso_seq : list obso map
    player_x : list of x cordinate
    player_y : list of y cordinate

    # Returns
    player_ev_seq : list of player evaluation
    """
    player_ev_seq = []
    for i in range(len(obso_seq)):
        player_ev = calc_player_evaluate([player_x[i], player_y[i]], obso_seq[i])
        player_ev_seq.append(player_ev)

    return player_ev_seq


def calc_shot_angle(tmp_pos):
    """
    # Args
    tmp_pos : temporary position, (2d numpy array)

    # Returns
    angle : shot angle
    """
    upper_post_pos = np.array([52.5, 3.66])
    lower_post_pos = np.array([52.5, -3.66])
    upper_vec = upper_post_pos - tmp_pos
    lower_vec = lower_post_pos - tmp_pos
    inner = np.inner(upper_vec, lower_vec)
    n = np.linalg.norm(upper_vec) * np.linalg.norm(lower_vec)
    cos = inner / n
    angle = np.rad2deg(np.arccos(np.clip(cos, -1.0, 1.0)))

    return angle


def create_shot_vec(tmp_pos, vec_num=100):
    """
    # Args
    tmp_pos : current position (grid center), numpy array (x, y)
    vec_num : the number of vector to the goal

    # Returns
    shot_vecs : shot vectors, 2D numpy array ([lower,..., upper])
    """
    # goal post
    upper_post_pos = np.array([52.5, 3.66])
    lower_post_pos = np.array([52.5, -3.66])
    goal_width = upper_post_pos[1] - lower_post_pos[1]
    # calcurate shot vector
    shot_vecs = []
    for i in range(vec_num):
        goal_pos = np.array([52.5, -3.66 + ((goal_width / 99) * i)])
        shot_vec = goal_pos - tmp_pos
        shot_vecs.append(shot_vec)

    return shot_vecs


def create_shot_vec_angle(tmp_pos):
    """
    # Args
    tmp_pos : current position (grid center), numpy array (x, y)

    # Returns
    shot_vecs : shot vectors based on shot angle, 2D numpy array ([lower, ..., upper])
    """
    shot_angle = calc_shot_angle(tmp_pos)
    shot_angle = int(shot_angle)
    if shot_angle < 1:
        shot_angle = 1
    # calc shot angle
    shot_vecs = []
    if shot_angle == 1:
        goal_pos = np.array([52.5, 0.0])
        shot_vec = goal_pos - tmp_pos
        shot_vecs.append(shot_vec)
    else:
        goal_width = 3.66 + 3.66
        for i in range(shot_angle):
            goal_pos = np.array([52.5, -3.66 + ((goal_width / (shot_angle - 1)) * i)])
            shot_vec = goal_pos - tmp_pos
            shot_vecs.append(shot_vec)

    return shot_vecs


def extract_shotblock_player(defense, tmp_pos):
    """
    # Args
    defense : pandas Series
    tmp_pos : current position (grid center), numpy array (x, y)
    # Returns
    block_players : the players who block shot, list
    """
    block_players = defense[defense > tmp_pos[0]].index
    block_players = [s[:-2] for s in block_players if re.match(r"Away_\d*_x", s)]
    # find GK
    df_players = defense.index
    df_players = [s[:-2] for s in df_players if re.match(r"Away_\d*_x", s)]
    x_cols = []
    for df_player in df_players:
        x_cols.append(defense[df_player + "_x"])
    x_cols = np.array(x_cols)
    GK_player = df_players[x_cols.argmax()]

    return block_players, GK_player


def calc_dis_point2line(start_pos, vec, point):
    """
    # Args
    start_pos: current position, passing line (numpy array 2d)
    vec : vector, (numpy array 2d)
    point : the point which calcurates the line

    # Returns
    dis : distance point to line
    """
    # calcurate distance as line:ax+by+c=0, point:(x0, y0)
    x0 = point[0]
    y0 = point[1]
    a = vec[1]
    b = -vec[0]
    c = -vec[1] * start_pos[0] + vec[0] * start_pos[1]
    dis = abs(a * x0 + b * y0 + c) / math.sqrt(pow(a, 2) + pow(b, 2))

    return dis


def calc_block_gauss(defense, grid):
    """
    # Args
    defense : defense position data, pandas Series
    grid : shot position like arrray (0...31, 0...49)

    # Returns
    gauss_map : shot block gauss distribution
    """
    # define shot block distribution
    x_cols = np.array([1.05 - 52.5 + 2.1 * i for i in range(50)])
    y_cols = np.array([1.0625 - 34 + 2.125 * i for i in range(32)])
    grid_center = np.array([x_cols[grid[1]], y_cols[grid[0]]])
    block_players, GK_player = extract_shotblock_player(defense, grid_center)
    gauss_map = np.zeros((680, 1050))
    x, y = np.meshgrid(np.linspace(-52.5, 52.5, 1050), np.linspace(-34, 34, 680))
    field_grid = np.dstack((x, y))
    for player in block_players:
        player_pos_x = defense[player + "_x"]
        player_pos_y = defense[player + "_y"]
        player_pos = np.array([player_pos_x, player_pos_y])
        sigma = 0.5 + np.linalg.norm(grid_center - player_pos)
        cov = np.array([[sigma, 0.0], [0.0, sigma]])
        z = scipy.stats.multivariate_normal(player_pos, cov).pdf(field_grid)
        gauss_map += z
    # add GK blocking gauss
    GK_pos_x = defense[GK_player + "_x"]
    GK_pos_y = defense[GK_player + "_y"]
    GK_pos = np.array([GK_pos_x, GK_pos_y])
    sigma = 0.5 + np.linalg.norm(grid_center - GK_pos)
    cov = np.array([[sigma, 0.0], [0.0, sigma]])
    z = scipy.stats.multivariate_normal(GK_pos, cov).pdf(field_grid)
    gauss_map += z

    return gauss_map


def calc_shot_proba(defense, grid):
    """
    # Args
    defense : defense position data, pandas Series
    grid : shot position like array (0...31, 0...49)

    # Returns
    shot_proba : list of shot probability (100 shots)
    shot_proba_mean : mean of shot probability, top of 20 shots
    """
    x_cols = np.array([1.05 - 52.5 + 2.1 * i for i in range(50)])
    y_cols = np.array([1.0625 - 34 + 2.125 * i for i in range(32)])
    grid_center = np.array([x_cols[grid[1]], y_cols[grid[0]]])
    gauss_map = calc_block_gauss(defense, grid)
    shot_vecs = create_shot_vec_angle(grid_center)
    shot_frame = 0.1
    block_odds = []
    for i, shot_vec in enumerate(shot_vecs):
        block_odd = 0
        tmp_pos = grid_center
        shot_len = np.linalg.norm(shot_vec)
        shot_points = int(shot_len // shot_frame)
        for point in range(shot_points):
            per_vec = shot_vec / shot_points
            tmp_pos = tmp_pos + per_vec
            grid_x = int(tmp_pos[0] * 10) + 525 - 1
            grid_y = int(tmp_pos[1] * 10) + 340 - 1
            block_odd += gauss_map[grid_y][grid_x]
        block_odds.append(block_odd)
    shot_proba = np.full(len(block_odds), 4) - np.array(block_odds)
    # mean 5 degrees
    shot_angle = calc_shot_angle(grid_center)
    aim_degree = 5
    if len(shot_proba) < 5:
        shot_proba_mean = np.sum(shot_proba) / 5
        # shot_proba_mean = shot_proba_mean * shot_angle /180
    else:
        shot_proba_mean = np.mean(np.sort(shot_proba)[::-1][0:aim_degree])
        shot_proba_mean = shot_proba_mean * shot_angle / 180

    return shot_proba, shot_proba_mean


def create_shotproba_map(defense):
    """
    # Args
    defense : defense position data, pandas Series
    # Returns
    shot_proba_map : the map of shot probability, shape=(32, 50)
    """
    shot_proba_map_mean = np.zeros((32, 50))
    shot_proba_map_sum = np.zeros((32, 50))
    for i in range(32):
        for j in range(50):
            grid_num = [i, j]
            shot_proba, shot_proba_mean = calc_shot_proba(defense, grid_num)
            shot_proba_map_mean[i][j] = shot_proba_mean
            shot_proba_map_sum[i][j] = np.sum(shot_proba)

    return shot_proba_map_mean, shot_proba_map_sum
