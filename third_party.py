#!/usr/bin/env python
# coding: utf-8
# import pdb
# from typing import Any, Optional

# In[113]:
import catboost
import xgboost
from sklearn.metrics import brier_score_loss, roc_auc_score, f1_score

import numpy as np
import pandas as pd

# import pandera as pa
# from pandera.typing import Series, DataFrame
import tqdm
import socceraction.atomic.vaep.formula as vaepformula
import socceraction.vaep.features as fs
import socceraction.vaep.labels as lab
import SPADL_config as spadlconfig

# third party in J League data


# In[114]:


def create_features(spadl_h5, features_h5, labels_h5, games):
    # Create features from original hdf files
    # input : hdf files spadl, features and labels
    # games
    # output : X, Y
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        # fs.bodypart,
        fs.bodypart_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        # fs.time,
        fs.time_delta,
        # fs.actiontype_result_onehot
    ]
    nb_prev_actions = 3
    Xcols = fs.feature_column_names(xfns, nb_prev_actions)
    X = []
    for game_id in tqdm.tqdm(games.game_id, desc="selecting features"):
        Xi = pd.read_hdf(features_h5, f"game_{game_id}")
        X.append(Xi[Xcols])
    X = pd.concat(X).reset_index(drop=True)

    Y_cols = ["scores", "concedes"]
    Y = []
    for game_id in tqdm.tqdm(games.game_id, desc="selecting features"):
        Yi = pd.read_hdf(labels_h5, f"game_{game_id}")
        Y.append(Yi[Y_cols])
    Y = pd.concat(Y).reset_index(drop=True)

    return X, Y


# In[115]:


def convert_J2spadl(J_eventdata_df: pd.DataFrame) -> pd.DataFrame:
    # convert J data to spadl
    # input:play * 173 J feature
    # output:play * 25 spadl feature

    spadl_df = pd.DataFrame()
    game_id = J_eventdata_df["試合ID"]
    game_len = len(game_id)
    # sort time sequence
    J_eventdata_df = J_eventdata_df.sort_values("絶対時間秒数").reset_index(drop=True)

    secondhalf_start_index = J_eventdata_df.reset_index().query('アクション名=="後半開始"').index[0]
    period_id = [1] * (secondhalf_start_index) + [2] * (game_len - secondhalf_start_index)
    first_start_frame = J_eventdata_df.loc[0, "フレーム番号"]
    first_end_frame = J_eventdata_df[J_eventdata_df["アクション名"] == "前半終了"].iloc[0]["フレーム番号"]
    second_start_frame = J_eventdata_df.loc[secondhalf_start_index]["フレーム番号"]
    start_frame = J_eventdata_df["フレーム番号"] - first_start_frame
    start_frame[secondhalf_start_index:] = start_frame[secondhalf_start_index:] - (second_start_frame - first_end_frame)

    def min2sec(min: float) -> float:
        return min % 100 + 60 * (min // 100)

    time_seconds = list(map(min2sec, J_eventdata_df["ハーフ開始相対時間"]))

    def ball_x_J2spadl(x: float) -> float:
        return (x + 157.5) / 3

    def ball_y_J2spadl(y: float) -> float:
        return (y + 102) / 3

    ball_x_changed = pd.Series(map(ball_x_J2spadl, J_eventdata_df["ボールＸ"]))
    ball_y_changed = pd.Series(map(ball_y_J2spadl, J_eventdata_df["ボールＹ"]))
    start_x = ball_x_changed
    start_y = ball_y_changed
    end_x = ball_x_changed
    end_x = end_x.shift(-1)
    end_y = ball_y_changed
    end_y = end_y.shift(-1)

    def _get_type_id(J_eventdata_df: pd.DataFrame) -> int:
        type_name_J = J_eventdata_df["アクション名"]
        type_name_spadl = "non_action"
        if type_name_J.endswith("パス"):
            type_name_spadl = "pass"  # default
        elif (type_name_J == "キックオフ") or (type_name_J == "フィード"):
            type_name_spadl = "pass"
        elif type_name_J == "クロス":
            type_name_spadl = "cross"
        elif type_name_J == "スローイン":
            type_name_spadl = "throw_in"
        elif type_name_J == "CK":
            type_name_spadl = "corner_crossed"
        elif type_name_J == "PK":
            type_name_spadl = "shot_penalty"
        elif type_name_J == "GK":
            type_name_spadl = "goalkick"
        elif type_name_J == "ドリブル":
            type_name_spadl = "take_on"
        elif type_name_J == "ファウルする":
            type_name_spadl = "foul"
        elif type_name_J == "タックル":
            type_name_spadl = "tackle"
        elif type_name_J == "インターセプト":
            type_name_spadl = "interception"
        elif type_name_J == "シュート":
            type_name_spadl = "shot"
        elif (type_name_J == "直接FK") or (type_name_J == "間接FK"):
            if J_eventdata_df["F_シュート"] == 1:
                type_name_spadl = "shot_freekick"
            else:
                type_name_spadl = "freekick_crossed"
        elif type_name_J == "オウンゴール":
            type_name_spadl = "shot"
        elif type_name_J == "キャッチ":
            type_name_spadl = "keeper_save"
        elif type_name_J == "ブロック":
            if J_eventdata_df["ポジションID"] == 1:  # Goal Keeper
                type_name_spadl = "keeper_save"
            elif 2 <= J_eventdata_df["ポジションID"] <= 4:  # Field Player
                type_name_spadl = "block"
        elif type_name_J == "クリア":
            type_name_spadl = "clearance"
        elif type_name_J == "ハンドクリア":
            type_name_spadl = "keeper_punch"
        elif (type_name_J == "トラップ") or (type_name_J == "タッチ"):
            type_name_spadl = "dribble"
        else:
            type_name_spadl = "non_action"
        return spadlconfig.actiontypes.index(type_name_spadl)

    def _get_result_id(J_eventdata_df: pd.DataFrame) -> int:
        type_name_J = J_eventdata_df["アクション名"]
        A1_action_id_J = J_eventdata_df["A1_アクションID"]
        A2_action_id_J = J_eventdata_df["A2_アクションID"]
        result_name_spadl = "success"
        if type_name_J == "オウンゴール":
            result_name_spadl = "owngoal"
        elif (type_name_J == "ファウルする") & (A2_action_id_J == 19):  # Yellow Card
            result_name_spadl = "yellow_card"
        elif (type_name_J == "ファウルする") & (A2_action_id_J == 24):  # Red Card
            result_name_spadl = "red_card"
        elif (type_name_J.endswith("パス")) & (A1_action_id_J == 23 or A2_action_id_J == 23):  # Offside
            result_name_spadl = "offside"
        elif (type_name_J == "クロス" or type_name_J == "フィード") & (
            A1_action_id_J == 23 or A2_action_id_J == 23
        ):  # Offside
            result_name_spadl = "offside"
        elif (type_name_J == "直接FK" or type_name_J == "間接FK") & (
            A1_action_id_J == 23 or A2_action_id_J == 23
        ):  # Offside
            result_name_spadl = "offside"
        elif (type_name_J == "シュート" or type_name_J == "クリア") & (
            A1_action_id_J == 23 or A2_action_id_J == 23
        ):  # Offside
            result_name_spadl = "offside"
        elif J_eventdata_df["F_成功"] == 0:
            result_name_spadl = "fail"
        else:
            result_name_spadl = "success"
        return spadlconfig.results.index(result_name_spadl)

    def _get_bodypart_id(J_eventdata_df: pd.DataFrame) -> int:
        bodypart_id_J = J_eventdata_df["部位ID"]
        bodypart_name_spadl = "foot"
        if (bodypart_id_J == 0) | (bodypart_id_J == 1) | (bodypart_id_J == 2):
            bodypart_name_spadl = "foot"
        elif bodypart_id_J == 3:
            bodypart_name_spadl = "head"
        elif bodypart_id_J == 4:
            bodypart_name_spadl = "other"
        return spadlconfig.bodyparts.index(bodypart_name_spadl)

    def add_names_jleague(spadl_df: pd.DataFrame) -> pd.DataFrame:
        """Add the type name, result name and bodypart name to a SPADL dataframe.

        Parameters
        ----------
        actions : pd.DataFrame
            A SPADL dataframe.

        Returns
        -------
        pd.DataFrame
            The original dataframe with a 'type_name', 'result_name' and
            'bodypart_name' appended.
        """
        return (
            spadl_df.drop(
                columns=["type_name", "result_name", "bodypart_name"],
                errors="ignore",
            )  # add timestamp
            .merge(spadlconfig.actiontypes_df(), how="left")
            .merge(spadlconfig.results_df(), how="left")
            .merge(spadlconfig.bodyparts_df(), how="left")
        )

    def which_team_is_away(J_eventdata_df: pd.DataFrame) -> int:
        home_or_away = J_eventdata_df["ホームアウェイF"]
        team_id = J_eventdata_df["チームID"]
        away_bool = 0
        if (home_or_away == 1) & (team_id != 0):
            away_bool = 0
        elif (home_or_away == 2) & (team_id != 0):
            away_bool = 1
        else:
            away_bool = -1
        return away_bool

    # set spadl feature
    spadl_df["game_id"] = game_id
    spadl_df["period_id"] = period_id
    spadl_df["time_seconds"] = time_seconds
    spadl_df["start_frame"] = start_frame
    spadl_df["timestamp"] = J_eventdata_df["ハーフ開始相対時間"]
    spadl_df["team_id"] = J_eventdata_df["チームID"]
    spadl_df["player_id"] = J_eventdata_df["選手ID"]
    spadl_df["start_x"] = start_x
    spadl_df["start_y"] = start_y
    spadl_df["end_x"] = end_x
    spadl_df["end_y"] = end_y
    spadl_df["type_id"] = J_eventdata_df.apply(_get_type_id, axis=1)
    spadl_df["result_id"] = J_eventdata_df.apply(_get_result_id, axis=1)
    spadl_df["bodypart_id"] = J_eventdata_df.apply(_get_bodypart_id, axis=1)
    spadl_df["away_team"] = J_eventdata_df.apply(which_team_is_away, axis=1)
    spadl_df["action_id"] = range(game_len)
    spadl_df = add_names_jleague(spadl_df)
    spadl_df["player_name"] = J_eventdata_df["選手名"]
    spadl_df["jersey_number"] = J_eventdata_df["選手背番号"]
    spadl_df["team_name"] = J_eventdata_df["チーム名"]
    spadl_df["player"] = J_eventdata_df["選手名"]

    spadl_needs = spadl_df.type_id != spadlconfig.actiontypes.index("non_action")
    spadl_df = spadl_df[spadl_needs].reset_index(drop=True)

    return spadl_df


# In[116]:


def convert_spadl2train(spadl):
    # convert spadl to train data
    # input:play * 25
    # out:X play * 148 and Y play * 3
    xfns = [
        fs.actiontype,
        fs.actiontype_onehot,
        # fs.bodypart,
        fs.bodypart_onehot,
        fs.result,
        fs.result_onehot,
        fs.goalscore,
        fs.startlocation,
        fs.endlocation,
        fs.movement,
        fs.space_delta,
        fs.startpolar,
        fs.endpolar,
        fs.team,
        # fs.time,
        fs.time_delta,
        # fs.actiontype_result_onehot
    ]
    type_id = list(range(23))
    type_name = [
        "pass",
        "cross",
        "throw_in",
        "freekick_crossed",
        "freekick_shot",
        "corner_crossed",
        "corner_shot",
        "take_on",
        "foul",
        "tackle",
        "interception",
        "shot",
        "shot_penalty",
        "shot_freekick",
        "keeper_save",
        "keeper_claim",
        "keeper_punch",
        "keeper_pick_up",
        "clearance",
        "bad_touch",
        "non_action",
        "dribble",
        "goalkick",
    ]
    actiontypes = pd.DataFrame(columns=["type_id", "type_name"])
    actiontypes["type_id"] = type_id
    actiontypes["type_name"] = type_name
    bodypart_id = list(range(3))
    bodypart_name = ["foot", "head", "other"]
    bodyparts = pd.DataFrame(columns=["bodypart_id", "bodypart_name"])
    bodyparts["bodypart_id"] = bodypart_id
    bodyparts["bodypart_name"] = bodypart_name
    result_id = list(range(6))
    result_name = ["fail", "success", "offside", "owngoal", "yellow_card", "red_card"]
    results = pd.DataFrame(columns=["result_id", "result_name"])
    results["result_id"] = result_id
    results["result_name"] = result_name
    spadl = spadl.merge(actiontypes, how="left").merge(results, how="left").reset_index(drop=True)
    gamestate = fs.gamestates(spadl, 3)
    X = pd.concat([fn(gamestate) for fn in xfns], axis=1)
    nb_prev_actions = 1
    X_cols = fs.feature_column_names(xfns, nb_prev_actions)

    yfns = [lab.scores, lab.concedes, lab.goal_from_shot]
    Y = pd.concat([fn(spadl) for fn in yfns], axis=1)
    Y_cols = ["scores", "concedes"]

    return X, X_cols, Y, Y_cols


# In[117]:


def train_model(X_train, Y_train):
    # create train model
    # input
    # X_train:play * 148 features
    # Y_train:play * 3 results
    # output
    # models:train model

    models_xgb = {}
    for col in Y_train.columns:
        model_xgb = xgboost.XGBClassifier()
        model_xgb.fit(X_train, Y_train[col])
        models_xgb[col] = model_xgb

    models_cat = {}
    for col in Y_train.columns:
        model_cat = catboost.CatBoostClassifier(custom_metric="F1")
        model_cat.fit(X_train, Y_train[col])
        models_cat[col] = model_cat

    return models_xgb, models_cat


# In[118]:


def estimate_vaep(models, X_test, Y_test, test_spadl):
    # estimate vaep values
    # input
    # models:train models
    # X_test:play * 148 features
    # Y_test:play * 3 results
    # output
    # vaep_values:play * 3 values
    Y_col = ["scores", "concedes"]
    Y_hat = pd.DataFrame()
    Y_hat_label = pd.DataFrame()
    print("0.5")
    for col in Y_col:
        Y_hat[col] = [p[1] for p in models[col].predict_proba(X_test)]
        Y_hat_label[col] = np.where(Y_hat[col] > 0.5, 1, 0)

        # error handling in case Y_test not in True
        # if len(Y_test[Y_test[col] == True]) == 0:
        if len(Y_test[Y_test[col]]) == 0:
            continue
        else:
            print("{}".format(col))
            print("ROC AUC:{}".format(roc_auc_score(Y_test[col], Y_hat[col])))
            print("Brier Score:{}".format(brier_score_loss(Y_test[col], Y_hat[col])))
            print("F1 Score:{}".format(f1_score(Y_test[col], Y_hat_label[col])))
    vaep_values = vaepformula.value(test_spadl, Y_hat.scores, Y_hat.concedes)

    return vaep_values


# In[119]:


def player_rating(spadl_df, values, player_data):
    # calculate player rating
    # input
    # spadl_df:play * 25, values:play * 3, player_data:player * 8
    # output
    # player_rating:player * 7
    player_rating = pd.DataFrame(
        columns=[
            "player_id",
            "team_id",
            "player",
            "vaep_value",
            "count",
            "minutes_played",
            "vaep_rating",
        ]
    )
    in_player = player_data[player_data.出場 == 1]
    player_id = in_player["選手ID"]
    team_id = in_player["チームID"]
    player = in_player["選手名"]

    player_rating["player_id"] = player_id
    player_rating["team_id"] = team_id
    player_rating["player"] = player

    total_data = pd.concat([spadl_df, values], axis=1)

    for player in player_rating["player_id"]:
        vaep_sum = sum(total_data[total_data.player_id == player].vaep_value)
        count = len(total_data[total_data.player_id == player].vaep_value)
        player_rating.loc[player_rating.player_id == player, "vaep_value"] = vaep_sum
        player_rating.loc[player_rating.player_id == player, "count"] = count

    return player_rating


# In[120]:


def convert_DMatrix(X_train, Y_train):
    # Convert DMatrix for XGBoost
    # input X_train, Y_train:score, concedes
    # event_num = len(Y_train["scores"])
    # scores_weight = [len(Y_train[Y_train["scores"]==True]) / event_num , len(Y_train[Y_train["scores"]==False]) / event_num ]
    # concedes_weight = [len(Y_train[Y_train["concedes"]==True]) / event_num , len(Y_train[Y_train["concedes"]==False]) / event_num ]

    scores_label = pd.DataFrame()
    scores_label["scores"] = Y_train["scores"]
    # scores_label["no_scores"] = ~Y_train["scores"]
    scores_weight = scores_label * 100 + 1

    concedes_label = pd.DataFrame()
    concedes_label["concedes"] = Y_train["concedes"]
    # concedes_label["no_concedes"] = ~Y_train["concedes"]
    concedes_weight = concedes_label * 100 + 1

    dm_train_scores = xgboost.DMatrix(X_train, label=Y_train["scores"], weight=scores_weight)
    dm_train_concedes = xgboost.DMatrix(X_train, label=Y_train["concedes"], weight=concedes_weight)

    return dm_train_scores, dm_train_concedes


# In[121]:


def model_train_DMatrix(DMatrix, X_test):
    # input:Dmatrix(weight), X_test(features)
    print("barori")
    params = {
        "objective": "reg:squarederror",
        "silent": 1,
        "random_state": 1234,
        # 学習用の指標 (RMSE)
        "eval_metric": "rmse",
    }
    num_round = 500
    model = xgboost.train(params, DMatrix, num_round)
    dm_test = xgboost.DMatrix(X_test)
    predict = model.predict(dm_test)
    print("predict")

    return predict


# In[ ]:
