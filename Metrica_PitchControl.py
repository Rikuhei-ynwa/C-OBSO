#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 14:52:19 2020

Module for calculating a Pitch Control surface using MetricaSports's tracking & event data.

Pitch control (at a given location on the field) is the probability that a team will gain
possession if the ball is moved to that location on the field.

Methdology is described in "Off the ball scoring opportunities" by William Spearman:
http://www.sloansportsconference.com/wp-content/uploads/2018/02/2002.pdf

GitHub repo for this code can be found here:
https://github.com/Friends-of-Tracking-Data-FoTD/LaurieOnTracking

Data can be found at: https://github.com/metrica-sports/sample-data

Functions
----------

calculate_pitch_control_at_target(): calculate the pitch control probability for the attacking and defending teams at a specified target position on the ball.

generate_pitch_control_for_event(): this function evaluates pitch control surface over the entire field at the moment
of the given event (determined by the index of the event passed as an input)

Classes
---------

The 'player' class collects and stores trajectory information for each player required by the pitch control calculations.

@author: Laurie Shaw (@EightyFivePoint)

"""

import numpy as np
import pandas as pd
import SPADL_config as spadlconfig


def initialise_players(team: pd.Series, teamname: str, params: dict, GKid: str) -> list:
    """
    initialise_players(team,teamname,params)

    create a list of player objects that holds their positions and velocities from the tracking data dataframe

    Parameters
    -----------

    team: row (i.e. instant) of either the home or away team tracking Series
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    GKid: a numeric number by type str regarding team name "Home" or "Away"

    Returns
    -----------

    team_players: list of player objects for the team at at given instant

    """

    # get player  ids
    player_ids = np.unique([c.split("_")[1] for c in team.keys() if c[:4] == teamname])
    player_ids = sorted(player_ids.tolist(), key=spadlconfig.natural_keys)

    # create list
    team_players = []
    for player_id in player_ids:
        # create a player object for player_id 'player_id'
        team_player = player(player_id, team, teamname, params, GKid)
        if team_player.inframe:
            team_players.append(team_player)
    return team_players


def check_offsides(
    attacking_players: list,
    defending_players: list,
    ball_position: np.ndarray,
    GK_numbers: list,
    verbose=False,
    tol=0.2,
) -> list:
    """
    check_offsides( attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):

    checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from
    the 'attacking_players' list and ignored in the pitch control calculation.

    Parameters
    -----------
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        verbose: if True, print a message each time a player is found to be offside
        tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m

    Returns
    -----------
        attacking_players: list of 'player' objects for the players on the attacking team with offside players removed
    """

    # find jersey number of defending goalkeeper (just to establish attack direction)
    defending_GK_id = GK_numbers[1] if attacking_players[0].teamname == "Home" else GK_numbers[0]

    # make sure defending goalkeeper is actually on the field!
    assert defending_GK_id in [
        player.id for player in defending_players
    ], "Defending goalkeeper jersey number not found in defending players"

    # get goalkeeper player object
    defending_GK = [player for player in defending_players if player.id == defending_GK_id][0]

    # use defending goalkeeper x position to figure out which half he is defending (-1: left goal, +1: right goal)
    defending_half = np.sign(defending_GK.position[0])

    # find the x-position of the second-deepest defeending player (including GK)
    second_deepest_defender_x = sorted(
        [defending_half * player.position[0] for player in defending_players],
        reverse=True,
    )[1]

    # define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
    offside_line = max(second_deepest_defender_x, defending_half * ball_position[0], 0.0) + tol

    # any attacking players with x-position greater than the offside line are offside
    if verbose:
        for player in attacking_players:
            if player.position[0] * defending_half > offside_line:
                print("player %s in %s team is offside" % (player.id, player.playername))
    attacking_players = [player for player in attacking_players if player.position[0] * defending_half <= offside_line]
    return attacking_players


class player(object):
    """
    player() class

    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player

    __init__ Parameters
    -----------
    player_id: id (jersey number) of player by str
    team: row (i.e. instant) of either the home or away team tracking Series
    teamname: team name "Home" or "Away"
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    GKid: a numeric number by type str regarding team name "Home" or "Away"


    methods include:
    -----------
    simple_time_to_intercept(r_final_pos): time take for player to get to target position (r_final_pos) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept

    """

    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self, player_id: str, team: pd.Series, teamname: str, params: dict, GKid: str):
        self.id = player_id
        self.is_gk = self.id == GKid
        self.teamname = teamname
        self.playername = f"{teamname}_{player_id}_"
        self.vmax = params["max_player_speed"]  # player max speed in m/s. Could be individualised
        self.reaction_time = params["reaction_time"]  # player reaction time in 's'. Could be individualised
        self.tti_sigma = params["tti_sigma"]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params["lambda_att"]  # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = (
            params["lambda_gk"] if self.is_gk else params["lambda_def"]
        )  # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.get_position(team)
        self.get_velocity(team)
        self.PPCF = 0.0  # initialise this for later

    def get_position(self, team: pd.Series):
        self.position = np.array([team[self.playername + "x"], team[self.playername + "y"]])
        self.inframe = not np.any(np.isnan(self.position))

    def get_velocity(self, team: pd.Series):
        self.velocity = np.array([team[self.playername + "vx"], team[self.playername + "vy"]])
        if np.any(np.isnan(self.velocity)):
            self.velocity = np.array([0.0, 0.0])

    def simple_time_to_intercept(self, r_final_pos: np.ndarray):
        self.PPCF = 0.0  # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.

        r_reaction_pos = self.position + self.velocity * self.reaction_time
        movement_time = np.linalg.norm(r_final_pos - r_reaction_pos) / self.vmax
        self.time_to_intercept = self.reaction_time + movement_time

        return self.time_to_intercept

    def probability_intercept_ball(self, T: float):
        # probability of a player arriving at target location at time 'T' given
        # their expected time_to_intercept (time of arrival), as described in Spearman 2018.
        # this function is similar to sigmoid function.

        input_to_function = np.pi * ((T - self.time_to_intercept) / (np.sqrt(3.0) * self.tti_sigma))
        f = 1 / (1.0 + np.exp(-input_to_function))

        return f


""" Generate pitch control map """


def default_model_params(time_to_control_veto=3) -> dict:
    """
    default_model_params()

    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.

    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control
    it is less than 10^-time_to_control_veto, ignore that player.


    Returns
    -----------

    params: dictionary of parameters required to determine and calculate the model

    """
    # key parameters for the model, as described in Spearman 2018
    params = {}

    # model parameters
    # maximum player acceleration m/s/s, not used in this implementation
    params["max_player_accel"] = 7.0

    # maximum player speed m/s
    params["max_player_speed"] = 5.0

    # seconds taken for player to react and change trajectory. Roughly determined as vmax/amax
    params["reaction_time"] = 0.7

    # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params["tti_sigma"] = 0.45

    # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball,
    # I have set to 1 so that home & away players have same ball control probability
    params["kappa_def"] = 1.0

    # ball control parameter for attacking team
    params["lambda_att"] = 4.3

    # ball control parameter for defending team
    params["lambda_def"] = 4.3 * params["kappa_def"]

    # make goal keepers must quicker to control ball (because they can catch it)
    params["lambda_gk"] = params["lambda_def"] * 3.0

    # average ball travel speed in m/s
    params["average_ball_speed"] = 15.0

    # numerical parameters for model evaluation
    # integration timestep (dt)
    params["int_dt"] = 0.04

    # upper limit on integral time
    params["max_int_time"] = 10

    # assume convergence when PPCF>0.99 at a given location.
    params["model_converge_tol"] = 0.01

    # The following are 'short-cut' parameters. We do not need to calculate PPCF explicitly
    # when a player has a sufficient head start. A sufficient head start is when a player
    # arrives at the target location at least 'time_to_control' seconds before the next player.
    params["time_to_control_att"] = (
        time_to_control_veto * np.log(10) * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_att"])
    )
    params["time_to_control_def"] = (
        time_to_control_veto * np.log(10) * (np.sqrt(3) * params["tti_sigma"] / np.pi + 1 / params["lambda_def"])
    )
    return params


def generate_pitch_control_for_event(
    event_id,
    events,
    tracking_home_df: pd.DataFrame,
    tracking_away_df: pd.DataFrame,
    params: dict,
    GK_numbers: list,
    field_dimen=(
        106.0,
        68.0,
    ),
    n_grid_cells_x=50,
    offsides=True,
):
    """generate_pitch_control_for_event

    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home_df: tracking DataFrame for the Home team
        tracking_away_df: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.

    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')

    Returrns
    -----------
        PPCF_attack: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCF_attack.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    # get the details of the event (frame, team in possession, ball_start_position)
    pass_frame = events.loc[event_id]["Start Frame"]
    pass_team = events.loc[event_id].Team
    ball_start_pos = np.array([events.loc[event_id]["Start X"], events.loc[event_id]["Start Y"]])
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2.0 + dy / 2.0
    # initialise pitch control grids for attacking and defending teams
    PPCF_attack = np.zeros(shape=(len(ygrid), len(xgrid)))
    PPCF_defense = np.zeros(shape=(len(ygrid), len(xgrid)))
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if pass_team == "Home":
        attacking_players = initialise_players(tracking_home_df.loc[pass_frame], "Home", params, GK_numbers[0])
        defending_players = initialise_players(tracking_away_df.loc[pass_frame], "Away", params, GK_numbers[1])
    elif pass_team == "Away":
        defending_players = initialise_players(tracking_home_df.loc[pass_frame], "Home", params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away_df.loc[pass_frame], "Away", params, GK_numbers[1])
    else:
        assert False, "Team in possession must be either home or away"

    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)
    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            PPCF_attack[i, j], PPCF_defense[i, j] = calculate_pitch_control_at_target(
                target_position,
                attacking_players,
                defending_players,
                ball_start_pos,
                params,
            )
    # check probabilitiy sums within convergence
    checksum = np.sum(PPCF_attack + PPCF_defense) / float(n_grid_cells_y * n_grid_cells_x)
    assert 1 - checksum < params["model_converge_tol"], "Checksum failed: %1.3f" % (1 - checksum)
    return PPCF_attack, xgrid, ygrid, attacking_players


def calculate_pitch_control_at_target(
    target_position: np.ndarray,
    attacking_players: list,
    defending_players: list,
    ball_start_pos: np.ndarray,
    params: dict,
):
    """calculate_pitch_control_at_target

    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.

    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )

    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )

    """

    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)):  # assume that ball is already at location
        ball_travel_time = 0.0
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_dist = np.linalg.norm(target_position - ball_start_pos)
        ball_travel_time = ball_travel_dist / params["average_ball_speed"]

    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin([player.simple_time_to_intercept(target_position) for player in attacking_players])
    tau_min_def = np.nanmin([player.simple_time_to_intercept(target_position) for player in defending_players])

    # check whether we actually need to solve equation 3
    if tau_min_att - max(ball_travel_time, tau_min_def) >= params["time_to_control_def"]:
        # if defending team can arrive significantly before attacking team,
        # no need to solve pitch control model
        return 0.0, 1.0
    elif tau_min_def - max(ball_travel_time, tau_min_att) >= params["time_to_control_att"]:
        # if attacking team can arrive significantly before defending team,
        # no need to solve pitch control model
        return 1.0, 0.0
    else:
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [
            player
            for player in attacking_players
            if player.time_to_intercept - tau_min_att < params["time_to_control_att"]
        ]
        defending_players = [
            player
            for player in defending_players
            if player.time_to_intercept - tau_min_def < params["time_to_control_def"]
        ]
        # set up integration arrays
        dT_array = np.arange(
            ball_travel_time - params["int_dt"],
            ball_travel_time + params["max_int_time"],
            params["int_dt"],
        )
        PPCFatt = np.zeros_like(dT_array)
        PPCFdef = np.zeros_like(dT_array)
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1 - ptot > params["model_converge_tol"] and i < dT_array.size:
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                PPCF_sum = PPCFatt[i - 1] + PPCFdef[i - 1]
                dPPCFdT = (1 - PPCF_sum) * player.probability_intercept_ball(T) * player.lambda_att

                # make sure it's greater than zero
                assert dPPCFdT >= 0, "Invalid attacking player probability (calculate_pitch_control_at_target)"

                # total contribution from individual player
                player.PPCF += dPPCFdT * params["int_dt"]

                # add to sum over players in the attacking team
                # (remembering array element is zero at the start of each integration iteration)
                PPCFatt[i] += player.PPCF

            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                PPCF_sum = PPCFatt[i - 1] + PPCFdef[i - 1]
                dPPCFdT = (1 - PPCF_sum) * player.probability_intercept_ball(T) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT >= 0, "Invalid defending player probability (calculate_pitch_control_at_target)"

                # total contribution from individual player
                player.PPCF += dPPCFdT * params["int_dt"]

                # add to sum over players in the defending team
                PPCFdef[i] += player.PPCF

            ptot = PPCFdef[i] + PPCFatt[i]  # total pitch control probability
            i += 1
        if i >= dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot))
        return PPCFatt[i - 1], PPCFdef[i - 1]


def generate_pitch_control_for_tracking(
    tracking_home_df: pd.DataFrame,
    tracking_away_df: pd.DataFrame,
    tracking_frame: int,
    attacking_team: str,
    params: dict,
    GK_numbers: list,
    field_dimen=(
        106.0,
        68.0,
    ),
    n_grid_cells_x=50,
    offsides=True,
):
    """generate_pitch_control_for_tracking

    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)

    Parameters
    -----------
        tracking_home_df: tracking DataFrame for the Home team
        tracking_away_df: tracking DataFrame for the Away team
        tracking_frame: tracking frame in type int
        attacking_team: Home or Away in ball possesion team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.


    Returrns
    -----------
        PPCF_attack: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team.
               Surface for the defending team is just 1-PPCF_attack.
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)

    """
    # get the details of the event (frame, team in possession, ball_start_position)
    # pass_frame = events.loc[event_id]['Start Frame']
    # pass_team = events.loc[event_id].Team
    ball_start_pos = np.array(
        [
            tracking_home_df.loc[tracking_frame]["ball_x"],
            tracking_home_df.loc[tracking_frame]["ball_y"],
        ]
    )

    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x * field_dimen[1] / field_dimen[0])
    dx = field_dimen[0] / n_grid_cells_x
    dy = field_dimen[1] / n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x) * dx - field_dimen[0] / 2.0 + dx / 2.0
    ygrid = np.arange(n_grid_cells_y) * dy - field_dimen[1] / 2.0 + dy / 2.0

    # initialise pitch control grids for attacking and defending teams
    PPCF_attack = np.zeros(shape=(len(ygrid), len(xgrid)))
    PPCF_defense = np.zeros(shape=(len(ygrid), len(xgrid)))

    # initialise player positions and velocities for pitch control calc
    # (so that we're not repeating this at each grid cell position)
    if attacking_team == "Home":
        attacking_players = initialise_players(tracking_home_df.loc[tracking_frame], "Home", params, GK_numbers[0])
        defending_players = initialise_players(tracking_away_df.loc[tracking_frame], "Away", params, GK_numbers[1])
    elif attacking_team == "Away":
        defending_players = initialise_players(tracking_home_df.loc[tracking_frame], "Home", params, GK_numbers[0])
        attacking_players = initialise_players(tracking_away_df.loc[tracking_frame], "Away", params, GK_numbers[1])
    else:
        assert False, "Team in possession must be either home or away"

    # find any attacking players that are offside and remove them from the pitch control calculation
    if offsides:
        attacking_players = check_offsides(attacking_players, defending_players, ball_start_pos, GK_numbers)
    # calculate pitch pitch control model at each location on the pitch
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array([xgrid[j], ygrid[i]])
            PPCF_attack[i, j], PPCF_defense[i, j] = calculate_pitch_control_at_target(
                target_position,
                attacking_players,
                defending_players,
                ball_start_pos,
                params,
            )
    # check probabilitiy sums within convergence
    checksum = np.sum(PPCF_attack + PPCF_defense) / float(n_grid_cells_y * n_grid_cells_x)
    assert 1 - checksum < params["model_converge_tol"], "Checksum failed: %1.3f" % (1 - checksum)
    return PPCF_attack, xgrid, ygrid, attacking_players
