import os
import time
from pathlib import Path

import spotpy
import spotpy_setup_MONICA
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Define experiments (=crops), parameters to calibrate, and repetitions.

crop_sim_site_MAP = "crop_sim_site_MAP_WW_mineral.csv"
calib_params_file = "calibratethese_ww_bio.csv"

# crop_sim_site_MAP = "crop_sim_site_MAP_SB_mineral.csv" 
# calib_params_file = "calibratethese_sb_bio.csv"

# crop_sim_site_MAP = "crop_sim_site_MAP_WR_mineral.csv"
# calib_params_file = "calibratethese_wr_bio.csv"

# crop_sim_site_MAP = "crop_sim_site_MAP_SM_mineral.csv"
# calib_params_file = "calibratethese_sm_bio.csv"

# crop_sim_site_MAP = "crop_sim_site_MAP_SW_mineral.csv" 
# calib_params_file = "calibratethese_sw_bio.csv"

rep = 5
show_plots = False
output_dir = spotpy_setup_MONICA.SpotSetup.OUTPUT_DIR


def add_identity(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)

    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])

    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


font = {'family': 'calibri',
        'weight': 'normal',
        'size': 18}


def make_lambda(excel):
    return lambda v, p: eval(excel)


def main():
    # read general settings
    basepath = Path(os.path.dirname(os.path.abspath(__file__)))
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_params_df = pd.read_csv(basepath / calib_params_file, delimiter=";")

    # Read list of selected experiments we want to run the calibration for.
    # Each map row is a plot simulation; each matched harvest yields one simulation column.
    crop_site_map_df = pd.read_csv(os.path.join(basepath, crop_sim_site_MAP), delimiter=",")

    # read observations for which the likelihood of parameter space is calculated
    # Read the observations from the measurements file.
    measurements_df = pd.read_csv(basepath / "yield_measured.csv")

    # Here, MONICA inputs and matched yield observations are prepared:
    # Arguments are: Parameters, Sites, Observations
    # spot_setup = spotpy_setup_MONICA.spot_setup(params, exp_maps, obslist)
    spot_setup = spotpy_setup_MONICA.SpotSetup(calib_params_df,
                                               crop_site_map_df,
                                               measurements_df)

    # the same as for example: spot_setup = spot_setup(spotpy.objectivefunctions.rmse)

    # Set up the sampler with the model above
    # sampler = spotpy.algorithms.mc(spot_setup, dbname=str(output_dir / "SCEUA_monica_results"), dbformat='csv')  # This is the original
    sampler = spotpy.algorithms.sceua(spot_setup, dbname=str(output_dir / "SCEUA_monica_results"), dbformat='csv')  # This is from agmip

    # Run the sampler to produce the parameter distribution and identify optimal parameters based on objective function
    sampler.sample(rep)


    def print_status_final(self, stream):
        print("\n*** Final SPOTPY summary ***", file=stream)
        print(
            "Total Duration: "
            + str(round((time.time() - self.starttime), 2))
            + " seconds"
            , file=stream)
        print("Total Repetitions:", self.rep, file=stream)

        if self.optimization_direction == "minimize":
            print("Minimal objective value: %g" % (self.objectivefunction_min), file=stream)
            print("Corresponding parameter setting:", file=stream)
            for i in range(self.parameters):
                text = "%s: %g" % (self.parnames[i], self.params_min[i])
                print(text, file=stream)

        if self.optimization_direction == "maximize":
            print("Maximal objective value: %g" % (self.objectivefunction_max), file=stream)
            print("Corresponding parameter setting:", file=stream)
            for i in range(self.parameters):
                text = "%s: %g" % (self.parnames[i], self.params_max[i])
                print(text, file=stream)

        if self.optimization_direction == "grid":
            print("Minimal objective value: %g" % (self.objectivefunction_min), file=stream)
            print("Corresponding parameter setting:", file=stream)
            for i in range(self.parameters):
                text = "%s: %g" % (self.parnames[i], self.params_min[i])
                print(text, file=stream)

            print("Maximal objective value: %g" % (self.objectivefunction_max), file=stream)
            print("Corresponding parameter setting:", file=stream)
            for i in range(self.parameters):
                text = "%s: %g" % (self.parnames[i], self.params_max[i])
                print(text, file=stream)

        print("******************************\n", file=stream)


    path_to_best_out_file = str(output_dir / "best.out")
    with open(path_to_best_out_file, "a") as _:
        print_status_final(sampler.status, _)


    # Extract the parameter samples from distribution
    results = spotpy.analyser.load_csv_results(str(output_dir / "SCEUA_monica_results"))

    fig = plt.figure(1, figsize=(9, 5))
    plt.plot(results['like1'], "r+")
    plt.ylabel('RMSE')
    plt.xlabel('Iteration')
    fig.savefig(str(output_dir / "SCEUA_objectivefunctiontrace.png"), dpi=300)

    best_simulation = spot_setup.best_simulation

    fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.scatter(spot_setup.evaluation(), best_simulation, c=".3")
    add_identity(ax, color='r', ls='--')
    plt.xlabel('Obs')
    plt.ylabel('Sim')
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    lims = [min(xlims[0], ylims[0]), max(xlims[1], ylims[1])]
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect('equal', adjustable='box')
    fig.savefig(str(output_dir / "SCEUA_best_modelrun.png"), dpi=300)

    # fig = plt.figure(figsize=(16, 9))
    # ax = plt.subplot(1, 1, 1)
    # ax.scatter(spot_setup.evaluation(), best_simulation, c=".3")
    # add_identity(ax, color='r', ls='--')
    # coefficients = np.polyfit(spot_setup.evaluation(), best_simulation, 1)
    # trendline = np.poly1d(coefficients)
    # plt.plot(spot_setup.evaluation(), trendline(spot_setup.evaluation()), color="blue")
    # plt.xlabel("Obs")
    # plt.ylabel("Sim")
    # plt.savefig(str(output_dir / "SCEUA_best_modelrun_with_trendline.png"), dpi=300)

    # Add observed data to the calibration results
    results_file = str(output_dir / "SCEUA_monica_results.csv")
    updated_results_file = str(output_dir / "SCEUA_monica_results_with_observed.csv")

    results_df = pd.read_csv(results_file)

    # Extract observed data from spot_setup.evaluation()
    observed_data = spot_setup.evaluation()

    # Get all simulation columns
    simulation_columns = [col for col in results_df.columns if col.startswith("simulation")]

    # Ensure observed data matches the number of simulations
    if len(observed_data) != len(simulation_columns):
        raise ValueError(f"Mismatch: Length of observed data ({len(observed_data)}) does not match no. of simulation columns "
                         f"({len(simulation_columns)})!")

    # Repeat observed data for each row in results_df
    num_rows = len(results_df)
    observed_data_repeated = np.tile(observed_data, (num_rows, 1))

    # Create a DataFrame for observed data
    observed_df = pd.DataFrame(
        observed_data_repeated,
        columns=[f"observed_{i}" for i in range(len(observed_data))]
    )

    # Concatenate the observed data to the results DataFrame
    results_df = pd.concat([results_df, observed_df], axis=1)

    results_df.to_csv(updated_results_file, index=False)

    if show_plots:
        plt.show()
    else:
        plt.close("all")


if __name__ == "__main__":
    main()
