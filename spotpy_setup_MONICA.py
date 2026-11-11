import copy
from concurrent.futures import ThreadPoolExecutor
import csv
import json
import os
import re
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
import spotpy


class SpotSetup(object):
    """SPOTPY setup for Müncheberg V140 crop rotations."""

    BASE_PATH = Path(__file__).resolve().parent
    MONICA_EXECUTABLE = Path(r"C:\MONICA\monica_win64_3.7.3.3_fixes_and_time_under_anoxia_365d\bin\monica-run.exe")
    MONICA_PARAMETERS = Path(os.environ.get("MONICA_PARAMETERS", r"C:\MONICA\monica_win64_3.7.3.3_fixes_and_time_under_anoxia_365d\monica-parameters_update_3.7.3.3"))
    OUTPUT_DIR = BASE_PATH / "calib_out"
    PARALLEL_RUNS = 4  # Number of plots to run at once

    def __init__(self, user_params: pd.DataFrame, exp_maps: pd.DataFrame, obslist: pd.DataFrame):
        """
        Constructor of a spotpy setup for MONICA calibration

        :param user_params: these are the model parameter we will sample through the calibration process
        :param exp_maps: these are the experiments we want to calibrate for
        :param obslist: these are the observations we want to compare the model output to, like the real measurements
        """

        self.cnt_calls = 0

        self.experiments = exp_maps

        # Uncomment parameters to include them in calibration.
        enabled_parameters = [
            # "StageTemperatureSum",
            # "BaseDaylength",
            # "DaylengthRequirement",
            # "VernalisationRequirement",  # only for winter crops

            # "StageKcFactor",
            # "CropSpecificMaxRootingDepth",
            # "RootPenetrationRate",
            # "SpecificRootLength",
            "DroughtStressThreshold",

            # "MaxAssimilationRate",
            # "SpecificLeafArea",
        ]

        self.user_params = user_params[user_params["name"].isin(enabled_parameters)]

        self.params = []
        for _, row in self.user_params.iterrows():
            parname = row["name"]
            if pd.notna(row["position in array"]):  # Check if position in array is not empty
                parname += "_" + str(int(row["position in array"]))  # spotpy does not allow two parameters to have the same name

            if pd.isna(row["derived params"]):
                self.params.append(spotpy.parameter.Uniform(
                    parname,
                    row["low"],
                    row["high"],
                    row["stepsize"],
                    # row["optguess"],
                    # row["minbound"],
                    # row["maxbound"]
                ))

        self.env_templates = self._create_env()  # create the environment.json for each experimen

        self.best_rmse = np.inf
        self.best_simulation = None

        # Sort measured yields by experiment ID and year so they match the simulated yields.
        matches = []
        for env in self.env_templates:
            selected = obslist[
                obslist["plot_no"].eq(env["plot_no"])
                & obslist["treatment"].eq(env["treatment"])
                & obslist["crop"].eq(env["measured_crop"])
                & obslist["year"].isin(env["harvest_years"])
                & obslist["yield_dm_kg_ha"].notna()
            ].copy()
            selected.insert(0, "exp_ID", env["exp_ID"])
            matches.append(selected)
        self.observation_table = pd.concat(matches, ignore_index=True).sort_values(
            ["exp_ID", "year"]).reset_index(drop=True)
        self.observations = pd.to_numeric(
            self.observation_table["yield_dm_kg_ha"], errors="raise"
        ).to_numpy(dtype=float).tolist()

    def parameters(self):
        return spotpy.parameter.generate(self.params)

    def simulation(self, vector):
        self.cnt_calls += 1
        self.out = {}
        environment = os.environ.copy()
        environment["MONICA_PARAMETERS"] = str(Path(self.MONICA_PARAMETERS).resolve())
        print(f"Call {self.cnt_calls}: running {len(self.env_templates)} experiments", flush=True)

        # Run independent plots together, then combine their yields.
        with ThreadPoolExecutor(max_workers=self.PARALLEL_RUNS) as executor:
            futures = [
                executor.submit(self._run_experiment, env, vector, environment)
                for env in self.env_templates
            ]
            for future in futures:
                self.out.update(future.result())

        simulations = []
        for row in self.observation_table.itertuples(index=False):
            key = (row.exp_ID, row.year)
            if key not in self.out or not np.isfinite(self.out[key]):
                raise ValueError(f"Missing or invalid simulated yield for {key}")
            simulations.append(self.out[key])

        rmse = self.objectivefunction(simulations, self.observations)
        if rmse < self.best_rmse:
            self.best_rmse = float(rmse)
            self.best_simulation = simulations.copy()
        return simulations

    def _run_experiment(self, env, vector, environment):
        """Run one plot with its own crop settings and temporary files."""
        results = {}
        current_env = copy.deepcopy(env)
        crop_params = self._set_parameters(current_env["crop_params"], vector)
        # Only replace sowings at the positions selected in the experiment map.
        crop = current_env["crop"]
        for position in current_env["positions"]:
            for step in crop["cropRotation"][position]["worksteps"]:
                if step["type"] == "Sowing":
                    step["crop"] = copy.deepcopy(crop_params)

        # Run MONICA using temporary files, read the simulated yields, then automatically delete the files.
        with TemporaryDirectory() as working_dir:
            directory = Path(working_dir)
            crop_path = directory / "crop.json"
            sim_path = directory / "sim.json"
            result_path = directory / "harvest.csv"
            self._write_json(crop_path, crop)
            sim = current_env["sim"]
            sim["crop.json"] = crop_path.as_posix()
            self._write_json(sim_path, sim)
            subprocess.run(
                [str(self.MONICA_EXECUTABLE), "-o", str(result_path), str(sim_path)],
                cwd=self.BASE_PATH,
                env=environment,
                check=True,
            )
            harvests = self._read_harvests(result_path)
        for year in env["harvest_years"]:
            key = (year, env["simulated_crop"])
            if key in harvests:
                results[env["exp_ID"], year] = harvests[key]

        return results

    def evaluation(self):
        return self.observations

    def objectivefunction(self, simulation, evaluation):
        """
        Spotpy uses this function to calculate the error between the simulation and the evaluation data
        :param simulation:
        :param evaluation:
        :return:
        """
        objectivefunction = spotpy.objectivefunctions.rmse(evaluation, simulation)  # We are using RMSE as objective function.
        return objectivefunction

    def _create_env(self):
        """Read the existing sim/crop/site files named by each experiment-map row."""
        templates = []
        for _, exp_row in self.experiments.iterrows():
            # The map identifies plot and treatment through exp_ID, e.g. 12007_10.
            match = re.fullmatch(r"([0-9]+)_([0-9]+)", str(exp_row["exp_ID"]))
            plot_no, treatment = map(int, match.groups())
            sim_path = self.BASE_PATH / exp_row["sim file"]
            sim = self._read_json(sim_path)
            crop = self._read_json(self.BASE_PATH / exp_row["crop file"])
            site_path = (self.BASE_PATH / exp_row["site file"]).resolve()
            climate_path = (self.BASE_PATH / exp_row["climate file"]).resolve()
            crop_id = exp_row["crop ID (as in crop.json)"]
            crop_params = copy.deepcopy(crop["crops"][crop_id])
            for section, column in (("species", "species params"), ("cultivar", "cv params")):
                declared = exp_row[column]
                if pd.notna(declared) and str(declared).strip():
                    definition = ["include-from-file", "crops/" + str(declared)]
                else:
                    definition = crop_params["cropParams"][section]
                crop_params["cropParams"][section] = self._load_parameters(definition)

            raw_positions = exp_row["position in rotation *"]
            positions = [int(p) for p in str(raw_positions).split("-")]
            harvest_years = set()
            for position in positions:
                steps = crop["cropRotation"][position]["worksteps"]
                for step in steps:
                    if step["type"] == "Harvest":
                        options = sim["climate.csv-options"]
                        if options["start-date"] <= step["date"] <= options["end-date"]:
                            year = int(step["date"][:4])
                            harvest_years.add(year)
            sim["site.json"] = site_path.as_posix()
            sim["climate.csv"] = climate_path.as_posix()
            sim["include-file-base-path"] = Path(self.MONICA_PARAMETERS).resolve().as_posix() + "/"

            # Save only harvest results during calibration to avoid writing daily outputs.
            sim["output"]["events"] = [
                "crop", ["Crop", ["Year|harv-year", "LAST"], ["Yield", "LAST"]]
            ]

            species = crop_params["cropParams"]["species"]["SpeciesName"]
            cultivar = crop_params["cropParams"]["cultivar"]["CultivarName"]
            templates.append({
                "exp_ID": exp_row["exp_ID"],
                "sim": sim, "crop": crop, "crop_params": crop_params,
                "positions": positions, "harvest_years": harvest_years,
                "plot_no": plot_no, "treatment": treatment,
                "measured_crop": cultivar, "simulated_crop": f"{species}/{cultivar}",
            })
        return templates

    def _set_parameters(self, crop, vector):
        for (_, row), value in zip(self.user_params.iterrows(), vector):
            name = row["name"]
            owners = [crop["cropParams"][s] for s in ("species", "cultivar")
                      if name in crop["cropParams"][s]]
            owner = owners[0]
            original = owner[name]
            if name == "SpecificLeafArea":
                factor = float(value)
                if isinstance(original[0], list):
                    original[0] = [float(v) * factor for v in original[0]]  # [[stage values], "unit"]
                else:
                    owner[name] = [float(v) * factor for v in original]  # [stage values]
                continue
            if pd.notna(row["position in array"]):
                array = original[0] if isinstance(original[0], list) else original
                index = int(row["position in array"]) - 1
                array[index] = float(value)
            elif isinstance(original, list) and len(original) == 2 and isinstance(original[1], str):
                original[0] = float(value)
            else:
                owner[name] = float(value)
        return crop

    def _load_parameters(self, value):
        if isinstance(value, list) and value[0] == "include-from-file":
            return self._load_parameters(self._read_json(Path(self.MONICA_PARAMETERS) / value[1]))
        if isinstance(value, dict):
            result = self._load_parameters(value["="]) if "=" in value else {}
            result.update({k: copy.deepcopy(v) for k, v in value.items() if k != "="})
            return result
        raise ValueError(f"Unsupported crop parameter definition: {value}")

    @staticmethod
    def _read_json(path):
        with Path(path).open(encoding="utf-8-sig") as source:
            return json.load(source)

    @staticmethod
    def _write_json(path, value):
        with Path(path).open("w", encoding="utf-8") as target:
            json.dump(value, target, indent=2, ensure_ascii=False)

    @staticmethod
    def _read_harvests(path):
        harvests = {}
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows = csv.reader(source, delimiter=";")
            for row in rows:
                if row == ["crop"]:
                    break
            else:
                raise ValueError(f"No crop output section in {path}")
            header = next(rows)
            for row in rows:
                if not row or row == ["daily"]:
                    break
                record = dict(zip(header, row))
                if not record.get("harv-year", "").isdigit():
                    continue  # skip units and aggregation metadata
                key = (int(record["harv-year"]), record["Crop"])
                harvests[key] = float(record["Yield"])
        return harvests
