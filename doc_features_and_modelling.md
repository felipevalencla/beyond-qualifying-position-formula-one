## Summary of Engineered Features

This section outlines all the new features created throughout the notebook, detailing their definitions and how they are computed. These features are crucial for both the predictive models and for understanding driver/team performance dynamics.

### Core Race Outcome Features

-   **`classification_points`**: The points awarded to a driver based on their `race_finishing_position`, using the FIA's 2025 points structure (25 for P1, 18 for P2, etc., down to 1 for P10, 0 otherwise). This ensures consistent scoring rules across the entire dataset, removing the fastest lap bonus point.
-   **`Top10`**: A binary indicator (`1` if `race_finishing_position <= 10`, else `0`).
-   **`Podium`**: A binary indicator (`1` if `race_finishing_position <= 3`, else `0`).
-   **`IncidentDNF`**: A binary indicator (`1` if the driver's `status` contains 'Accident', 'Collision', or 'Spun off', else `0`).

### Driver-Specific Rolling Features (Conventional Baselines)

These features capture a driver's recent performance based on their past races, always excluding the current race's data. Calculations are grouped by `DriverId`.

-   **`driver_points_mean_last3`**: The rolling mean of `classification_points` over the last 3 races for a given `DriverId`, shifted by 1 to represent information *before* the current race.
-   **`driver_qualifying_mean_last3`**: The rolling mean of `qualifying_position` over the last 3 races for a given `DriverId`, shifted by 1.
-   **`driver_incident_rate_last10`**: The rolling incident rate, calculated as the sum of `IncidentDNF` over the last 10 races divided by the count of races in that window for a `DriverId`, shifted by 1.
-   **`driver_previous_starts`**: The cumulative count of previous race starts for each `DriverId`, indicating their career experience up to the current race.

### Team-Specific Rolling Features (Conventional Baselines)

These features reflect a team's recent performance. First, an average is taken for the team's drivers in a given race, then rolling means are computed.

-   **`mean_team_classification_points_per_race`**: The mean `classification_points` scored by all drivers of a `TeamId` in a specific `season` and `round`.
-   **`mean_team_qualifying_position_per_race`**: The mean `qualifying_position` of all drivers of a `TeamId` in a specific `season` and `round`.
-   **`team_points_mean_last3`**: The rolling mean of `mean_team_classification_points_per_race` over the last 3 distinct races for a given `TeamId`, shifted by 1. Calculated on unique team-race events to avoid double-counting.
-   **`team_qualifying_mean_last3`**: The rolling mean of `mean_team_qualifying_position_per_race` over the last 3 distinct races for a given `TeamId`, shifted by 1. Calculated on unique team-race events.

### Season-to-Date Averages

These features provide an average of driver's and team's performance *within the current season* up to the previous race.

-   **`driver_season_to_date_qualifying_average`**: The expanding mean of `qualifying_position` for a `DriverId` within a `season`, shifted by 1 to exclude the current race.
-   **`team_season_to_date_qualifying_average`**: The expanding mean of `mean_team_qualifying_position_per_race` for a `TeamId` within a `season`, shifted by 1.
-   **`driver_season_to_date_classification_points_average`**: The expanding mean of `classification_points` for a `DriverId` within a `season`, shifted by 1 to exclude the current race.
-   **`team_season_to_date_classification_points_average`**: The expanding mean of `mean_team_classification_points_per_race` for a `TeamId` within a `season`, shifted by 1.

### Predicted Features (from Auxiliary Models)

These features are the outputs of the two auxiliary predictive models.

-   **`previous_qualifying_position`**: The `qualifying_position` from the immediately preceding race for each `DriverId`, shifted by 1.
-   **`previous_classification_points`**: The `classification_points` from the immediately preceding race for each `DriverId`, shifted by 1. This feature will be `NaN` for Round 1 races, and its inclusion in the points model's feature set will cause these races to be excluded from prediction, resulting in `NaN` predictions for them.
-   **`predicted_qualifying_score`**: The raw continuous prediction from the CatBoost Regressor for a driver's expected qualifying performance.
-   **`predicted_qualifying_position`**: The rank (P1-P20) derived from `predicted_qualifying_score` within each race. Lower scores indicate better positions.
-   **`qualifying_surprise`**: Computed as `predicted_qualifying_position - qualifying_position`. A positive value indicates qualifying better than expected, a negative value worse.
-   **`positive_qualifying_surprise`**: `max(qualifying_surprise, 0)`. Captures only instances of exceeding qualifying expectations.
-   **`negative_qualifying_surprise`**: `max(-qualifying_surprise, 0)`. Captures only instances of underperforming qualifying expectations.
-   **`predicted_classification_points_raw`**: The raw continuous prediction from the CatBoost Regressor for a driver's expected `classification_points`, constrained to `[0, 25]`.
-   **`predicted_classification_points`**: The discrete F1 points (0, 1, 2, ..., 25) derived by ranking `predicted_classification_points_raw` within each race and mapping them to the FIA 2025 points structure.

### Race Dynamics Features (Exposure to Gain/Loss)

These features quantify a driver's recent history of gaining or losing positions/points during races, relative to their expected performance.

-   **`race_gain`**: `max(classification_points - predicted_classification_points, 0)`. Represents how many points a driver gained during a race relative to their *predicted* points for that race. A positive value means outperforming expectations.
-   **`race_loss`**: `max(predicted_classification_points - classification_points, 0)`. Represents how many points a driver lost during a race relative to their *predicted* points. A positive value means underperforming expectations.
-   **`recent_gain_exposure`**: Rolling sum of `race_gain` over the last 3 races for a `DriverId` within a `season`, shifted by 1. Filled with 0 for initial races with no prior exposure.
-   **`recent_loss_exposure`**: Rolling sum of `race_loss` over the last 3 races for a `DriverId` within a `season`, shifted by 1. Filled with 0 for initial races with no prior exposure.
-   **`exposure_history_count`**: Number of races contributing to the `recent_gain_exposure` and `recent_loss_exposure` calculations (up to 3, adjusted for season boundaries and early career). Filled with 0 for initial races.

-----------------------------------------------------------------

## Summary of Predictive Models

Two auxiliary predictive models were developed using `CatBoostRegressor` to generate expected qualifying positions and expected classification points. Both models employ a rigorous walk-forward validation strategy to simulate real-world prediction scenarios, ensuring continuous learning and adaptation.

### 1. Expected Qualifying Position Model

-   **Objective**: To predict a driver's `qualifying_position` for a given race.
-   **Model Type**: `CatBoostRegressor`.
-   **Target Variable**: `qualifying_position`.
-   **Features Used**: A comprehensive set of conventional and rolling features, including:
    -   `driver_qualifying_mean_last3`
    -   `team_qualifying_mean_last3`
    -   `previous_qualifying_position`
    -   `driver_season_to_date_qualifying_average`
    -   `team_season_to_date_qualifying_average`
    -   `driver_previous_starts`
    -   `DriverId` (categorical)
    -   `TeamId` (categorical)
    -   `location` (categorical)
    -   `season`
    -   `round`

-   **Modeling Strategy (Walk-Forward Prediction)**:
    1.  **Initial Training (2018 Data)**: The model was initially trained on data from the `2018` season.
    2.  **Hyperparameter Tuning (2019 Data)**: `GridSearchCV` was performed on the `2018` training data, with validation on `2019` data, to select optimal hyperparameters (`depth`, `learning_rate`, `l2_leaf_reg`). The best hyperparameters found were `{'depth': 4, 'l2_leaf_reg': 10, 'learning_rate': 0.05}`. These hyperparameters are then **frozen** for the subsequent walk-forward prediction phase.
    3.  **Walk-Forward Prediction (2020 Onwards - Race-by-Race Retraining)**: From the `2020` season onwards, the model performs a sequential, race-by-race prediction:
        -   **For each individual race**, the model is **re-trained** using an expanding window of historical data. This means it incorporates *all* race outcomes from before 2020, *including the 2019 validation data*, plus the actual outcomes of all preceding races within the 2020-2025 period up to the current race.
        -   Predictions (`predicted_qualifying_score`) are generated for all drivers in the current race *before* its actual outcome is known.
        -   These raw scores are then ranked within the race to assign discrete `predicted_qualifying_position` (P1-P20).
        -   **Crucially, after the predictions are made, the actual outcome of the current race is then added to the historical training data**, making it available for retraining the model for the *next* race. This simulates a real-world scenario where the model continuously learns and adapts with new information.

-   **Key Considerations**:
    -   The `predicted_qualifying_score` is a continuous value, while `predicted_qualifying_position` is an integer rank.
    -   Predictions are not generated for races before `PREDICT_START_YEAR` (2020) as these years are used for initial training and validation.

-   **Evaluation**: The model was evaluated using Mean Absolute Error (MAE), Spearman Rank Correlation, and Percentage of predictions within 3 positions. The CatBoost Regressor showed the lowest overall MAE (3.024) compared to simpler benchmarks, justifying its use.

### 2. Expected Classification Points Model

-   **Objective**: To predict a driver's `classification_points` for a given race.
-   **Model Type**: `CatBoostRegressor`.
-   **Target Variable**: `classification_points`.
-   **Features Used**: Similar to the qualifying model, but tailored for points prediction. It explicitly excludes features that might introduce information about qualifying surprise, as per the design:
    -   `qualifying_position`
    -   `driver_points_mean_last3`
    -   `team_points_mean_last3`
    -   `driver_qualifying_mean_last3`
    -   `team_qualifying_mean_last3`
    -   `driver_incident_rate_last10`
    -   `driver_previous_starts`
    -   `previous_classification_points`
    -   `driver_season_to_date_classification_points_average`
    -   `team_season_to_date_classification_points_average`
    -   `DriverId` (categorical)
    -   `TeamId` (categorical)
    -   `location` (categorical)
    -   `season`
    -   `round`

-   **Modeling Strategy (Walk-Forward Prediction)**:
    1.  **Walk-Forward Prediction (2020 Onwards - Race-by-Race Retraining)**: The same walk-forward logic as the qualifying model is applied:
        -   **For each individual race**, the model is **re-trained** using an expanding window of historical data (all data up to the current race's start, including previously completed races from 2020 onwards, and explicitly incorporating the 2019 validation data).
        -   Raw continuous predictions (`predicted_classification_points_raw`) are generated, constrained to the `[0, 25]` range.
        -   These raw scores are then ranked within the race, and discrete `predicted_classification_points` (F1 points) are assigned based on the FIA 2025 points structure (25 for P1, 18 for P2, etc., down to 1 for P10). This ranking is performed *within each specific race*.
        -   **After predictions are made, the actual outcome of the current race is then added to the historical training data** for the subsequent iteration, allowing the model to continuously update its knowledge.

-   **Key Considerations**:
    -   `RMSE` was used as the loss function during training, with `MAE` monitored.
    -   Raw predictions were clipped to be between 0 and 25 points to reflect the physical bounds of points in F1.
    -   Predictions (both raw and discrete) retain `NaN` values where no prediction could be made (e.g., for races before the `PREDICT_START_YEAR`). This was a specific user request to represent missing data accurately, rather than filling with zeros.

-   **Evaluation**: The model was evaluated using MAE and RMSE. The `CatBoost Regressor (Discrete)` model achieved the best overall MAE (2.864), demonstrating superior performance over the benchmarks.