# 🏎️ F1 Race Outcome Predictor – 2025 Season

An ML-powered prediction system for Formula 1 race outcomes, built across the 2025 season. Uses real FastF1 telemetry, live OpenWeatherMap forecasts, and constructor performance analytics to train a Gradient Boosting model that predicts lap times and race finishing order for each Grand Prix.

Built as a passion project by an F1 fan who wanted to apply data science to a domain where I genuinely understand the variables — clean air race pace, wet-weather performance, circuit-specific position change patterns, and the impact of constructor strength.

Hope you find it interesting!


## 💡 Project Overview

For each Grand Prix in the 2025 F1 season, this project:
- Gathers sector-wise lap timing data using FastF1
- Integrates clean air race pace and qualifying performance
- Retrieves weather forecasts for the race location
- Factors in team performance and historical driver trends
- Trains a machine learning model to **predict race times**
- Outputs predicted podium finishers with performance plots

## 🔍 Key Features
- **FastF1 Integration**: Access to official session telemetry and lap data
- **Clean Air Race Pace Modeling**: Uses historical stint data to approximate performance without traffic
- **Qualifying-Based Prediction**: Uses qualifying time with race pace trends for performance estimation
- **Weather-Aware Predictions**: Adjusts based on rain probability and temperature
- **Team Scoring System**: Reflects relative constructor strength from current standings
- **ML-Driven Forecasts**: Gradient Boosting Regressor to predict lap times and overall outcome

## 🧠 Technologies Used

- **Python 3**
- **FastF1** – F1 telemetry and session data
- **pandas / numpy** – Data manipulation
- **scikit-learn** – Regression modeling (GradientBoostingRegressor)
- **matplotlib** – Data visualization
- **OpenWeatherMap API** – Weather forecasting
- **SimpleImputer** – Handling missing values


> Each script is modular and can run independently to predict a specific Grand Prix using up-to-date inputs.

Includes visualisations:

- Clean air race pace vs predicted time
- Feature importance (qualifying, weather, team, etc.)

## 🚀 Future Improvements

- Automate clean-air stint detection from full race data
- Include tire strategy and pit stop modeling
- Build a Streamlit or Gradio interface for live predictions (maybe something similar to f1-dash in the future>)
- Explore classification models for podium/points range prediction
