# Student Placement Predictor

This project predicts whether a student will get placed based on their CGPA and IQ using a machine learning model. The project includes a trained model, a Streamlit application for prediction, and is fully containerized using Docker.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Docker Setup](#docker-setup)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Student Placement Predictor uses a machine learning model trained on student data (CGPA and IQ) to predict their placement outcomes. The model is deployed in a Streamlit app where users can input their CGPA and IQ to receive a placement prediction.

## Features

- Predicts placement outcomes based on CGPA and IQ.
- Uses a RandomForestClassifier model with hyperparameter tuning.
- Streamlit-based web app for user interaction.
- Fully containerized with Docker for easy deployment.

## Installation

### Prerequisites

- Python 3.9 or higher
- Docker (if you wish to run the application in a container)

### Clone the Repository

```
git clone https://github.com/yourusername/student-placement-predictor.git
cd student-placement-predictor

```

### Install Dependencies

You can install the necessary Python packages using the requirements.txt file:

```
pip install -r requirements.txt
```

### To run the Streamlit app locally, use the following command:

```
streamlit run app.py
```
Navigate to http://localhost:8501 in your web browser to access the app.
Docker Setup

To build and run the application in a Docker container:
Build the Docker Image

```

docker build -t placement-predictor .
```
Run the Docker Container

```

docker run -p 8501:8501 placement-predictor
```
Access the Streamlit app at http://localhost:8501.
Requirements

The project requires the following Python packages:

    streamlit
    scikit-learn
    pandas
    seaborn
    matplotlib

These can be installed via the requirements.txt file.