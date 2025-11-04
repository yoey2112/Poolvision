# PoolVision App (Skeleton)

This is a **starter implementation** for the PoolVision project.  It provides a basic project structure and some example code to get you started on building a full billiards vision and scoring application.  The goal of PoolVision is to track billiard balls on a pool table, automatically score APA‐style 8‑ball and 9‑ball games, manage player profiles, and provide training drills and analytics.

## Project Structure

```
poolvision_app/
├── README.md             # Project overview and setup instructions
├── requirements.txt      # Python package dependencies
├── src/
│   ├── api/              # FastAPI application
│   │   └── main.py
│   ├── vision/           # Computer vision modules
│   │   ├── __init__.py
│   │   ├── ball_tracking.py
│   │   └── calibration.py
│   ├── rules/            # Game rules and scoring
│   │   ├── __init__.py
│   │   ├── eightball.py
│   │   └── nineball.py
│   ├── engine/           # Orchestration layer
│   │   ├── __init__.py
│   │   └── engine.py
│   └── db/               # Database models
│       ├── __init__.py
│       └── models.py
├── tests/                # Unit tests (skeleton)
│   └── test_sample.py
└── web/                  # Placeholder for front‑end code
    └── README.md
```

## Getting Started

1. **Install dependencies:**  Install Python 3.9+ and then run:

   ```sh
   pip install -r requirements.txt
   ```

2. **Run the API server:**  Navigate to the `src/api` directory and start the FastAPI server:

   ```sh
   uvicorn src.api.main:app --reload
   ```

3. **Run the sample unit tests:**  From the project root, run:

   ```sh
   pytest
   ```

4. **Extend the code:**  Use the provided modules as a starting point.  You will need to implement full ball detection and tracking, rule enforcement for APA 8‑ball and 9‑ball games, database persistence, analytics, and a front‑end interface.

## Notes

- The current implementation only includes placeholder logic for ball detection and game rules.  You must extend these modules to achieve full functionality (e.g., tracking multiple balls across frames, detecting pockets, scoring games, etc.).
- The `web` directory is empty except for a README—this is where you can build a React front end or other user interface.
