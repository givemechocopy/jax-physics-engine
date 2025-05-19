# backend/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .simulator import run_simulation
import json
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 로컬 개발 시 전체 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

trajectory_path = os.path.join(os.path.dirname(__file__), "trajectory.json")

def save_trajectory():
    data = run_simulation()
    with open(trajectory_path, "w") as f:
        json.dump(data, f)

if not os.path.exists(trajectory_path):
    save_trajectory()

@app.get("/trajectory")
def get_trajectory():
    with open(trajectory_path, "r") as f:
        return json.load(f)

@app.post("/trajectory/refresh")
def refresh_trajectory():
    save_trajectory()
    return {"message": "Trajectory refreshed."}
