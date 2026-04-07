from __future__ import annotations

import importlib


MODULES = [
    "envs",
    "envs.experiment_rescue_lab",
    "envs.experiment_rescue_lab.models",
    "envs.experiment_rescue_lab.client",
    "envs.experiment_rescue_lab.server",
    "envs.experiment_rescue_lab.server.config",
    "envs.experiment_rescue_lab.server.environment",
    "envs.experiment_rescue_lab.server.reward",
    "envs.experiment_rescue_lab.server.grader",
    "envs.experiment_rescue_lab.server.scenarios",
    "envs.experiment_rescue_lab.server.app",
]


def test_all_core_modules_import():
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None, f"Failed to import {module_name}"


def test_app_object_exists():
    app_module = importlib.import_module("envs.experiment_rescue_lab.server.app")
    assert hasattr(app_module, "app"), "FastAPI app must be exposed as 'app'"


def test_environment_factory_exists():
    env_module = importlib.import_module("envs.experiment_rescue_lab.server.environment")
    assert hasattr(env_module, "make_environment"), "make_environment() must exist"