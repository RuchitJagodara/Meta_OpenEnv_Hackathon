from __future__ import annotations

import importlib


MODULES = [
    "server",
    "server",
    "models",
    "client",
    "server",
    "server.config",
    "server.environment",
    "server.reward",
    "server.grader",
    "server.scenarios",
    "server.app",
]


def test_all_core_modules_import():
    for module_name in MODULES:
        module = importlib.import_module(module_name)
        assert module is not None, f"Failed to import {module_name}"


def test_app_object_exists():
    app_module = importlib.import_module("server.app")
    assert hasattr(app_module, "app"), "FastAPI app must be exposed as 'app'"


def test_environment_factory_exists():
    env_module = importlib.import_module("server.environment")
    assert hasattr(env_module, "make_environment"), "make_environment() must exist"