from __future__ import annotations

from pathlib import Path
from typing import Any

from .representation import Command, Expression, Guard, Module, ParsedModel, Update, Variable


def model_to_data(parsed_model: ParsedModel) -> dict[str, Any]:
    properties, constants, functions, modules = (
        parsed_model.properties, parsed_model.constants, parsed_model.functions, parsed_model.modules
    )
    return {
        "properties": {name: stringify_value(value) for name, value in properties.items()},
        "constants": {name: stringify_value(value) for name, value in constants.items()},
        "functions": {name: stringify_value(value) for name, value in functions.items()},
        "modules": [module_to_data(module) for module in modules],
    }


def module_to_data(module: Module) -> dict[str, Any]:
    return {
        "name": module.name,
        "variables": [variable_to_data(variable) for variable in module.variables],
        "commands": [command_to_data(command) for command in module.commands],
    }


def variable_to_data(variable: Variable) -> dict[str, Any]:
    return {
        "name": variable.name,
        "kind": variable.__class__.__name__,
        "domain_kind": variable.domain_kind,
        "domain": stringify_value(variable.domain),
        "values": stringify_value(variable.values),
        "lower": stringify_value(variable.lower),
        "upper": stringify_value(variable.upper),
        "initial": stringify_value(variable.initial),
    }


def command_to_data(command: Command) -> dict[str, Any]:
    return {
        "action": command.action,
        "guard": guard_to_data(command.guard),
        "updates": [update_to_data(update) for update in command.updates],
    }


def guard_to_data(guard: Guard) -> dict[str, Any]:
    return {"expr": stringify_value(guard.expr)}


def update_to_data(update: Update) -> dict[str, Any]:
    return {
        "prob": stringify_value(update.prob),
        "assignments": {
            name: stringify_value(value) for name, value in update.assignments.items()
        },
    }


def stringify_value(value: Any) -> Any:
    if isinstance(value, Expression):
        return repr(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [stringify_value(item) for item in value]
    if isinstance(value, tuple):
        return [stringify_value(item) for item in value]
    if isinstance(value, dict):
        return {key: stringify_value(item) for key, item in value.items()}
    return value
