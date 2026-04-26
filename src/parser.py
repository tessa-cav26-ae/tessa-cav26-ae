from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from .representation import (
    BinaryOp,
    CallOp,
    Command,
    Const,
    Expression,
    Guard,
    IfThenElse,
    Module,
    NaryOp,
    ParsedModel,
    PropertyOp,
    RewardVariable,
    SpecialOp,
    StateVariable,
    UnaryOp,
    Update,
    Var,
)

_IDENTIFIER_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")


@dataclass(frozen=True)
class PrismRewardItem:
    action: str | None
    guard: Expression
    value: Expression


@dataclass(frozen=True)
class PrismRewardModel:
    name: str
    items: list[PrismRewardItem]


def parse_properties(jani: Dict[str, Any]) -> Dict[str, Expression | None]:
    prop_table = {}
    for prop in jani.get("properties", []):
        name = prop.get("name", "unnamed_property")
        expr_json = prop.get("expression")
        if expr_json is not None:
            prop_table[name] = parse_expression(expr_json)
        else:
            prop_table[name] = None
    return prop_table


def parse_constants(jani: Dict[str, Any]) -> Dict[str, Any]:
    const_env = {}
    for constant in jani.get("constants", []):
        name = constant["name"]
        if "value" in constant:
            value = constant["value"]
            # Bare strings in JANI constant values reference other constants
            # (e.g. {"name": "N1", "value": "N"}). parse_expression returns
            # Var(...) for strings, which we resolve below once all entries
            # are loaded.
            if isinstance(value, (dict, str)):
                value = parse_expression(value)
            const_env[name] = value
        else:
            const_env[name] = None
    return const_env


def resolve_constant(expr: Expression, const_env: Mapping[str, Any]) -> Any:
    if isinstance(expr, Const):
        return expr.value
    if isinstance(expr, Var):
        if expr.name in const_env and const_env[expr.name] is not None:
            value = const_env[expr.name]
            if isinstance(value, Expression):
                return value.evaluate(const_env)
            return value
        raise ValueError(f"Unresolved constant: {expr.name}")
    raise ValueError(f"Unsupported constant expression: {expr!r}")


def parse_expression(expr_json: Any) -> Expression:
    if expr_json is None:
        raise ValueError("Empty expression")
    if isinstance(expr_json, (int, float, bool)):
        return Const(expr_json)
    if isinstance(expr_json, str):
        return Var(expr_json)
    if not isinstance(expr_json, dict):
        raise ValueError(f"Unhandled expression from JANI JSON: {expr_json!r}")

    expr_type = expr_json.get("type")
    if expr_type in ("const", "constant", "int", "float", "number"):
        if "value" in expr_json:
            return Const(expr_json["value"])
        if "valueString" in expr_json:
            return Const(float(expr_json["valueString"]))
        raise ValueError(f"Const with no value field: {expr_json!r}")

    if expr_type in ("identifier", "var", "variable"):
        if "name" in expr_json:
            return Var(expr_json["name"])
        if "value" in expr_json:
            return Var(expr_json["value"])
        if "id" in expr_json:
            return Var(expr_json["id"])
        raise ValueError(f"Variable expression missing name: {expr_json!r}")

    if expr_json.get("fun") == "values":
        states = expr_json.get("states")
        parsed_states = parse_expression(states) if isinstance(states, dict) else states
        values = expr_json.get("values")
        parsed_values = {}
        if isinstance(values, dict):
            for key, value in values.items():
                parsed_values[key] = parse_expression(value) if isinstance(value, dict) else value
        return PropertyOp(
            fun=expr_json.get("fun"),
            op=expr_json.get("op"),
            states=parsed_states,
            values=parsed_values,
        )

    if "exp" in expr_json:
        parsed_expr = parse_expression(expr_json["exp"])
        op = expr_json.get("op") or expr_json.get("operator") or expr_json.get("operation")
        if op in ("¬", "not"):
            return UnaryOp("¬", parsed_expr)
        return parsed_expr

    op = expr_json.get("op") or expr_json.get("operator") or expr_json.get("operation")
    args = (
        expr_json.get("args")
        or expr_json.get("arguments")
        or expr_json.get("operands")
    )
    if args is None and "left" in expr_json and "right" in expr_json:
        args = [expr_json["left"], expr_json["right"]]

    if op is not None and args is not None:
        parsed_args = [parse_expression(arg) for arg in args]
        if op == "call":
            return CallOp(expr_json.get("function"), parsed_args)
        if len(parsed_args) == 1:
            return UnaryOp(op, parsed_args[0])
        if len(parsed_args) == 2:
            return BinaryOp(op, parsed_args[0], parsed_args[1])
        return NaryOp(op, parsed_args)

    if op is not None:
        if op == "ite":
            cond = expr_json.get("if")
            then_expr = expr_json.get("then")
            else_expr = expr_json.get("else")
            if cond is None or then_expr is None or else_expr is None:
                raise ValueError(f"ite expression missing if/then/else: {expr_json!r}")
            return IfThenElse(
                parse_expression(cond),
                parse_expression(then_expr),
                parse_expression(else_expr),
            )
        return SpecialOp(op)

    if len(expr_json) == 1:
        key = next(iter(expr_json.keys()))
        value = expr_json[key]
        if isinstance(value, list) and len(value) in (1, 2):
            parsed_args = [parse_expression(item) for item in value]
            if len(parsed_args) == 1:
                return UnaryOp(key, parsed_args[0])
            if len(parsed_args) == 2:
                return BinaryOp(key, parsed_args[0], parsed_args[1])
            return NaryOp(key, parsed_args)

    if "value" in expr_json:
        return Const(expr_json["value"])

    raise NotImplementedError(f"Unhandled expression JANI JSON: {json.dumps(expr_json)}")


def parse_functions(jani: Dict[str, Any]) -> Dict[str, Expression | None]:
    func_table = {}
    for function in jani.get("functions", []):
        name = function["name"]
        body = function.get("body")
        func_table[name] = None if body in (False, None) else parse_expression(body)
    return func_table


def parse_variable(var_json: Dict[str, Any]) -> StateVariable:
    if "name" in var_json:
        name = var_json["name"]
    elif "id" in var_json:
        name = var_json["id"]
    elif "identifier" in var_json:
        name = var_json["identifier"]
    else:
        raise ValueError(f"Variable with no name: {var_json!r}")

    initial = var_json.get("initial-value", var_json.get("initial"))
    # initial-value may be a dict (compound), a bare string (constant ref), or
    # a primitive literal (int/float/bool). Always wrap so downstream code can
    # uniformly call .evaluate(context).
    if initial is not None:
        initial = parse_expression(initial)

    variable_class = RewardVariable if var_json.get("transient") else StateVariable

    if "domain" in var_json:
        raw_domain = var_json["domain"]
        if isinstance(raw_domain, list):
            values = [parse_expression(item) if isinstance(item, dict) else item for item in raw_domain]
            return variable_class(
                name=name,
                initial=initial,
                domain_kind="explicit",
                values=values,
            )
        if isinstance(raw_domain, dict):
            lower = raw_domain.get("lower", raw_domain.get("lower-bound", 0))
            upper = raw_domain.get("upper", raw_domain.get("upper-bound", 1))
            return variable_class(
                name=name,
                initial=initial,
                domain_kind="bounded",
                lower=parse_expression(lower),
                upper=parse_expression(upper),
            )
        raise ValueError(f"Unsupported domain format: {raw_domain!r}")

    if var_json.get("type") in ("int", "integer") and "lowerBound" in var_json and "upperBound" in var_json:
        return variable_class(
            name=name,
            initial=initial,
            domain_kind="bounded",
            lower=parse_expression(var_json["lowerBound"]),
            upper=parse_expression(var_json["upperBound"]),
        )

    if isinstance(var_json.get("type"), dict) and var_json["type"].get("kind") == "bounded":
        type_info = var_json["type"]
        return variable_class(
            name=name,
            initial=initial,
            domain_kind="bounded",
            lower=parse_expression(type_info.get("lower-bound", 0)),
            upper=parse_expression(type_info.get("upper-bound", 1)),
        )

    if var_json.get("type") == "bool":
        return variable_class(
            name=name,
            initial=initial,
            domain_kind="bounded",
            lower=Const(0),
            upper=Const(1),
        )

    return variable_class(
        name=name,
        initial=initial,
        domain_kind="bounded",
        lower=Const(0),
        upper=Const(1),
    )


def parse_update(branch_json: Dict[str, Any]) -> Update:
    probability_expr = parse_expression(branch_json["probability"]) if "probability" in branch_json else Const(1.0)
    assignments = {}
    for assignment in branch_json.get("assignments", []):
        if "ref" in assignment:
            variable_name = assignment["ref"]
        elif "variable" in assignment:
            variable_name = assignment["variable"]
        elif "lval" in assignment:
            variable_name = assignment["lval"]
        elif "lhs" in assignment:
            variable_name = assignment["lhs"]
        else:
            variable_name = None

        if variable_name is None and "target" in assignment:
            target = assignment["target"]
            variable_name = target.get("name") or target.get("id")
        if variable_name is None:
            raise ValueError(f"Assignment missing variable: {assignment!r}")

        if "value" in assignment:
            value_expr = assignment["value"]
        elif "rhs" in assignment:
            value_expr = assignment["rhs"]
        elif "expression" in assignment:
            value_expr = assignment["expression"]
        else:
            value_expr = None
        assignments[variable_name] = parse_expression(value_expr)

    return Update(prob=probability_expr, assignments=assignments)


def parse_command(cmd_json: Dict[str, Any]) -> Command:
    action = cmd_json.get("action") or cmd_json.get("label") or cmd_json.get("name")
    guard_expr = cmd_json.get("guard") or cmd_json.get("condition") or {"type": "const", "value": True}
    guard_node = Guard(parse_expression(guard_expr))

    distribution = (
        cmd_json.get("destinations")
        or cmd_json.get("distribution")
        or cmd_json.get("branches")
    )
    updates: List[Update] = []
    if distribution is None:
        distribution = cmd_json.get("branches", [])

    if isinstance(distribution, dict):
        branches = (
            distribution.get("choices")
            or distribution.get("branches")
            or distribution.get("values")
            or distribution.get("elements")
        )
        if branches is None:
            if "assignments" in distribution:
                updates.append(parse_update(distribution))
        else:
            for branch in branches:
                updates.append(parse_update(branch))
    elif isinstance(distribution, list):
        for branch in distribution:
            updates.append(parse_update(branch))
    else:
        raise NotImplementedError(f"Unhandled distribution type: {distribution!r}")

    return Command(action=action, guard=guard_node, updates=updates)


def parse_module(automaton_json: Dict[str, Any]) -> Module:
    name = automaton_json.get("name") or automaton_json.get("id") or automaton_json.get("identifier") or "unnamed_module"
    variables_json = automaton_json.get("localVariables") or automaton_json.get("variables") or automaton_json.get("vars") or []
    commands_json = automaton_json.get("transitions") or automaton_json.get("commands") or automaton_json.get("edges") or []
    return Module(
        name=name,
        variables=[parse_variable(variable) for variable in variables_json],
        commands=[parse_command(command) for command in commands_json],
    )


def jani_to_modules(jani: Dict[str, Any]) -> ParsedModel:
    modules: List[Module] = []
    prop_table = parse_properties(jani)
    const_env = parse_constants(jani)
    func_table = parse_functions(jani)
    global_vars: list[StateVariable | RewardVariable] = []

    globals_list = jani.get("variables") or jani.get("globalVariables") or []
    if globals_list:
        global_vars = [parse_variable(variable) for variable in globals_list]

    automata_list = jani.get("automata") or jani.get("components") or jani.get("modules") or []
    if automata_list:
        # The compiler requires each StateVariable to be owned by exactly one
        # module. Storm-conv-style JANI declares all variables as globals and
        # leaves automata locals empty, so we distribute each global to the
        # automaton whose edges actually assign to it. This recovers the
        # PRISM-rename ownership pattern (process1 owns x1, process2 owns x2,
        # …) when round-tripping through storm-conv. Globals that no automaton
        # writes to (typically transient labels / rewards) stay under
        # __globals__.
        global_by_name = {v.name: v for v in global_vars}
        var_owner: dict[str, int] = {}
        automaton_assigns: list[set[str]] = []
        for idx, automaton in enumerate(automata_list):
            assigns: set[str] = set()
            for edge in (automaton.get("edges") or automaton.get("transitions") or []):
                for dest in edge.get("destinations", []):
                    for assignment in dest.get("assignments", []):
                        ref = assignment.get("ref")
                        if isinstance(ref, str) and ref in global_by_name:
                            assigns.add(ref)
            automaton_assigns.append(assigns)
            for ref in assigns:
                # Last writer wins. For PRISM-rename models each global is
                # only assigned by one automaton anyway.
                var_owner[ref] = idx

        unowned = [v for v in global_vars if v.name not in var_owner]
        if unowned:
            modules.append(Module(name="__globals__", variables=unowned, commands=[]))

        for idx, automaton in enumerate(automata_list):
            module = parse_module(automaton)
            owned_state_vars = [
                global_by_name[name]
                for name in automaton_assigns[idx]
                if isinstance(global_by_name[name], StateVariable)
                and var_owner.get(name) == idx
            ]
            if owned_state_vars:
                module.variables = list(module.variables) + owned_state_vars
            modules.append(module)
        return ParsedModel(properties=prop_table, constants=const_env, functions=func_table, modules=modules)

    network = jani.get("system") or jani.get("network") or {}
    if network:
        components = network.get("components") or network.get("automata") or []
        for automaton in components:
            modules.append(parse_module(automaton))
        return ParsedModel(properties=prop_table, constants=const_env, functions=func_table, modules=modules)

    model = jani.get("model") or jani.get("models")
    if isinstance(model, dict):
        automata_list = model.get("automata") or model.get("components") or []
        for automaton in automata_list:
            modules.append(parse_module(automaton))

    return ParsedModel(properties=prop_table, constants=const_env, functions=func_table, modules=modules)


def load_model(
    model_type: str,
    model_path: str | Path,
    *,
    constants: Mapping[str, int | float | bool] | None = None,
    defer_constants: Iterable[str] | None = None,
) -> ParsedModel:
    normalized_model_type = model_type.lower()
    if normalized_model_type == "jani":
        if defer_constants:
            raise ValueError("defer_constants is only supported for PRISM models")
        return load_jani_model(model_path, constants=constants)
    if normalized_model_type == "prism":
        return load_prism_model(model_path, constants=constants, defer_constants=defer_constants)
    raise ValueError("model_type must be either 'jani' or 'prism'")


def load_jani_model(
    model_path: str | Path,
    *,
    constants: Mapping[str, int | float | bool] | None = None,
) -> ParsedModel:
    path = Path(model_path)
    with path.open() as handle:
        model = jani_to_modules(json.load(handle))
    if constants:
        for name, value in constants.items():
            if name not in model.constants:
                raise ValueError(f"Unknown JANI constant '{name}'")
            model.constants[name] = value
    # Resolve constant chains (e.g. N1 -> N -> 2) so downstream code sees
    # scalar values rather than Var references. Iterate until no Expression
    # values remain or we make no progress (cycle / undefined dependency).
    for _ in range(len(model.constants) + 1):
        progress = False
        for name, value in list(model.constants.items()):
            if isinstance(value, Expression):
                try:
                    model.constants[name] = value.evaluate(model.constants)
                    progress = True
                except KeyError:
                    pass
        if not progress:
            break
    return model


def load_prism_model(
    model_path: str | Path,
    *,
    constants: Mapping[str, int | float | bool] | None = None,
    defer_constants: Iterable[str] | None = None,
) -> ParsedModel:
    """Parse a PRISM model into tessa's ParsedModel representation.

    ``constants`` supplies values for any ``const`` declarations that lack
    a default. By default these are substituted into the stormpy IR before
    tessa sees the expression tree, so the resulting model is closed.

    ``defer_constants`` is the escape hatch for runtime-overridable PRISM
    constants. Names listed here are *not* substituted into the stormpy IR
    even if a value is supplied in ``constants``: instead, the value lives
    in ``ParsedModel.constants`` as a placeholder, and the model's
    transition expressions retain ``Var(name)`` references that downstream
    consumers (notably ``CompiledReachabilityModel.run(constants_override=...)``)
    can re-evaluate with fresh values per call.
    """
    stormpy = _import_stormpy()
    path = Path(model_path)
    source_text = _strip_prism_comments(path.read_text())

    constant_declarations = _parse_prism_constant_declarations(source_text)
    deferred = set(defer_constants or ())
    const_env = _build_prism_constant_environment(constant_declarations, constants or {})

    prism_program = stormpy.parse_prism_program(str(path))
    # Define everything except deferred constants so they remain symbolic
    # in the stormpy expression tree after `substitute_constants()`.
    constants_to_define = {
        name: value for name, value in (constants or {}).items() if name not in deferred
    }
    prism_program = _define_prism_constants(prism_program, constants_to_define, stormpy)
    _ensure_all_prism_constants_defined(prism_program, const_env, allowed_undefined=deferred)
    prism_program = prism_program.substitute_constants()
    prism_program = prism_program.substitute_formulas()

    prop_table = _parse_prism_labels(source_text)
    func_table = _parse_prism_formulas(source_text)
    modules = [_parse_prism_module(module) for module in prism_program.modules]

    reward_models = _parse_prism_reward_models(source_text)
    if reward_models:
        _inject_prism_rewards(modules, reward_models)
        reward_variables = [RewardVariable(name=model.name, initial=0.0) for model in reward_models]
        modules.insert(0, Module(name="__globals__", variables=reward_variables, commands=[]))

    return ParsedModel(properties=prop_table, constants=const_env, functions=func_table, modules=modules)


def parse_prism_expression(expr: str) -> Expression:
    normalized_expr = expr.strip()
    if not normalized_expr:
        raise ValueError("Empty PRISM expression")

    python_expr, label_placeholders = _translate_prism_expression_to_python(normalized_expr)
    parsed = ast.parse(python_expr, mode="eval")
    return _ast_to_expression(parsed.body, label_placeholders)


def _parse_prism_module(module: Any) -> Module:
    unsupported_variables = []
    for attr_name in ("clock_variables", "array_variables"):
        values = list(getattr(module, attr_name, []))
        if values:
            unsupported_variables.extend(values)
    if unsupported_variables:
        raise NotImplementedError("Clock and array PRISM variables are not supported")

    variables: list[StateVariable] = []
    for variable in getattr(module, "integer_variables", []):
        variables.append(
            StateVariable(
                name=variable.name,
                initial=_parse_storm_expression(variable.initial_value_expression),
                domain_kind="bounded",
                lower=_parse_storm_expression(variable.lower_bound_expression),
                upper=_parse_storm_expression(variable.upper_bound_expression),
            )
        )
    for variable in getattr(module, "boolean_variables", []):
        variables.append(
            StateVariable(
                name=variable.name,
                initial=_parse_storm_expression(variable.initial_value_expression),
                domain_kind="bounded",
                lower=Const(0),
                upper=Const(1),
            )
        )

    commands = [_parse_prism_command(command) for command in module.commands]
    return Module(name=module.name, variables=variables, commands=commands)


def _parse_prism_command(command: Any) -> Command:
    action_name = getattr(command, "action_name", None)
    is_labeled = bool(getattr(command, "is_labeled", getattr(command, "labeled", False)))
    action = action_name if is_labeled and action_name else None
    return Command(
        action=action,
        guard=Guard(_parse_storm_expression(command.guard_expression)),
        updates=[_parse_prism_update(update) for update in command.updates],
    )


def _parse_prism_update(update: Any) -> Update:
    assignments = {
        assignment.variable.name: _parse_storm_expression(assignment.expression)
        for assignment in update.assignments
    }
    return Update(
        prob=_parse_storm_expression(update.probability_expression),
        assignments=assignments,
    )


def _parse_storm_expression(expression: Any) -> Expression:
    return parse_prism_expression(str(expression))


def _import_stormpy():
    try:
        import stormpy
    except ImportError as exc:
        raise RuntimeError(
            "PRISM loading requires stormpy. In this repo, PRISM support is expected "
            "from the Linux/Nix environment."
        ) from exc
    return stormpy


def _define_prism_constants(program: Any, constants: Mapping[str, int | float | bool], stormpy: Any) -> Any:
    if not constants:
        return program

    constant_map = {}
    for name, value in constants.items():
        prism_constant = program.get_constant(name)
        if isinstance(value, bool):
            translated_value = program.expression_manager.create_boolean(value)
        elif isinstance(value, int):
            translated_value = program.expression_manager.create_integer(value)
        elif isinstance(value, float):
            translated_value = program.expression_manager.create_rational(stormpy.Rational(str(value)))
        else:
            raise TypeError(f"Unsupported constant type for '{name}': {type(value).__name__}")
        constant_map[prism_constant.expression_variable] = translated_value

    return program.define_constants(constant_map)


def _ensure_all_prism_constants_defined(
    program: Any,
    const_env: Mapping[str, Any],
    *,
    allowed_undefined: Iterable[str] = (),
) -> None:
    allowed = set(allowed_undefined)
    undefined_constants = []
    for constant in program.constants:
        if constant.name in allowed:
            continue
        if not constant.defined or const_env.get(constant.name) is None:
            undefined_constants.append(constant.name)
    if undefined_constants:
        joined = ", ".join(sorted(undefined_constants))
        raise ValueError(f"Missing values for PRISM constants: {joined}")


def _build_prism_constant_environment(
    declarations: list[tuple[str, str | None]],
    provided_constants: Mapping[str, int | float | bool],
) -> dict[str, Any]:
    const_env: dict[str, Any] = {}
    declared_constant_names = {name for name, _ in declarations}

    unknown_constants = sorted(set(provided_constants) - declared_constant_names)
    if unknown_constants:
        raise ValueError(f"Unknown PRISM constants: {', '.join(unknown_constants)}")

    for name, raw_value in declarations:
        if raw_value is not None and name in provided_constants:
            raise ValueError(f"PRISM constant '{name}' is already defined in the model")
        if name in provided_constants:
            const_env[name] = provided_constants[name]
        elif raw_value is None:
            const_env[name] = None
        else:
            const_env[name] = parse_prism_expression(raw_value)

    return const_env


def _parse_prism_constant_declarations(source_text: str) -> list[tuple[str, str | None]]:
    declarations: list[tuple[str, str | None]] = []
    for match in re.finditer(
        r"\bconst\s+[A-Za-z_][A-Za-z0-9_]*\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:=\s*(.*?))?;",
        source_text,
        flags=re.DOTALL,
    ):
        declarations.append((match.group(1), _clean_prism_fragment(match.group(2))))
    return declarations


def _parse_prism_formulas(source_text: str) -> dict[str, Expression]:
    formulas: dict[str, Expression] = {}
    for match in re.finditer(
        r"\bformula\s+([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?);",
        source_text,
        flags=re.DOTALL,
    ):
        formulas[match.group(1)] = parse_prism_expression(match.group(2))
    return formulas


def _parse_prism_labels(source_text: str) -> dict[str, Expression]:
    labels: dict[str, Expression] = {}
    for match in re.finditer(
        r'\blabel\s+"([^"]+)"\s*=\s*(.*?);',
        source_text,
        flags=re.DOTALL,
    ):
        labels[match.group(1)] = parse_prism_expression(match.group(2))
    return labels


def _parse_prism_reward_models(source_text: str) -> list[PrismRewardModel]:
    reward_models: list[PrismRewardModel] = []
    for index, match in enumerate(
        re.finditer(r'rewards(?:\s+"([^"]+)")?\s*(.*?)endrewards', source_text, flags=re.DOTALL)
    ):
        reward_name = match.group(1) or f"reward_{index}"
        items: list[PrismRewardItem] = []
        for raw_item in _split_top_level(match.group(2), ";"):
            fragment = raw_item.strip()
            if not fragment:
                continue

            action = None
            remainder = fragment
            if fragment.startswith("["):
                end_bracket = _find_matching_bracket(fragment, 0, "[", "]")
                action = _clean_prism_fragment(fragment[1:end_bracket])
                remainder = fragment[end_bracket + 1 :]

            guard_text, value_text = _split_first_top_level(remainder, ":")
            items.append(
                PrismRewardItem(
                    action=action,
                    guard=parse_prism_expression(guard_text),
                    value=parse_prism_expression(value_text),
                )
            )
        reward_models.append(PrismRewardModel(name=reward_name, items=items))
    return reward_models


def _inject_prism_rewards(modules: list[Module], reward_models: list[PrismRewardModel]) -> None:
    existing_variables = {variable.name for module in modules for variable in module.variables}
    reward_names = [model.name for model in reward_models]
    collisions = sorted(existing_variables.intersection(reward_names))
    if collisions:
        raise ValueError(f"Reward variable names collide with model variables: {', '.join(collisions)}")

    action_carriers: dict[str, int] = {}
    for module_index, module in enumerate(modules):
        for command in module.commands:
            if command.action is not None and command.action not in action_carriers:
                action_carriers[command.action] = module_index

    for module_index, module in enumerate(modules):
        for command in module.commands:
            if command.action is not None and action_carriers.get(command.action) != module_index:
                continue

            reward_assignments = {
                reward_model.name: _build_reward_expression(reward_model.items, command.action)
                for reward_model in reward_models
            }
            reward_assignments = {
                name: expr for name, expr in reward_assignments.items() if not _is_zero_expression(expr)
            }
            if not reward_assignments:
                continue

            for update in command.updates:
                update.assignments.update(reward_assignments)


def _build_reward_expression(items: list[PrismRewardItem], action: str | None) -> Expression:
    expressions = []
    for item in items:
        if item.action is not None and item.action != action:
            continue
        expressions.append(IfThenElse(item.guard, item.value, Const(0)))
    return _sum_expressions(expressions)


def _sum_expressions(expressions: list[Expression]) -> Expression:
    if not expressions:
        return Const(0)
    if len(expressions) == 1:
        return expressions[0]
    return NaryOp("+", expressions)


def _is_zero_expression(expr: Expression) -> bool:
    return isinstance(expr, Const) and expr.value == 0


def _translate_prism_expression_to_python(expr: str) -> tuple[str, dict[str, str]]:
    rewritten_expr = _rewrite_prism_ternary(expr)
    label_placeholders: dict[str, str] = {}
    translated_tokens: list[str] = []
    index = 0
    while index < len(rewritten_expr):
        if rewritten_expr.startswith("<=>", index):
            raise NotImplementedError("PRISM operator '<=>' is not supported")
        if rewritten_expr.startswith("=>", index):
            raise NotImplementedError("PRISM operator '=>' is not supported")
        if rewritten_expr[index] == '"':
            closing_quote = _find_string_end(rewritten_expr, index)
            label_name = rewritten_expr[index + 1 : closing_quote]
            placeholder = f"__label_{len(label_placeholders)}"
            label_placeholders[placeholder] = label_name
            translated_tokens.append(placeholder)
            index = closing_quote + 1
            continue
        if rewritten_expr.startswith("!=", index):
            translated_tokens.append("!=")
            index += 2
            continue
        if rewritten_expr.startswith("<=", index) or rewritten_expr.startswith(">=", index) or rewritten_expr.startswith("==", index):
            translated_tokens.append(rewritten_expr[index : index + 2])
            index += 2
            continue

        current = rewritten_expr[index]
        if current == "=":
            translated_tokens.append("==")
        elif current == "!":
            translated_tokens.append("~")
        elif current == "&":
            translated_tokens.append(" and ")
        elif current == "|":
            translated_tokens.append(" or ")
        elif current == "^":
            raise NotImplementedError("PRISM operator '^' is not supported")
        else:
            translated_tokens.append(current)
        index += 1

    python_expr = "".join(translated_tokens)
    python_expr = _IDENTIFIER_RE.sub(_replace_prism_boolean_keyword, python_expr)
    return python_expr, label_placeholders


def _replace_prism_boolean_keyword(match: re.Match[str]) -> str:
    identifier = match.group(0)
    if identifier == "true":
        return "True"
    if identifier == "false":
        return "False"
    return identifier


def _rewrite_prism_ternary(expr: str) -> str:
    stripped_expr = expr.strip()
    rewritten_segments: list[str] = []
    index = 0
    while index < len(stripped_expr):
        char = stripped_expr[index]
        if char == '"':
            string_end = _find_string_end(stripped_expr, index)
            rewritten_segments.append(stripped_expr[index : string_end + 1])
            index = string_end + 1
            continue
        if char in "([{":
            closing = {"(": ")", "[": "]", "{": "}"}[char]
            matching_index = _find_matching_bracket(stripped_expr, index, char, closing)
            inner = _rewrite_prism_ternary(stripped_expr[index + 1 : matching_index])
            rewritten_segments.append(f"{char}{inner}{closing}")
            index = matching_index + 1
            continue
        rewritten_segments.append(char)
        index += 1
    stripped_expr = "".join(rewritten_segments)

    question_index = _find_top_level_character(stripped_expr, "?")
    if question_index == -1:
        return stripped_expr

    colon_index = _find_ternary_colon(stripped_expr, question_index)
    condition = _rewrite_prism_ternary(stripped_expr[:question_index])
    then_expr = _rewrite_prism_ternary(stripped_expr[question_index + 1 : colon_index])
    else_expr = _rewrite_prism_ternary(stripped_expr[colon_index + 1 :])
    return f"__ite__({condition}, {then_expr}, {else_expr})"


def _find_top_level_character(text: str, target: str) -> int:
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _find_string_end(text, index)
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == target and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            return index
        index += 1
    return -1


def _find_ternary_colon(text: str, question_index: int) -> int:
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    nested_ternaries = 0
    index = question_index + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _find_string_end(text, index)
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == "?" and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            nested_ternaries += 1
        elif char == ":" and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            if nested_ternaries == 0:
                return index
            nested_ternaries -= 1
        index += 1
    raise ValueError(f"Malformed PRISM ternary expression: {text}")


def _split_top_level(text: str, separator: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    segment_start = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _find_string_end(text, index)
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == separator and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            parts.append(text[segment_start:index])
            segment_start = index + 1
        index += 1
    parts.append(text[segment_start:])
    return parts


def _split_first_top_level(text: str, separator: str) -> tuple[str, str]:
    depth = 0
    bracket_depth = 0
    brace_depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _find_string_end(text, index)
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "[":
            bracket_depth += 1
        elif char == "]":
            bracket_depth -= 1
        elif char == "{":
            brace_depth += 1
        elif char == "}":
            brace_depth -= 1
        elif char == separator and depth == 0 and bracket_depth == 0 and brace_depth == 0:
            return text[:index].strip(), text[index + 1 :].strip()
        index += 1
    raise ValueError(f"Expected separator '{separator}' in PRISM fragment: {text}")


def _find_matching_bracket(text: str, start_index: int, opening: str, closing: str) -> int:
    depth = 1
    index = start_index + 1
    while index < len(text):
        char = text[index]
        if char == '"':
            index = _find_string_end(text, index)
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return index
        index += 1
    raise ValueError(f"Unmatched '{opening}' in PRISM fragment: {text}")


def _find_string_end(text: str, start_index: int) -> int:
    index = start_index + 1
    while index < len(text):
        if text[index] == '"' and text[index - 1] != "\\":
            return index
        index += 1
    raise ValueError(f"Unterminated string literal in PRISM fragment: {text}")


def _ast_to_expression(node: ast.AST, label_placeholders: Mapping[str, str]) -> Expression:
    if isinstance(node, ast.Name):
        if node.id == "True":
            return Const(True)
        if node.id == "False":
            return Const(False)
        if node.id in label_placeholders:
            return Var(label_placeholders[node.id])
        return Var(node.id)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return Var(node.value)
        return Const(node.value)

    if isinstance(node, ast.BoolOp):
        op = "∧" if isinstance(node.op, ast.And) else "∨"
        return NaryOp(op, [_ast_to_expression(value, label_placeholders) for value in node.values])

    if isinstance(node, ast.BinOp):
        left = _ast_to_expression(node.left, label_placeholders)
        right = _ast_to_expression(node.right, label_placeholders)
        if isinstance(node.op, ast.Add):
            return BinaryOp("+", left, right)
        if isinstance(node.op, ast.Sub):
            return BinaryOp("-", left, right)
        if isinstance(node.op, ast.Mult):
            return BinaryOp("*", left, right)
        if isinstance(node.op, ast.Div):
            return BinaryOp("/", left, right)
        raise NotImplementedError(f"Unsupported PRISM binary operator: {ast.dump(node.op)}")

    if isinstance(node, ast.UnaryOp):
        operand = _ast_to_expression(node.operand, label_placeholders)
        if isinstance(node.op, ast.Not):
            return UnaryOp("¬", operand)
        if isinstance(node.op, ast.Invert):
            return UnaryOp("¬", operand)
        if isinstance(node.op, ast.USub):
            if isinstance(operand, Const) and isinstance(operand.value, (int, float)):
                return Const(-operand.value)
            return BinaryOp("-", Const(0), operand)
        raise NotImplementedError(f"Unsupported PRISM unary operator: {ast.dump(node.op)}")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise NotImplementedError("Chained PRISM comparisons are not supported")
        left = _ast_to_expression(node.left, label_placeholders)
        right = _ast_to_expression(node.comparators[0], label_placeholders)
        op = node.ops[0]
        if isinstance(op, ast.Eq):
            return BinaryOp("==", left, right)
        if isinstance(op, ast.NotEq):
            return BinaryOp("!=", left, right)
        if isinstance(op, ast.Lt):
            return BinaryOp("<", left, right)
        if isinstance(op, ast.LtE):
            return BinaryOp("<=", left, right)
        if isinstance(op, ast.Gt):
            return BinaryOp(">", left, right)
        if isinstance(op, ast.GtE):
            return BinaryOp(">=", left, right)
        raise NotImplementedError(f"Unsupported PRISM comparison operator: {ast.dump(op)}")

    if isinstance(node, ast.IfExp):
        return IfThenElse(
            _ast_to_expression(node.test, label_placeholders),
            _ast_to_expression(node.body, label_placeholders),
            _ast_to_expression(node.orelse, label_placeholders),
        )

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise NotImplementedError("Only simple PRISM function names are supported")
        function_name = node.func.id
        args = [_ast_to_expression(arg, label_placeholders) for arg in node.args]
        if function_name == "__ite__":
            if len(args) != 3:
                raise ValueError("__ite__ requires exactly three arguments")
            return IfThenElse(args[0], args[1], args[2])
        if function_name == "max":
            return _fold_binary_function("max", args)
        if function_name == "min":
            return _fold_binary_function("min", args)
        if function_name == "exactlyOneOf":
            return _translate_exactly_one_of(args)
        return CallOp(function_name, args)

    raise NotImplementedError(f"Unsupported PRISM expression node: {ast.dump(node)}")


def _fold_binary_function(function_name: str, args: list[Expression]) -> Expression:
    if not args:
        raise ValueError(f"{function_name} requires at least one argument")
    result = args[0]
    for arg in args[1:]:
        result = BinaryOp(function_name, result, arg)
    return result


def _translate_exactly_one_of(args: list[Expression]) -> Expression:
    counted_terms = [IfThenElse(arg, Const(1), Const(0)) for arg in args]
    return BinaryOp("==", _sum_expressions(counted_terms), Const(1))


def _strip_prism_comments(source_text: str) -> str:
    without_block_comments = re.sub(r"/\*.*?\*/", "", source_text, flags=re.DOTALL)
    return re.sub(r"//.*?$", "", without_block_comments, flags=re.MULTILINE)


def _clean_prism_fragment(fragment: str | None) -> str | None:
    if fragment is None:
        return None
    cleaned_fragment = fragment.strip()
    return cleaned_fragment or None


__all__ = [
    "jani_to_modules",
    "load_model",
    "load_jani_model",
    "load_prism_model",
    "parse_prism_expression",
]
