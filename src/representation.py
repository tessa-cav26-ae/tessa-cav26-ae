from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import numpy as np

_array_api = np


@contextmanager
def use_array_api(array_api):
    global _array_api
    previous = _array_api
    _array_api = array_api
    try:
        yield
    finally:
        _array_api = previous


class Module:
    def __init__(self, name, variables, commands):
        self.name = name
        self.variables = variables
        self.commands = commands

class Command:
    def __init__(self, action, guard: Guard, updates: list[Update]):
        self.action = action
        self.guard = guard
        self.updates = updates
    
class Guard:
    def __init__(self, expr: Expression):
        self.expr = expr

class Update:
    def __init__(self, prob, assignments: dict[str, Expression]):
        self.prob = prob
        self.assignments = assignments

class Expression:
    def evaluate(self, context):
        raise NotImplementedError
    
class Const(Expression):
    def __init__(self, value):
        self.value = value

    def evaluate(self, context):
        return self.value
    
    def __repr__(self):
        return f"{self.value}"


class Var(Expression):
    def __init__(self, name):
        self.name = name

    def evaluate(self, context):
        return context[self.name]
    
    def __repr__(self):
        return self.name

class UnaryOp(Expression):
    def __init__(self, op, operand):
        self.op = op
        self.operand = operand

    def evaluate(self, context):
        val = self.operand.evaluate(context)
        if self.op in ("¬", "not"):
            return _array_api.logical_not(val)
        raise NotImplementedError

    def __repr__(self):
        return f"{self.op}({self.operand})"

class BinaryOp(Expression):
    def __init__(self, op, left, right):
        self.op = op
        self.left = left
        self.right = right

    def evaluate(self, context):
        left = self.left.evaluate(context)
        right = self.right.evaluate(context)

        if self.op == "+":
            return left + right
        if self.op == "<":
            return left < right
        if self.op == ">":
            return left > right
        if self.op == "==" or self.op == "=":
            return left == right
        if self.op == "!=":
            return left != right
        if self.op == "<=":
            return left <= right
        if self.op == ">=":
            return left >= right
        if self.op == "-":
            return left - right
        if self.op == "*":
            return left * right
        if self.op == "/":
            return left / right
        if self.op == "∧":
            return _array_api.logical_and(left, right)
        if self.op == "∨":
            return _array_api.logical_or(left, right)
        if self.op == "max":
            return _array_api.maximum(left, right)
        if self.op == "min":
            return _array_api.minimum(left, right)
        raise NotImplementedError(f"BinaryOp '{self.op}' not implemented")

    def __repr__(self):
        return f"BinaryOp('{self.op}', {self.left!r}, {self.right!r})"

class NaryOp(Expression):
    def __init__(self, op: str, args: list[Expression]):
        self.op = op
        self.args = args

    def evaluate(self, context):
        vals = [a.evaluate(context) for a in self.args]
        if self.op == "+":
            out = vals[0]
            for v in vals[1:]:
                out = out + v
            return out
        if self.op == "*":
            out = vals[0]
            for v in vals[1:]:
                out = out * v
            return out
        if self.op == "∧":
            out = vals[0]
            for v in vals[1:]:
                out = _array_api.logical_and(out, v)
            return out
        if self.op == "∨":
            out = vals[0]
            for v in vals[1:]:
                out = _array_api.logical_or(out, v)
            return out
        if len(vals) >= 2:
            L, R = vals[0], vals[1]
            if self.op == "<":
                return L < R
            if self.op == ">":
                return L > R
            if self.op == "==":
                return L == R
            if self.op == "!=":
                return L != R
        raise NotImplementedError(f"NaryOp '{self.op}' not implemented")

    def __repr__(self):
        rendered_args = ", ".join(repr(arg) for arg in self.args)
        return f"NaryOp('{self.op}', {rendered_args})"

class CallOp(Expression):
    def __init__(self, function, args):
        self.function = function
        self.args = args

    def evaluate(self, context):
        fn = context[self.function]
        if isinstance(fn, Expression):
            # Zero-arg function stored as its body Expression; evaluate in same context
            return fn.evaluate(context)
        if not callable(fn):
            # Pre-evaluated result (e.g. a JAX array over the grid)
            return fn
        vals = [a.evaluate(context) for a in self.args]
        return fn(*vals)

    def __repr__(self):
        return f"{self.function}({', '.join(repr(a) for a in self.args)})"

class IfThenElse(Expression):
    def __init__(self, cond, then_branch, else_branch):
        self.cond = cond
        self.then_branch = then_branch
        self.else_branch = else_branch

    def evaluate(self, context):
        c = self.cond.evaluate(context)
        t = self.then_branch.evaluate(context)
        e = self.else_branch.evaluate(context)
        return _array_api.where(c, t, e)

    def __repr__(self):
        return f"ite({self.cond}, {self.then_branch}, {self.else_branch})"

class PropertyOp(Expression):
    def __init__(self, fun=None, op=None, states=None, values=None):
        self.fun = fun
        self.op = op
        self.states = states
        self.values = values

    def __repr__(self):
        return f"PropertyOp(fun={self.fun}, op={self.op}, states={self.states}, values={self.values})"

class SpecialOp(Expression):
    def __init__(self, op=None):
        self.op = op

    def __repr__(self):
        return f"SpecialOp(op={self.op})"

class Variable:
    def __init__(self, name, domain_kind=None, domain=None, values=None, lower=None, upper=None, initial=None):
        self.name = name
        self.domain_kind = domain_kind
        self.domain = domain
        self.values = values
        self.lower = lower
        self.upper = upper
        self.initial = initial

    def resolve(self, const_env=None):
        """Compute concrete domain array from bounds, resolving constant expressions."""
        if const_env is None:
            const_env = {}
        if self.domain is not None:
            return
        if self.domain_kind == "explicit":
            vals = []
            for v in (self.values or []):
                vals.append(int(v.evaluate(const_env) if isinstance(v, Expression) else v))
            self.domain = _array_api.array(vals)
        elif self.domain_kind == "bounded":
            lo = self.lower if self.lower is not None else 0
            hi = self.upper if self.upper is not None else 1
            if isinstance(lo, Expression):
                lo = int(lo.evaluate(const_env))
            if isinstance(hi, Expression):
                hi = int(hi.evaluate(const_env))
            self.domain = _array_api.arange(lo, hi + 1)

    @property
    def size(self):
        if self.domain is not None:
            return len(self.domain)
        if self.domain_kind == "explicit":
            return len(self.values) if self.values is not None else 0
        return None

    def is_reward(self):
        return isinstance(self, RewardVariable)

    def is_state(self):
        return isinstance(self, StateVariable)

class StateVariable(Variable):
    pass

class RewardVariable(Variable):
    pass


@dataclass
class ParsedModel:
    properties: dict[str, Expression | None]
    constants: dict[str, Any]
    functions: dict[str, Expression | None]
    modules: list[Module]
