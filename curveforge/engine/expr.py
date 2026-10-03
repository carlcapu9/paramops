# SPDX-License-Identifier: GPL-3.0-or-later
"""Safe numeric expressions for the Expression node.

Expressions use Python syntax restricted to numbers, variables, arithmetic,
comparisons, boolean logic, ``a if c else b`` and a whitelist of functions::

    index % 3 == 0 and distance > 2
    clamp(random * 2, 0, 1)
    sin(distance_pct * pi)
"""

import ast
import math

FUNCTIONS = {
    "abs": abs, "min": min, "max": max, "round": round, "int": int, "float": float,
    "floor": math.floor, "ceil": math.ceil, "sqrt": lambda v: math.sqrt(max(v, 0.0)),
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "asin": math.asin, "acos": math.acos,
    "atan": math.atan, "atan2": math.atan2, "radians": math.radians, "degrees": math.degrees,
    "pow": pow, "log": lambda v, b=math.e: math.log(v, b) if v > 0 else 0.0, "exp": math.exp,
    "clamp": lambda v, lo=0.0, hi=1.0: max(lo, min(hi, v)),
    "lerp": lambda a, b, t: a + (b - a) * t,
    "step": lambda edge, v: 1.0 if v >= edge else 0.0,
    "sign": lambda v: (v > 0) - (v < 0),
    "fract": lambda v: v - math.floor(v),
    "mod": lambda a, b: math.fmod(a, b) if b else 0.0,
}
CONSTANTS = {"pi": math.pi, "tau": math.tau, "e": math.e, "true": 1.0, "false": 0.0,
             "True": 1.0, "False": 0.0}

_ALLOWED = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.IfExp,
            ast.Call, ast.Name, ast.Load, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div,
            ast.FloorDiv, ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Not, ast.And, ast.Or,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE)


class ExpressionError(ValueError):
    pass


def compile_expression(text, names):
    """Compile ``text`` into ``f(values: dict) -> float``.

    ``names`` lists the variable names the expression may use (besides the
    constants and functions above). Raises :class:`ExpressionError`.
    """
    text = (text or "").strip() or "0"
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError as exc:
        raise ExpressionError("syntax error: %s" % exc.msg) from None
    allowed_names = set(names) | set(CONSTANTS)
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise ExpressionError("'%s' is not allowed" % type(node).__name__)
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, bool)):
            raise ExpressionError("only numbers are allowed")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in FUNCTIONS or node.keywords:
                raise ExpressionError("unknown function")
        elif isinstance(node, ast.Name):
            if node.id not in allowed_names and node.id not in FUNCTIONS:
                raise ExpressionError("unknown name '%s'" % node.id)
    code = compile(tree, "<expression>", "eval")
    env = {"__builtins__": {}}
    env.update(FUNCTIONS)
    env.update(CONSTANTS)

    def run(values):
        try:
            result = eval(code, env, values)  # noqa: S307 - validated AST, no builtins
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            return 0.0
        return float(result)

    return run
