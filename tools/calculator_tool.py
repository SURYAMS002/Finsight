import ast
import operator as op
from typing import Union

# Allowed AST node types for safe mathematical evaluation
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.Mod: op.mod,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}

def safe_eval_expr(expr_str: str) -> Union[int, float]:
    """
    Safely evaluates a mathematical expression string using Python's AST module.
    Prevents execution of arbitrary Python code or dangerous builtins.
    """
    # Clean expression string
    clean_expr = expr_str.replace("₹", "").replace(",", "").replace("%", "").strip()
    
    node = ast.parse(clean_expr, mode='eval').body

    def _eval(node):
        if isinstance(node, ast.Constant):  # Python 3.8+ numbers/constants
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Unsupported constant type: {type(node.value)}")
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type}")
            if op_type == ast.Div and right == 0:
                raise ZeroDivisionError("Division by zero in calculation.")
            return ALLOWED_OPERATORS[op_type](left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type not in ALLOWED_OPERATORS:
                raise ValueError(f"Unsupported unary operator: {op_type}")
            return ALLOWED_OPERATORS[op_type](operand)
        else:
            raise ValueError(f"Unsupported AST node: {type(node)}")

    return _eval(node)

def calculator(expression: str) -> str:
    """
    Perform exact financial arithmetic calculations (growth rate, percentage change, ratio, difference).
    
    Args:
        expression: Mathematical formula (e.g. '(44960 - 40000) / 40000 * 100')
        
    Returns:
        Exact calculation result string.
    """
    try:
        res = safe_eval_expr(expression)
        if isinstance(res, float):
            # Format nicely to 4 decimal places max, avoiding unnecessary trail zeroes
            formatted_res = f"{res:.4f}".rstrip('0').rstrip('.')
        else:
            formatted_res = str(res)

        return f"Calculation Result: {formatted_res}\nFormula: {expression} = {formatted_res}"
    except ZeroDivisionError:
        return f"Calculation Error: Division by zero in expression '{expression}'"
    except Exception as e:
        return f"Calculation Error: Invalid expression '{expression}' ({str(e)})"
