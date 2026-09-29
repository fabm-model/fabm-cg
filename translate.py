#!/usr/bin/env python
"""Translate a Python-based FABM model description into a Fortran/FABM module.

The translator loads a Python source file that defines one or more
:class:`ingredients.Model` subclasses and optional helper functions, parses it
into an abstract syntax tree (AST), and emits an equivalent Fortran 90 source
file that is compatible with the Framework for Aquatic Biogeochemical Models
(FABM).

Example:
    From the command line::

        python translate.py ergom.py ergom.F90

    Programmatically::

        from translate import translate
        translate("ergom.py", "ergom.F90")

Key components:
    * :class:`Block` — a nestable list of Fortran source lines with indentation
      support.
    * :class:`Visitor` — base AST visitor that collects type information and
      dispatches to concrete translation helpers.
    * :class:`FABMTranslator` — concrete subclass of :class:`Visitor` that emits
      FABM-specific Fortran constructs (``type_base_model`` extensions, loop
      macros, registration calls, …).
"""

import ast
import argparse
from typing import TextIO
from types import ModuleType
import os
import importlib.util
import collections
from collections.abc import Iterable, Mapping, Callable
import sys
from pathlib import Path

import numpy as np

import ingredients

# Mapping from Python built-in types to their Fortran equivalents.
FORTRAN_TYPES = {float: "real(rk)", int: "integer", bool: "logical"}

# Mapping from Python AST binary-operator nodes to Fortran operator strings.
BINOPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.Pow: "**"}

# Mapping from Python AST unary-operator nodes to Fortran operator strings.
UNARYOPS = {ast.UAdd: "+", ast.USub: "-"}

# Mapping from Python AST comparison-operator nodes to Fortran operator strings.
CMPOPS = {
    ast.Eq: "==",
    ast.NotEq: "/=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Lt: "<",
    ast.LtE: "<=",
}

# Mapping from NumPy ufunc objects to the equivalent Fortran intrinsic names.
NUMPY_UFUNCS = {
    np.exp: "exp",
    np.tanh: "tanh",
    np.log10: "log10",
    np.log: "log",
    np.sqrt: "sqrt",
}


def translate(fn: os.PathLike, outpath: os.PathLike | None = None):
    """Translate a Python FABM model file to a Fortran source file.

    The function imports the specified file as a Python module (so that
    class/attribute metadata defined at runtime is available), then parses
    the same file as an Abstract Syntax Tree (AST).
    An :class:`FABMTranslator` visitor walks the tree and produces a
    :class:`Block` tree, which is written out as indented Fortran source.

    Args:
        fn: Path to the input Python source file.
        outpath: Path for the output Fortran file. When ``None`` the Fortran
            source is written to *stdout*.
    """
    # Load the module
    path = Path(fn)
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    # Get abstract syntax tree
    with open(path) as f:
        tree = ast.parse(f.read(), type_comments=True)

    # Translate each AST node
    vis = FABMTranslator(module)
    translation = vis.visit(tree)

    # Write final translation to file or stdout
    out = sys.stdout if outpath is None else open(outpath, "w")
    translation.write(out)
    if out is not sys.stdout:
        out.close()


class Expression:
    def __init__(self, value: str, tp: type):
        assert tp in FORTRAN_TYPES, f"Unsupported type {tp}"
        self.value = value
        self.type = tp

    def __str__(self) -> str:
        return self.value


class Function:
    def __init__(self, value: str, tp: Callable[..., type]):
        assert callable(tp), f"Expected a callable type for function {value}"
        self.value = value
        self.type = tp

    def __str__(self) -> str:
        return self.value


class Block(collections.UserList["Block | Expression"]):
    """A nestable list of Fortran source lines.

    Each element of the list is either a plain ``str`` (a single Fortran
    statement) or a nested :class:`Block`.  Nesting increases the indentation
    level when the tree is serialised by :meth:`write`.

    Args:
        items: Initial contents of the block.
        separate: When ``True`` a blank line is emitted before and after the
            block, and between consecutive plain-string items inside it.
    """

    def __init__(self, items: Iterable = (), separate: bool = False):
        super().__init__(items)
        self.separate = separate

    def write(self, f: TextIO, depth: int = 0):
        """Write the block tree to *f* with two-space indentation per level.

        Args:
            f: A writable file-like object.
            depth: Current indentation depth (number of two-space levels to
                prepend).
        """
        if self.separate and self.data:
            f.write("\n")
        previous_at_top = False
        for item in self.data:
            if isinstance(item, Block):
                item.write(f, depth + 1)
                previous_at_top = False
            else:
                if previous_at_top and self.separate:
                    f.write("\n")
                f.write(f"{'  ' * depth}{item}\n")
                previous_at_top = True
        if self.separate and self.data:
            f.write("\n")


class FixedReturnType:
    def __init__(self, return_type: type):
        self.return_type = return_type

    def __call__(self, *args, **kwargs) -> type:
        return self.return_type


def get_type_from_args(*args: Expression) -> type:
    types = {arg.type for arg in args}
    if float in types:
        return float
    return args[0].type


def infer_type_from_annotation(node: ast.AST, default: type) -> type:
    if isinstance(node, ast.Name):
        return eval(node.id)
    return default


class Visitor(ast.NodeVisitor):
    """Base AST visitor that traverses a Python FABM model and collects type
    information, delegating the actual Fortran emission to abstract helper
    methods.

    Subclasses must implement the ``translate_*`` methods called from the
    visitor, which receive pre-processed arguments and return :class:`Block`
    instances or strings of Fortran code.

    Attributes:
        functions (set): Names of all callable identifiers that are allowed in
            expressions (pre-populated with ``min`` and ``max``).
        globals (OrderedDict): Module-level constant assignments mapped to
            ``(type, fortran_value)``.
        locals (OrderedDict): Local variable names encountered inside the
            current function body, mapped to their inferred Python types.
        module (ModuleType): The live Python module being translated (used for
            runtime introspection of class metadata such as state variables and
            parameters).
        cls (type[ingredients.Model] or None): The :class:`ingredients.Model`
            subclass currently being visited, or ``None`` when outside a class
            definition.
        args (OrderedDict): Argument names of the current function, mapped to
            their types.
        return_name (str or None): Name used for the Fortran function result
            variable.
        return_type (type or None): Inferred return type of the current
            function.
        self_name (str or None): The name of the first parameter of a class
            method (usually ``'self'``).
        readable (dict): Attribute names readable inside the current class
            (state variables, dependencies, and parameters).
        read (OrderedDict): Subset of *readable* that have actually been
            accessed in the current method body; used to emit ``GET`` macro
            calls.
    """

    def __init__(self, module: ModuleType):
        self.module = module
        self.globals = collections.OrderedDict[str, type]()
        self.locals = collections.OrderedDict[str, type]()
        self.args = collections.OrderedDict[str, type]()
        self.scope = collections.ChainMap(self.locals, self.args, self.globals)

        self.functions = dict[str, Callable[..., type]](
            min=get_type_from_args, max=get_type_from_args
        )

        # context when inside a class or function
        self.cls: type[ingredients.Model] | None = None
        self.return_name: str | None = None
        self.return_type: type | None = None
        self.self_name: str | None = None
        self.readable = {}
        self.read = collections.OrderedDict()

    def visit_Module(self, node: ast.Module) -> Block:
        """Visit the top-level module node.

        Collects module-level constant assignments and function names, then
        dispatches to :meth:`translate_module` with the translated classes and
        functions.
        """
        self.globals.clear()

        # global scalars
        globals = collections.OrderedDict[str, Expression]()
        for child in node.body:
            if isinstance(child, ast.Assign):
                assert len(child.targets) == 1 and isinstance(
                    child.targets[0], ast.Name
                )
                translated_value = self.visit(child.value)
                assert isinstance(translated_value, Expression)
                globals[child.targets[0].id] = translated_value
                self.globals[child.targets[0].id] = translated_value.type

        # global functions
        functions: list[Block] = []
        for child in node.body:
            if isinstance(child, ast.FunctionDef):
                translated_function = self.visit(child)
                functions.append(translated_function)
                self.functions[child.name] = FixedReturnType(
                    translated_function.return_type
                )

        classes: list[tuple[Block, Block]] = []
        for n in node.body:
            if isinstance(n, ast.ClassDef):
                classes.append(self.visit(n))
        return self.translate_module(self.module.__name__, classes, functions, globals)

    def visit_ClassDef(self, node: ast.ClassDef) -> tuple[Block, Block]:
        """Visit a class definition that must subclass :class:`ingredients.Model`.

        Populates *readable* with the class's state variables, dependencies,
        and parameters, visits all method definitions, and delegates to
        :meth:`translate_class`.  Cleans up class-level state on exit.
        """
        assert self.cls is None, "nested classes not supported"
        cls: type = getattr(self.module, node.name)
        assert issubclass(cls, ingredients.Model)
        self.cls = cls
        self.readable.update(self.cls.state_variables)
        self.readable.update(self.cls.dependencies)
        self.readable.update(self.cls.parameters)

        funcs = collections.OrderedDict()
        for n in node.body:
            if isinstance(n, ast.FunctionDef):
                funcs[n.name] = self.visit(n)

        res = self.translate_class(node.name, self.cls, funcs)

        self.cls = None
        self.readable.clear()

        return res

    def visit_FunctionDef(self, node: ast.FunctionDef) -> Block:
        """Visit a function or class-method definition.

        For class methods the method body is translated via
        :meth:`translate_method`; for module-level functions via
        :meth:`translate_function`.  In both cases argument and local-variable
        type information is reset before and after the visit.
        """
        self.locals.clear()
        if self.cls:
            # class method
            self.self_name = node.args.args[0].arg
            self.read.clear()
        else:
            # regular [unbound] function
            for a in node.args.args:
                self.args[a.arg] = infer_type_from_annotation(a.annotation, float)
            self.return_name = node.name

        body_block = Block()
        for n in node.body:
            body_block.extend(self.visit(n))

        if self.cls:
            # class method
            translation = self.translate_method(
                node.name, body_block, self.cls, self.read, self.locals
            )
        else:
            # regular [unbound] function
            return_type = infer_type_from_annotation(node.returns, float)
            translation = self.translate_function(
                node.name, body_block, self.args, return_type, self.locals
            )
            translation.return_type = return_type
            self.args.clear()
            self.return_name = None
        return translation

    def visit_If(self, node: ast.If) -> Block:
        """Translate a Python ``if``/``else`` statement to Fortran ``if/then/end if``."""
        block = Block([f"if ({self.visit(node.test)}) then"])
        for n in node.body:
            block.append(self.visit(n))
        if node.orelse:
            block.append("else")
            for n in node.orelse:
                block.append(self.visit(n))
        block.append("end if")
        return block

    def visit_Return(self, node: ast.Return) -> Block:
        """Translate a ``return`` statement to a Fortran result assignment plus ``return``."""
        assert self.return_name
        return Block([f"{self.return_name} = {self.visit(node.value)}", "return"])

    def visit_Assign(self, node: ast.Assign) -> Block:
        """Translate an assignment statement.

        Handles two cases:

        * Assignment to a plain name — treated as a local variable; the name
          and its inferred type are recorded in *locals*.
        * Assignment to a ``self.<attr>`` attribute — the attribute must be a
          diagnostic variable; delegates to
          :meth:`translate_diagnostic_assignment`.
        """
        assert len(node.targets) == 1, "assign to multipe targets not supported"
        target = node.targets[0]
        if isinstance(target, ast.Name):
            # assign to local name
            translated_value = self.visit(node.value)
            self.locals[target.id] = translated_value.type
            return Block([f"{target.id} = {translated_value}"])
        elif isinstance(target, ast.Attribute):
            # assign to attribute - typically a diagnostic variable
            assert self.cls
            self.ensure_self(target.value)
            diag = self.cls.diagnostic_variables.get(target.attr)
            assert (
                diag
            ), f"Cannot assign to {target.attr} because it is not a diagnostic variable"
            return self.translate_diagnostic_assignment(diag, self.visit(node.value))

    def visit_AnnAssign(self, node: ast.AnnAssign) -> Block:
        """Translate an annotated assignment to a local variable declaration plus assignment."""
        translated_value = self.visit(node.value)
        self.locals[node.target.id] = infer_type_from_annotation(node.annotation, float)
        return Block([f"{node.target.id} = {translated_value}"])

    def visit_AugAssign(self, node: ast.AugAssign) -> Block:
        """Translate an augmented assignment (``+=`` / ``-=``) on a state variable flux.

        Only ``self.<statevar>.source``, ``.bottom_flux``, and
        ``.surface_flux`` are supported.  Delegates to
        :meth:`translate_source_increment`.
        """
        assert self.cls
        assert isinstance(node.target, ast.Attribute)
        assert node.target.attr in ("source", "bottom_flux", "surface_flux")
        owner = node.target.value
        assert isinstance(owner, ast.Attribute)
        statevar = self.cls.state_variables[owner.attr]
        assert (
            isinstance(statevar, ingredients.InteriorStateVariable)
            or node.target.attr == "source"
        )
        assert statevar is not None
        self.ensure_self(owner.value)
        assert isinstance(node.op, (ast.Add, ast.Sub))
        value = self.visit(node.value)
        if isinstance(node.op, ast.Sub):
            value = "-" + value
        return self.translate_source_increment(statevar, value, node.target.attr)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Expression:
        """Translate a unary operation to a parenthesised Fortran expression."""
        operand = self.visit(node.operand)
        assert isinstance(operand, Expression)
        op = UNARYOPS[type(node.op)]
        return Expression(f"({op}{operand})", operand.type)

    def visit_BinOp(self, node: ast.BinOp) -> Expression:
        """Translate a binary operation to a parenthesised Fortran expression."""
        left = self.visit(node.left)
        right = self.visit(node.right)
        assert isinstance(left, Expression) and isinstance(right, Expression)
        op = BINOPS[type(node.op)]
        tp = left.type if left.type == right.type and node.op != ast.Div else float
        return Expression(f"({left} {op} {right})", tp)

    def visit_Compare(self, node: ast.Compare) -> Expression:
        """Translate a comparison expression to a Fortran relational expression."""
        assert len(node.ops) == 1 and len(node.comparators) == 1
        left = self.visit(node.left)
        right = self.visit(node.comparators[0])
        op = CMPOPS[type(node.ops[0])]
        return Expression(f"({left} {op} {right})", bool)

    def visit_Attribute(self, node: ast.Attribute) -> Expression:
        """Translate attribute access.

        * When the attribute is being *called* (flagged by ``node.called``)
          the owning object must be a module and the attribute a NumPy ufunc;
          the corresponding Fortran intrinsic name is returned.
        * Otherwise the attribute is expected to be a readable member of the
          current class (state variable, dependency, or parameter).  Non-
          parameter members are recorded in *read* so that emitter can insert
          the appropriate ``GET`` macro call.
        """
        if isinstance(node.ctx, ast.Load):
            # Attribute is being read from, not assigned to or deleted.
            if getattr(node, "called", False):
                assert isinstance(node.value, ast.Name)
                m = getattr(self.module, node.value.id)
                f = getattr(m, node.attr)
                return Function(NUMPY_UFUNCS[f], lambda *args: float)
            else:
                self.ensure_self(node.value)
                obj = self.readable.get(node.attr)
                assert obj, f"Do not know how to load {node.attr} (line {node.lineno})"
                if isinstance(obj, ingredients.Parameter):
                    return Expression(f"self%{node.attr}", obj.type)
                else:
                    self.read[obj] = True
                    return Expression(node.attr, float)

    def visit_Name(self, node: ast.Name) -> Expression | Function:
        """Return the identifier string, validating that called names are known functions."""
        if node.id in self.scope:
            return Expression(node.id, self.scope[node.id])
        else:
            return Function(node.id, self.functions[node.id])

    def visit_Constant(self, node: ast.Constant) -> Expression:
        """Translate a literal constant via :meth:`translate_constant`."""
        return self.translate_constant(node.value)

    def visit_Call(self, node: ast.Call) -> Expression:
        """Translate a function or method call to a Fortran call expression."""
        node.func.called = True
        fn = self.visit(node.func)
        assert isinstance(fn, Function)
        args = [self.visit(n) for n in node.args]
        return Expression(f"{fn}({', '.join(map(str,args))})", fn.type(*args))

    def ensure_self(self, node: ast.AST):
        """Assert that *node* is a Name node referencing the method's self parameter."""
        assert isinstance(node, ast.Name) and node.id == self.self_name


class FABMTranslator(Visitor):
    """Concrete translator that emits FABM-compatible Fortran source.

    Inherits all AST visiting logic from :class:`Visitor` and provides the
    ``translate_*`` implementations that produce actual Fortran text.

    The generated Fortran follows the FABM conventions:
    * The module includes `fabm_driver.h` and adds ``use fabm_types``
    * Model classes are added as derived types that extend ``type_base_model``
    * Class methods become subroutines that use the
      ``_LOOP_BEGIN_`` / ``_LOOP_END_`` (and bottom/surface variants) macros.
    * State-variable flux increments use the ``_ADD_SOURCE_``-family macros.
    * Diagnostic assignments use the ``_SET_DIAGNOSTIC_``-family macros.
    * Scalar helper functions are emitted as elemental functions.
    """

    METHODMAP = {
        "process_interior": "do",
        "process_bottom": "do_bottom",
        "process_surface": "do_surface",
    }

    def translate_module(
        self,
        name: str,
        classes: Iterable[tuple[Block, Block]],
        functions: Iterable[Block],
        globals: Mapping[str, Expression],
    ) -> Block:
        """Emit the top-level Fortran ``module`` block.

        Args:
            name: Module name (derived from the source file stem).
            classes: Iterable of ``(declaration_block, body_block)`` pairs
                produced by :meth:`translate_class`.
            functions: Iterable of function body :class:`Block` produced by
                :meth:`translate_function`.
            globals: Module-level constants mapping name to :class:`Expression`.
        """
        declarations = Block(
            ["use fabm_types", "implicit none", "private"], separate=True
        )
        for varname, expr in globals.items():
            declarations.append(
                f"{FORTRAN_TYPES[expr.type]}, parameter :: {varname} = {expr}"
            )
        contains = Block(separate=True)
        for class_declarations, class_body in classes:
            declarations.extend(class_declarations)
            contains.extend(class_body)
        for function_body in functions:
            contains.extend(function_body)
        return Block(
            [
                '#include "fabm_driver.h"',
                f"module {name}",
                declarations,
                "contains",
                contains,
                "end module",
            ],
            separate=True,
        )

    def translate_class(
        self, name: str, cls: ingredients.Model, funcs: Mapping[str, Block]
    ) -> tuple[Block, Block]:
        """Emit the Fortran derived-type declaration and its procedure bodies.

        Generates:
        * A ``type, extends(type_base_model)`` declaration with ``id_*``
          fields for each state variable, diagnostic, and dependency, and
          plain scalar fields for each parameter.
        * A ``subroutine initialize`` that calls the FABM ``register_*`` and
          ``get_parameter`` routines.
        * All translated method subroutines.

        Returns:
            tuple: ``(declaration_block, body_block)`` to be inserted into the
                module.
        """
        type_contains = Block(["procedure :: initialize"])
        type_contains.extend(
            f"procedure :: {self.METHODMAP.get(fname, fname)}" for fname in funcs
        )

        func_block = Block()
        for fnbody in funcs.values():
            func_block.extend(fnbody)

        # Translate to Fortran/FABM
        type_members = Block()
        initialize_body = Block(
            [
                f"class(type_{name}), intent(inout), target :: self",
                "integer, intent(in) :: configunit",
            ]
        )
        for vname, var in cls.state_variables.items():
            type_members.append(f"type({var.id_type}) :: id_{vname}")
            initialize_body.append(
                f"call self%register_state_variable("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.diagnostic_variables.items():
            type_members.append(f"type({var.id_type}) :: id_{vname}")
            initialize_body.append(
                f"call self%register_diagnostic_variable("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.dependencies.items():
            type_members.append(f"type({var.id_type}) :: id_{vname}")
            initialize_body.append(
                f"call self%register_dependency("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.parameters.items():
            type_members.append(f"{FORTRAN_TYPES[var.type]} :: {vname}")
            default = self.translate_constant(var.default)
            initialize_body.append(
                f"call self%get_parameter("
                f"self%{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                f", default={default}"
                ")"
            )
        declaration = Block(
            [
                f"type, extends(type_base_model), public :: type_{name}",
                type_members,
                "contains",
                type_contains,
                "end type",
            ]
        )
        procedures = Block(
            [
                "subroutine initialize(self, configunit)",
                initialize_body,
                "end subroutine initialize",
            ]
            + func_block
        )
        return declaration, procedures

    def translate_function(
        self,
        name: str,
        body_block: Block,
        args: Mapping[str, type],
        return_type: type,
        locals: Mapping[str, type],
    ) -> Block:
        """Emit a Fortran ``elemental function`` for a module-level Python function.

        All arguments are declared ``intent(in)``.
        The data type of arguments and the return type are inferred from type annotations
        if available.
        """
        arg_block = Block()
        for argname, argtype in args.items():
            arg_block.append(f"{FORTRAN_TYPES[argtype]}, intent(in) :: {argname}")
        arg_block.extend(f"{FORTRAN_TYPES[t]} :: {n}" for n, t in locals.items())
        strreturn_type = FORTRAN_TYPES[return_type]
        strargs = ", ".join(args)
        return Block(
            [
                f"elemental {strreturn_type} function {name}({strargs})",
                arg_block,
                body_block,
                f"end function {name}",
            ]
        )

    def translate_method(
        self,
        name: str,
        body_block: Block,
        cls: type[ingredients.Model],
        inputs: Iterable[ingredients.Base],
        locals: Mapping[str, type],
    ) -> Block:
        """Emit a FABM ``subroutine`` for a class method.

        Selects the correct FABM loop-macro variant (interior, bottom, or
        surface) based on the method name, emits ``GET`` macros for every
        readable variable that was actually accessed in the method body, and
        wraps the translated body in the appropriate ``_LOOP_BEGIN_`` /
        ``_LOOP_END_`` pair.

        Args:
            name: Python method name (e.g. ``process_interior``, ``process_bottom``, ``process_surface``).
            body_block: Pre-translated :class:`Block` of Fortran statements
                for the body.
            cls: The owning :class:`ingredients.Model` subclass.
            inputs: Ordered collection of :class:`ingredients.Base` objects
                that were read inside the method. For each of these the value will be retrieved
                by emitting an appropriate ``GET`` macro call.
            locals: Local variable names and types declared inside the method.
        """
        fname = self.METHODMAP.get(name, name)
        context = {"process_bottom": "bottom", "process_surface": "surface"}.get(
            name, "interior"
        )
        args, loop = {
            "interior": ("_ARGUMENTS_DO_", "_LOOP_"),
            "bottom": ("_ARGUMENTS_DO_BOTTOM_", "_BOTTOM_LOOP_"),
            "surface": ("_ARGUMENTS_DO_SURFACE_", "_SURFACE_LOOP_"),
        }[context]
        arg_block = Block()
        arg_block.append(f"class (type_{cls.__name__}), intent(in) :: self")
        arg_block.append(f"_DECLARE{args}")
        arg_block.extend(f"real(rk) :: {v.name}" for v in inputs)
        arg_block.extend(f"{FORTRAN_TYPES[t]} :: {n}" for n, t in locals.items())
        get_block = Block()
        for v in inputs:
            get_block.append(f"{v.get_macro}(self%id_{v.name}, {v.name})")
        return Block(
            [
                f"subroutine {fname}(self, {args})",
                arg_block,
                Block(
                    [
                        f"{loop}BEGIN_",
                        get_block,
                        body_block,
                        f"{loop}END_",
                    ]
                ),
                f"end subroutine {fname}",
            ]
        )

    def translate_constant(self, value) -> Expression:
        """Convert a Python literal value to its Fortran source representation.

        * ``float`` values are suffixed with ``_rk``.
        * ``bool`` values become ``.true.`` / ``.false.``.
        * All other values are rendered with ``repr()``.
        """
        if isinstance(value, float):
            return Expression(f"{value}_rk", float)
        elif isinstance(value, bool):
            return Expression(".true." if value else ".false.", bool)
        else:
            return Expression(repr(value), type(value))

    def translate_diagnostic_assignment(
        self, var: ingredients.BaseDiagnosticVariable, value: str
    ) -> Block:
        """Return the Fortran ``SET_DIAGNOSTIC`` macro call for *var*."""
        return Block([f"{var.set_macro}(self%id_{var.name}, {value})"])

    def translate_source_increment(
        self, var: ingredients.StateVariable, value, type: str = "source"
    ) -> Block:
        """Return the Fortran ``ADD_SOURCE``-family macro call for *var*.

        Args:
            var: The state variable whose source is being modified.
            value: The Fortran expression string to add.
            type: One of ``'source'``, ``'bottom_flux'``, or
                ``'surface_flux'``; selects the appropriate FABM macro.
        """
        return Block([f"{var.add_source_macros[type]}(self%id_{var.name}, {value})"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Python file to convert to FABM/Fortran")
    parser.add_argument(
        "out",
        help="Fortran file to write to (default to stdout)",
        nargs="?",
        default=None,
    )
    args = parser.parse_args()
    translate(args.file, args.out)
