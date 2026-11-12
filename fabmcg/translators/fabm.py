from . import Visitor, Block, Expression, Any
import ast
from collections.abc import Iterable, Mapping
import argparse

import numpy as np

from .. import ingredients


# Mapping from Python built-in types to their Fortran equivalents.
FORTRAN_TYPES = {float: "real(rk)", int: "integer", bool: "logical"}


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

    def make_expression(self, value: Any, tp: type) -> Expression:
        assert tp in FORTRAN_TYPES, f"Unsupported type {tp}"
        return super().make_expression(value, tp)

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
    FABMTranslator.translate(args.file, args.out)
