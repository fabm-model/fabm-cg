"""Translate a Python-based FABM model description into a Fortran/FABM module.

The translator loads a Python source file that defines one or more
:class:`ingredients.Model` subclasses and optional helper functions, parses it
into an abstract syntax tree (AST), and emits an equivalent Fortran 90 source
file that is compatible with the Framework for Aquatic Biogeochemical Models
(FABM).

Example:
    From the command line::

        python -m fabmcg.translators.fabm ergom.py ergom.F90

    Programmatically::

        from fabmcg.translators.fabm import Translator
        Translator.translate("ergom.py", "ergom.F90")

Key components:
    * :class:`Translator` — concrete subclass of :class:`Visitor` that emits
      FABM-specific Fortran constructs (``type_base_model`` extensions, loop
      macros, registration calls, …).
"""

import ast
from collections.abc import Iterable, Mapping
import argparse

import numpy as np

from .. import ingredients
from . import Visitor, Block, Expression, Any

# Mapping from Python built-in types to their Fortran equivalents.
FORTRAN_TYPES = {float: "real(rk)", int: "integer", bool: "logical"}


class Translator(Visitor):
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
        module_block = Block(
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
        module_block.name = name
        module_block.model_types = [d.name for d, _ in classes]
        return module_block

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
        # Note that calls to registration routines (register_state_variable,
        # register_diagnostic_variable, register_dependency) do not need to include
        # the domain, as they are implemented as generic procedures in FABM.
        initialize_body = Block(
            [
                f"class(type_{name}), intent(inout), target :: self",
                "integer, intent(in) :: configunit",
            ]
        )
        type_members: dict[str, str] = {}

        def add_member(ftype, name):
            key = name.lower()
            if key in type_members:
                raise ValueError(f"Duplicate type member: {name}")
            type_members[key] = ftype

        for vname, var in cls.state_variables.items():
            add_member(f"type({var.id_type})", f"id_{vname}")
            initialize_body.append(
                f"call self%register_state_variable("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.diagnostic_variables.items():
            add_member(f"type({var.id_type})", f"id_{vname}")
            initialize_body.append(
                f"call self%register_diagnostic_variable("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.dependencies.items():
            add_member(f"type({var.id_type})", f"id_{vname}")
            initialize_body.append(
                f"call self%register_dependency("
                f"self%id_{vname}, '{vname}', '{var.units}', '{var.long_name}'"
                ")"
            )
        for vname, var in cls.parameters.items():
            add_member(FORTRAN_TYPES[var.type], vname)
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
                Block(f"{t} :: {n}" for n, t in type_members.items()),
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
        declaration.name = name
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
        DOMAINS = {"process_bottom": "bottom", "process_surface": "surface"}
        domain = DOMAINS.get(name, "interior")
        args, loop = {
            "interior": ("_ARGUMENTS_DO_", "_LOOP_"),
            "bottom": ("_ARGUMENTS_DO_BOTTOM_", "_BOTTOM_LOOP_"),
            "surface": ("_ARGUMENTS_DO_SURFACE_", "_SURFACE_LOOP_"),
        }[domain]
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


def generate_institute_files(institute_name: str, block: Block):
    cases = "\n".join(
        [
            f"         case ('{name}'); allocate(type_{name}::model)"
            for name in getattr(block, "model_types", [])
        ]
    )
    with open(f"{institute_name}_model_library.F90", "w") as f:
        f.write(f"""
module {institute_name}_model_library

   use fabm_types, only: type_base_model_factory, type_base_model

   implicit none

   private

   type, extends(type_base_model_factory) :: type_factory
   contains
      procedure :: create
   end type

   type (type_factory), save, target, public :: {institute_name}_model_factory

contains

   subroutine create(self, name, model)

      use {block.name}
      ! Add new {institute_name} models here

      class (type_factory), intent(in) :: self
      character(*),         intent(in) :: name
      class (type_base_model), pointer :: model

      select case (name)
{cases}
         ! Add new {institute_name} models here
      end select

   end subroutine

end module
""")
    with open(f"CMakeLists.txt", "w") as f:
        f.write(f"""
add_library(fabm_models_{institute_name} OBJECT
            {institute_name}_model_library.F90
            {block.file}
           )

target_link_libraries(fabm_models_{institute_name} PRIVATE fabm_base)
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="Python file to convert to FABM/Fortran")
    parser.add_argument(
        "out",
        help="Fortran file to write to (default to stdout)",
        nargs="?",
        default=None,
    )
    parser.add_argument(
        "--institute",
        help="Create institute files (library, CMakeLists.txt)",
        default=None,
    )
    args = parser.parse_args()
    block = Translator.translate(args.file, args.out)

    if args.institute:
        block.file = args.out
        generate_institute_files(args.institute, block)
