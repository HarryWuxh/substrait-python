import substrait.algebra_pb2 as stalg
import substrait.extended_expression_pb2 as stee
import substrait.extensions.extensions_pb2 as ste
import substrait.type_pb2 as stt

from substrait.builders import type as types
from substrait.builders.extended_expression import (
    column,
    if_then,
    literal,
    scalar_function,
)
from substrait.extension_registry import ExtensionRegistry
from substrait.type_inference import infer_expression_type

struct = stt.Type.Struct(
    types=[
        stt.Type(i64=stt.Type.I64(nullability=stt.Type.NULLABILITY_REQUIRED)),
        stt.Type(string=stt.Type.String(nullability=stt.Type.NULLABILITY_NULLABLE)),
        stt.Type(fp32=stt.Type.FP32(nullability=stt.Type.NULLABILITY_NULLABLE)),
    ]
)

named_struct = stt.NamedStruct(
    names=["order_id", "description", "order_total"], struct=struct
)


def test_nullable_return_branches_lift_only_the_literal_type():
    schema = types.named_struct(
        ["count"], types.struct([types.i64(True)], nullable=False)
    )
    condition = literal(False, types.boolean(False))
    zero = literal(0, types.i64(False))(schema, None)
    inner = if_then([(condition, zero)], column("count"))
    outer = if_then(
        [(condition, inner), (condition, zero)], literal(None, types.i64(True))
    )

    bound = outer(schema, None).referred_expr[0].expression
    first, second = bound.if_then.ifs
    inner_zero = first.then.if_then.ifs[0].then.literal
    assert inner_zero.i64 == second.then.literal.i64 == 0
    assert inner_zero.nullable and second.then.literal.nullable
    assert not getattr(first, "if").literal.boolean
    assert getattr(bound.if_then, "else").literal.WhichOneof("literal_type") == "null"
    assert infer_expression_type(bound, schema.struct) == types.i64(True)
    assert not zero.referred_expr[0].expression.literal.nullable
    assert outer(schema, None).referred_expr[0].expression == bound

    reverse = if_then([(condition, literal(None, types.i64(True)))], zero)
    reversed_if = reverse(schema, None).referred_expr[0].expression.if_then
    assert getattr(reversed_if, "else").literal.nullable


def test_mixed_branches_stay_intact_if_a_required_branch_is_not_a_liftable_literal():
    schema = types.named_struct(
        ["count"], types.struct([types.i64(False)], nullable=False)
    )
    condition = literal(True, types.boolean(False))
    conditional = (
        if_then(
            [(condition, literal(0, types.i64(False))), (condition, column("count"))],
            literal(None, types.i64(True)),
        )(schema, None)
        .referred_expr[0]
        .expression.if_then
    )
    assert not conditional.ifs[0].then.literal.nullable
    assert conditional.ifs[1].then.HasField("selection")


def test_literal_alignment_preserves_distinct_base_types_and_parameters():
    from datetime import datetime

    condition = literal(True, types.boolean(False))
    for value, required, nullable in (
        (0, types.i32(False), types.i64(True)),
        (
            datetime(2026, 11, 19),
            types.precision_timestamp(6, False),
            types.precision_timestamp(3, True),
        ),
    ):
        conditional = if_then(
            [(condition, literal(value, required))], literal(None, nullable)
        )
        bound = conditional(named_struct, None).referred_expr[0].expression
        assert not bound.if_then.ifs[0].then.literal.nullable
        assert infer_expression_type(bound, named_struct.struct) == required


def test_if_else():
    actual = if_then(
        ifs=[
            (
                literal(
                    True,
                    type=stt.Type(
                        bool=stt.Type.Boolean(nullability=stt.Type.NULLABILITY_REQUIRED)
                    ),
                ),
                literal(
                    10,
                    type=stt.Type(
                        i8=stt.Type.I8(nullability=stt.Type.NULLABILITY_REQUIRED)
                    ),
                ),
            )
        ],
        _else=literal(
            20, type=stt.Type(i8=stt.Type.I8(nullability=stt.Type.NULLABILITY_REQUIRED))
        ),
    )(named_struct, None)

    expected = stee.ExtendedExpression(
        referred_expr=[
            stee.ExpressionReference(
                expression=stalg.Expression(
                    if_then=stalg.Expression.IfThen(
                        **{
                            "ifs": [
                                stalg.Expression.IfThen.IfClause(
                                    **{
                                        "if": stalg.Expression(
                                            literal=stalg.Expression.Literal(
                                                boolean=True, nullable=False
                                            )
                                        ),
                                        "then": stalg.Expression(
                                            literal=stalg.Expression.Literal(
                                                i8=10, nullable=False
                                            )
                                        ),
                                    }
                                )
                            ],
                            "else": stalg.Expression(
                                literal=stalg.Expression.Literal(i8=20, nullable=False)
                            ),
                        }
                    )
                ),
                output_names=["IfThen(Literal(True),Literal(10),Literal(20))"],
            )
        ],
        base_schema=named_struct,
    )

    assert actual == expected


def test_if_then_with_extension():
    import yaml

    registry = ExtensionRegistry(load_default_extensions=False)
    content = """%YAML 1.2
---
urn: extension:io.substrait:functions_comparison
scalar_functions:
  - name: "gt"
    description: ""
    impls:
      - args:
          - value: fp32
          - value: fp32
        return: boolean
"""
    registry.register_extension_dict(yaml.safe_load(content))

    # Create if_then: if order_total > 100 then "expensive" else "cheap"
    actual = if_then(
        ifs=[
            (
                scalar_function(
                    "extension:io.substrait:functions_comparison",
                    "gt",
                    expressions=[
                        column("order_total"),
                        literal(
                            100.0,
                            type=stt.Type(
                                fp32=stt.Type.FP32(
                                    nullability=stt.Type.NULLABILITY_REQUIRED
                                )
                            ),
                        ),
                    ],
                ),
                literal(
                    "expensive",
                    type=stt.Type(
                        string=stt.Type.String(
                            nullability=stt.Type.NULLABILITY_REQUIRED
                        )
                    ),
                ),
            )
        ],
        _else=literal(
            "cheap",
            type=stt.Type(
                string=stt.Type.String(nullability=stt.Type.NULLABILITY_REQUIRED)
            ),
        ),
    )(named_struct, registry)

    expected = stee.ExtendedExpression(
        extension_urns=[
            ste.SimpleExtensionURN(
                extension_urn_anchor=1,
                urn="extension:io.substrait:functions_comparison",
            )
        ],
        extensions=[
            ste.SimpleExtensionDeclaration(
                extension_function=ste.SimpleExtensionDeclaration.ExtensionFunction(
                    extension_urn_reference=1,
                    function_anchor=1,
                    name="gt:fp32_fp32",
                )
            )
        ],
        referred_expr=[
            stee.ExpressionReference(
                expression=stalg.Expression(
                    if_then=stalg.Expression.IfThen(
                        **{
                            "ifs": [
                                stalg.Expression.IfThen.IfClause(
                                    **{
                                        "if": stalg.Expression(
                                            scalar_function=stalg.Expression.ScalarFunction(
                                                function_reference=1,
                                                output_type=stt.Type(
                                                    bool=stt.Type.Boolean(
                                                        nullability=stt.Type.NULLABILITY_NULLABLE
                                                    )
                                                ),
                                                arguments=[
                                                    stalg.FunctionArgument(
                                                        value=stalg.Expression(
                                                            selection=stalg.Expression.FieldReference(
                                                                direct_reference=stalg.Expression.ReferenceSegment(
                                                                    struct_field=stalg.Expression.ReferenceSegment.StructField(
                                                                        field=2
                                                                    )
                                                                ),
                                                                root_reference=stalg.Expression.FieldReference.RootReference(),
                                                            )
                                                        )
                                                    ),
                                                    stalg.FunctionArgument(
                                                        value=stalg.Expression(
                                                            literal=stalg.Expression.Literal(
                                                                fp32=100.0
                                                            )
                                                        )
                                                    ),
                                                ],
                                            )
                                        ),
                                        "then": stalg.Expression(
                                            literal=stalg.Expression.Literal(
                                                string="expensive"
                                            )
                                        ),
                                    }
                                )
                            ],
                            "else": stalg.Expression(
                                literal=stalg.Expression.Literal(string="cheap")
                            ),
                        }
                    )
                ),
                output_names=[
                    "IfThen(gt(order_total,Literal(100.0)),Literal(expensive),Literal(cheap))"
                ],
            )
        ],
        base_schema=named_struct,
    )

    assert actual == expected
