from collections.abc import Sequence
from typing import Final, final

@final
class Attribute:
    @property
    def channel(self, /) -> str | None: ...
    @channel.setter
    def channel(self, /, value: str | None) -> None: ...
    @property
    def name(self, /) -> str: ...
    @name.setter
    def name(self, /, value: str) -> None: ...

@final
class AttributeXyz:
    @property
    def channel(self, /) -> ChannelXyz | None: ...
    @channel.setter
    def channel(self, /, value: ChannelXyz | None) -> None: ...
    @property
    def name(self, /) -> str: ...
    @name.setter
    def name(self, /, value: str) -> None: ...

@final
class ChannelXyz:
    W: Final[ChannelXyz]
    X: Final[ChannelXyz]
    Xyz: Final[ChannelXyz]
    Y: Final[ChannelXyz]
    Z: Final[ChannelXyz]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class Operation:
    Abs: Final[Operation]
    Add: Final[Operation]
    AnisotropicSpecular: Final[Operation]
    BlinnPhongSpecular: Final[Operation]
    Clamp: Final[Operation]
    Cos: Final[Operation]
    Div: Final[Operation]
    Dot: Final[Operation]
    Equal: Final[Operation]
    Exp2: Final[Operation]
    FloatBitsToInt: Final[Operation]
    Floor: Final[Operation]
    Fma: Final[Operation]
    Fract: Final[Operation]
    Fresnel: Final[Operation]
    Greater: Final[Operation]
    GreaterEqual: Final[Operation]
    IntBitsToFloat: Final[Operation]
    InverseSqrt: Final[Operation]
    Less: Final[Operation]
    LessEqual: Final[Operation]
    LocalToWorldPointX: Final[Operation]
    LocalToWorldPointY: Final[Operation]
    LocalToWorldPointZ: Final[Operation]
    LocalToWorldVectorX: Final[Operation]
    LocalToWorldVectorY: Final[Operation]
    LocalToWorldVectorZ: Final[Operation]
    Log2: Final[Operation]
    Max: Final[Operation]
    Min: Final[Operation]
    Mix: Final[Operation]
    Mul: Final[Operation]
    Negate: Final[Operation]
    NormalMapX: Final[Operation]
    NormalMapY: Final[Operation]
    NormalMapZ: Final[Operation]
    NormalizeX: Final[Operation]
    NormalizeY: Final[Operation]
    NormalizeZ: Final[Operation]
    NotEqual: Final[Operation]
    Power: Final[Operation]
    Select: Final[Operation]
    Sin: Final[Operation]
    SphereMapCoordX: Final[Operation]
    SphereMapCoordY: Final[Operation]
    Sqrt: Final[Operation]
    Sub: Final[Operation]
    TintColorX: Final[Operation]
    TintColorY: Final[Operation]
    TintColorZ: Final[Operation]
    Unk: Final[Operation]
    VarianceShadow: Final[Operation]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class OperationXyz:
    Abs: Final[OperationXyz]
    Add: Final[OperationXyz]
    AnisotropicSpecular: Final[OperationXyz]
    BlinnPhongSpecular: Final[OperationXyz]
    Clamp: Final[OperationXyz]
    Cos: Final[OperationXyz]
    Div: Final[OperationXyz]
    Dot: Final[OperationXyz]
    Equal: Final[OperationXyz]
    Exp2: Final[OperationXyz]
    FloatBitsToInt: Final[OperationXyz]
    Floor: Final[OperationXyz]
    Fma: Final[OperationXyz]
    Fract: Final[OperationXyz]
    Fresnel: Final[OperationXyz]
    Greater: Final[OperationXyz]
    GreaterEqual: Final[OperationXyz]
    IntBitsToFloat: Final[OperationXyz]
    InverseSqrt: Final[OperationXyz]
    Less: Final[OperationXyz]
    LessEqual: Final[OperationXyz]
    LocalToWorldPoint: Final[OperationXyz]
    LocalToWorldVector: Final[OperationXyz]
    Log2: Final[OperationXyz]
    Max: Final[OperationXyz]
    Min: Final[OperationXyz]
    Mix: Final[OperationXyz]
    Mul: Final[OperationXyz]
    Negate: Final[OperationXyz]
    NormalMap: Final[OperationXyz]
    Normalize: Final[OperationXyz]
    NotEqual: Final[OperationXyz]
    Power: Final[OperationXyz]
    Select: Final[OperationXyz]
    Sin: Final[OperationXyz]
    Sqrt: Final[OperationXyz]
    Sub: Final[OperationXyz]
    TintColor: Final[OperationXyz]
    Unk: Final[OperationXyz]
    VarianceShadow: Final[OperationXyz]
    def __eq__(self, /, other: object) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: object) -> bool: ...
    def __repr__(self, /) -> str: ...

@final
class OutputExpr:
    def func(self, /) -> OutputExprFunc | None: ...
    def value(self, /) -> Value | None: ...

@final
class OutputExprFunc:
    @property
    def args(self, /) -> list[int]: ...
    @args.setter
    def args(self, /, value: Sequence[int]) -> None: ...
    @property
    def op(self, /) -> Operation: ...
    @op.setter
    def op(self, /, value: Operation) -> None: ...

@final
class OutputExprFuncXyz:
    @property
    def args(self, /) -> list[int]: ...
    @args.setter
    def args(self, /, value: Sequence[int]) -> None: ...
    @property
    def op(self, /) -> OperationXyz: ...
    @op.setter
    def op(self, /, value: OperationXyz) -> None: ...

@final
class OutputExprXyz:
    def func(self, /) -> OutputExprFuncXyz | None: ...
    def value(self, /) -> ValueXyz | None: ...

@final
class Parameter:
    @property
    def channel(self, /) -> str | None: ...
    @channel.setter
    def channel(self, /, value: str | None) -> None: ...
    @property
    def field(self, /) -> str: ...
    @field.setter
    def field(self, /, value: str) -> None: ...
    @property
    def index(self, /) -> int | None: ...
    @index.setter
    def index(self, /, value: int | None) -> None: ...
    @property
    def name(self, /) -> str: ...
    @name.setter
    def name(self, /, value: str) -> None: ...

@final
class ParameterXyz:
    @property
    def channel(self, /) -> ChannelXyz | None: ...
    @channel.setter
    def channel(self, /, value: ChannelXyz | None) -> None: ...
    @property
    def field(self, /) -> str: ...
    @field.setter
    def field(self, /, value: str) -> None: ...
    @property
    def index(self, /) -> int | None: ...
    @index.setter
    def index(self, /, value: int | None) -> None: ...
    @property
    def name(self, /) -> str: ...
    @name.setter
    def name(self, /, value: str) -> None: ...

@final
class ShaderDatabase:
    @staticmethod
    def from_file(path: str) -> ShaderDatabase: ...
    def get_shader(self, /, shader_id: int) -> ShaderProgram | None: ...

@final
class ShaderProgram:
    @property
    def attributes(self, /) -> list[str]: ...
    @property
    def exprs(self, /) -> list[OutputExpr]: ...
    @property
    def exprs_xyz(self, /) -> list[OutputExprXyz]: ...
    @property
    def output_dependencies(self, /) -> dict[str, int]: ...
    @property
    def output_dependencies_xyz(self, /) -> dict[str, int]: ...
    def parameter_value(self, /, parameter: Parameter) -> float | None: ...
    @property
    def parameters(self, /) -> list[str]: ...
    @property
    def samplers(self, /) -> list[str]: ...

@final
class Texture:
    @property
    def channel(self, /) -> str | None: ...
    @channel.setter
    def channel(self, /, value: str | None) -> None: ...
    @property
    def name(self, /) -> str: ...
    @name.setter
    def name(self, /, value: str) -> None: ...
    @property
    def texcoords(self, /) -> list[int]: ...
    @texcoords.setter
    def texcoords(self, /, value: Sequence[int]) -> None: ...

@final
class TextureXyz:
    @property
    def channel(self, /) -> ChannelXyz | None: ...
    @channel.setter
    def channel(self, /, value: ChannelXyz | None) -> None: ...
    @property
    def name(self, /) -> str: ...
    @name.setter
    def name(self, /, value: str) -> None: ...
    @property
    def texcoords(self, /) -> list[int]: ...
    @texcoords.setter
    def texcoords(self, /, value: Sequence[int]) -> None: ...

@final
class Value:
    def attribute(self, /) -> Attribute | None: ...
    def float(self, /) -> float | None: ...
    def int(self, /) -> int | None: ...
    def parameter(self, /) -> Parameter | None: ...
    def texture(self, /) -> Texture | None: ...

@final
class ValueXyz:
    def attribute(self, /) -> AttributeXyz | None: ...
    def float(self, /) -> tuple[float, float, float] | None: ...
    def parameter(self, /) -> ParameterXyz | None: ...
    def texture(self, /) -> TextureXyz | None: ...
