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
    Unk: Final[Operation]
    VarianceShadow: Final[Operation]
    def __eq__(self, /, other: Operation | int) -> bool: ...
    def __int__(self, /) -> int: ...
    def __ne__(self, /, other: Operation | int) -> bool: ...
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
    def output_dependencies(self, /) -> dict[str, int]: ...
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
class Value:
    def attribute(self, /) -> Attribute | None: ...
    def float(self, /) -> float | None: ...
    def int(self, /) -> int | None: ...
    def parameter(self, /) -> Parameter | None: ...
    def texture(self, /) -> Texture | None: ...
