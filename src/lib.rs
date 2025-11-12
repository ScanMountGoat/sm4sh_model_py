use pyo3::prelude::*;

#[macro_export]
macro_rules! python_enum {
    ($py_ty:ident, $rust_ty:ty, $( $i:ident ),+) => {
        #[pyclass(eq, eq_int)]
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum $py_ty {
            $($i),*
        }

        // These will generate a compile error if variant names don't match.
        impl From<$rust_ty> for $py_ty {
            fn from(value: $rust_ty) -> Self {
                match value {
                    $(<$rust_ty>::$i => Self::$i),*
                }
            }
        }

        impl From<$py_ty> for $rust_ty {
            fn from(value: $py_ty) -> Self {
                match value {
                    $(<$py_ty>::$i => Self::$i),*
                }
            }
        }

        impl ::map_py::MapPy<$rust_ty> for $py_ty {
            fn map_py(self, _py: Python) -> PyResult<$rust_ty> {
                Ok(self.into())
            }
        }

        impl ::map_py::MapPy<$py_ty> for $rust_ty {
            fn map_py(self, _py: Python) -> PyResult<$py_ty> {
                Ok(self.into())
            }
        }
    };
}

python_enum!(
    BoneFlags,
    sm4sh_model::BoneFlags,
    Disabled,
    Skinning,
    ParentBone
);

python_enum!(
    PrimitiveType,
    sm4sh_model::PrimitiveType,
    TriangleList,
    TriangleStrip
);

python_enum!(
    SrcFactor,
    sm4sh_model::SrcFactor,
    One,
    SourceAlpha,
    One2,
    SourceAlpha2,
    Zero,
    SourceAlpha3,
    DestinationAlpha,
    DestinationAlpha7,
    DestinationColor,
    SrcAlpha3,
    SrcAlpha4,
    Unk16,
    Unk33,
    SrcAlpha5
);

python_enum!(
    DstFactor,
    sm4sh_model::DstFactor,
    Zero,
    OneMinusSourceAlpha,
    One,
    OneReverseSubtract,
    SourceAlpha,
    SourceAlphaReverseSubtract,
    OneMinusDestinationAlpha,
    One2,
    Zero2,
    Unk10,
    OneMinusSourceAlpha2,
    One3,
    Zero5,
    Zero3,
    One4,
    OneMinusSourceAlpha3,
    One5
);

python_enum!(
    AlphaFunc,
    sm4sh_model::AlphaFunc,
    Disabled,
    Never,
    Less,
    Equal,
    Greater,
    NotEqual,
    GreaterEqual,
    Always
);

python_enum!(
    CullMode,
    sm4sh_model::CullMode,
    Disabled,
    Outside,
    Inside,
    Disabled2,
    Inside2,
    Outside2
);

python_enum!(
    MapMode,
    sm4sh_model::MapMode,
    TexCoord,
    EnvCamera,
    Projection,
    EnvLight,
    EnvSpec
);

python_enum!(
    MinFilter,
    sm4sh_model::MinFilter,
    LinearMipmapLinear,
    Nearest,
    Linear,
    NearestMipmapLinear
);

python_enum!(MagFilter, sm4sh_model::MagFilter, Unk0, Nearest, Linear);

python_enum!(
    MipDetail,
    sm4sh_model::MipDetail,
    OneMipLevelAnisotropicOff,
    Unk1,
    OneMipLevelAnisotropicOff2,
    FourMipLevels,
    FourMipLevelsAnisotropic,
    FourMipLevelsTrilinear,
    FourMipLevelsTrilinearAnisotropic
);

python_enum!(
    WrapMode,
    sm4sh_model::WrapMode,
    Repeat,
    MirroredRepeat,
    ClampToEdge
);

python_enum!(
    NutFormat,
    sm4sh_model::NutFormat,
    BC1Unorm,
    BC2Unorm,
    BC3Unorm,
    Bgr5A1Unorm,
    Bgr5A1Unorm2,
    B5G6R5Unorm,
    Rgb5A1Unorm,
    Rgba8Unorm,
    R32Float,
    Rgba82,
    BC5Unorm
);

python_enum!(
    BoneType,
    sm4sh_model::BoneType,
    Normal,
    Follow,
    Helper,
    Swing
);

python_enum!(
    BoneElementType,
    sm4sh_model::vertex::BoneElementType,
    Float32,
    Float16,
    Byte
);

python_enum!(
    ColorElementType,
    sm4sh_model::vertex::ColorElementType,
    Byte,
    Float16
);

python_enum!(
    Operation,
    sm4sh_model::database::Operation,
    Unk,
    Add,
    Sub,
    Mul,
    Div,
    Mix,
    Clamp,
    Min,
    Max,
    Abs,
    Floor,
    Power,
    Sqrt,
    InverseSqrt,
    Fma,
    Dot4,
    Sin,
    Cos,
    Exp2,
    Log2,
    Fract,
    IntBitsToFloat,
    FloatBitsToInt,
    Select,
    Negate,
    Equal,
    NotEqual,
    Less,
    Greater,
    LessEqual,
    GreaterEqual,
    NormalMapX,
    NormalMapY,
    NormalMapZ,
    NormalizeX,
    NormalizeY,
    NormalizeZ,
    SphereMapCoordX,
    SphereMapCoordY
);

// Match the module hierarchy and types of sm4sh_model as closely as possible.
#[pymodule]
mod sm4sh_model_py {
    use super::*;

    use std::io::Cursor;
    use std::ops::Deref;

    use map_py::helpers::{from_option_py, into_option_py};
    use map_py::{MapPy, TypedList};
    use numpy::{PyArray1, PyArray3};
    use pyo3::types::PyBytes;
    use rayon::prelude::*;

    #[pyfunction]
    fn load_model(py: Python, path: &str) -> PyResult<NudModel> {
        // TODO: Create an error type.
        let model = sm4sh_model::load_model(path).unwrap();
        model.map_py(py)
    }

    #[pyclass]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_lib::nud::Nud)]
    pub struct Nud(sm4sh_lib::nud::Nud);

    #[pymethods]
    impl Nud {
        fn save(&self, path: &str) -> PyResult<()> {
            self.0.save(path).map_err(Into::into)
        }
    }

    #[pyfunction]
    fn decode_images_png(
        py: Python,
        image_textures: Vec<PyRef<ImageTexture>>,
    ) -> PyResult<Vec<Py<PyBytes>>> {
        // TODO: avoid unwrap.
        let textures: Vec<&ImageTexture> = image_textures.iter().map(|i| i.deref()).collect();
        let buffers = textures
            .par_iter()
            .map(|image| {
                // Create the surface manually to avoid copies.
                let format: sm4sh_model::NutFormat = image.image_format.into();
                let surface = image_dds::Surface {
                    width: image.width,
                    height: image.height,
                    depth: 1,
                    layers: 1,
                    mipmaps: image.mipmap_count,
                    image_format: format.into(),
                    data: &image.image_data,
                };

                Ok(surface
                    .decode_layers_mipmaps_rgba8(0..surface.layers, 0..1)
                    .unwrap()
                    .data)
            })
            .collect::<PyResult<Vec<_>>>()?;

        buffers
            .into_iter()
            .zip(textures)
            .map(|(buffer, texture)| {
                let mut writer = Cursor::new(Vec::new());
                let image =
                    image_dds::image::RgbaImage::from_raw(texture.width, texture.height, buffer)
                        .unwrap();
                image
                    .write_to(&mut writer, image_dds::image::ImageFormat::Png)
                    .unwrap();

                Ok(PyBytes::new(py, &writer.into_inner()).into())
            })
            .collect()
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::NudModel)]
    pub struct NudModel {
        pub groups: TypedList<NudMeshGroup>,
        pub textures: TypedList<ImageTexture>,
        pub bounding_sphere: [f32; 4],
        #[map(from(into_option_py), into(from_option_py))]
        pub skeleton: Option<Py<VbnSkeleton>>,
    }

    #[pymethods]
    impl NudModel {
        #[new]
        fn new(
            groups: TypedList<NudMeshGroup>,
            textures: TypedList<ImageTexture>,
            bounding_sphere: [f32; 4],
            skeleton: Option<Py<VbnSkeleton>>,
        ) -> Self {
            Self {
                groups,
                textures,
                bounding_sphere,
                skeleton,
            }
        }

        fn to_nud(&self, py: Python) -> PyResult<Nud> {
            // TODO: Avoid unwrap.
            let model: sm4sh_model::NudModel = self.clone().map_py(py)?;
            let nud = model.to_nud().unwrap();
            Ok(Nud(nud))
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::NudMeshGroup)]
    pub struct NudMeshGroup {
        pub name: String,
        pub meshes: TypedList<NudMesh>,
        pub sort_bias: f32,
        pub bounding_sphere: [f32; 4],
        pub parent_bone_index: Option<usize>,
    }

    #[pymethods]
    impl NudMeshGroup {
        #[new]
        fn new(
            name: String,
            meshes: TypedList<NudMesh>,
            sort_bias: f32,
            bounding_sphere: [f32; 4],
            parent_bone_index: Option<usize>,
        ) -> Self {
            Self {
                name,
                meshes,
                sort_bias,
                bounding_sphere,
                parent_bone_index,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::NudMesh)]
    pub struct NudMesh {
        pub vertices: vertex::Vertices,
        pub vertex_indices: Py<PyArray1<u16>>,
        pub primitive_type: PrimitiveType,
        pub material1: Option<NudMaterial>,
        pub material2: Option<NudMaterial>,
        pub material3: Option<NudMaterial>,
        pub material4: Option<NudMaterial>,
    }

    #[pymethods]
    impl NudMesh {
        #[new]
        fn new(
            vertices: vertex::Vertices,
            vertex_indices: Py<PyArray1<u16>>,
            primitive_type: PrimitiveType,
            material1: Option<NudMaterial>,
            material2: Option<NudMaterial>,
            material3: Option<NudMaterial>,
            material4: Option<NudMaterial>,
        ) -> Self {
            Self {
                vertices,
                vertex_indices,
                primitive_type,
                material1,
                material2,
                material3,
                material4,
            }
        }

        pub fn triangle_list_indices(&self, py: Python) -> PyResult<Py<PyArray1<u16>>> {
            let mesh: sm4sh_model::NudMesh = self.clone().map_py(py)?;
            mesh.triangle_list_indices().to_vec().map_py(py)
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::NudMaterial)]
    pub struct NudMaterial {
        pub shader_id: u32,
        pub src_factor: SrcFactor,
        pub dst_factor: DstFactor,
        pub alpha_func: AlphaFunc,
        pub alpha_test_ref: u16,
        pub cull_mode: CullMode,
        pub textures: TypedList<NudTexture>,
        pub properties: TypedList<NudProperty>,
    }

    #[pymethods]
    impl NudMaterial {
        #[new]
        fn new(
            shader_id: u32,
            src_factor: SrcFactor,
            dst_factor: DstFactor,
            alpha_func: AlphaFunc,
            alpha_test_ref: u16,
            cull_mode: CullMode,
            textures: TypedList<NudTexture>,
            properties: TypedList<NudProperty>,
        ) -> Self {
            Self {
                shader_id,
                src_factor,
                dst_factor,
                alpha_func,
                alpha_test_ref,
                cull_mode,
                textures,
                properties,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::NudTexture)]
    pub struct NudTexture {
        pub hash: u32,
        pub map_mode: MapMode,
        pub wrap_mode_s: WrapMode,
        pub wrap_mode_t: WrapMode,
        pub min_filter: MinFilter,
        pub mag_filter: MagFilter,
        pub mip_detail: MipDetail,
    }

    #[pymethods]
    impl NudTexture {
        #[new]
        fn new(
            hash: u32,
            map_mode: MapMode,
            wrap_mode_s: WrapMode,
            wrap_mode_t: WrapMode,
            min_filter: MinFilter,
            mag_filter: MagFilter,
            mip_detail: MipDetail,
        ) -> Self {
            Self {
                hash,
                map_mode,
                wrap_mode_s,
                wrap_mode_t,
                min_filter,
                mag_filter,
                mip_detail,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::NudProperty)]
    pub struct NudProperty {
        pub name: String,
        pub values: Vec<f32>,
    }

    #[pymethods]
    impl NudProperty {
        #[new]
        fn new(name: String, values: Vec<f32>) -> Self {
            Self { name, values }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::ImageTexture)]
    pub struct ImageTexture {
        pub hash_id: u32,
        pub width: u32,
        pub height: u32,
        pub mipmap_count: u32,
        pub layers: u32,
        pub image_format: NutFormat,
        pub image_data: Vec<u8>,
    }

    #[pymethods]
    impl ImageTexture {
        #[new]
        fn new(
            hash_id: u32,
            width: u32,
            height: u32,
            mipmap_count: u32,
            layers: u32,
            image_format: NutFormat,
            image_data: Vec<u8>,
        ) -> Self {
            Self {
                hash_id,
                width,
                height,
                mipmap_count,
                layers,
                image_format,
                image_data,
            }
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::VbnSkeleton)]
    pub struct VbnSkeleton {
        pub bones: TypedList<VbnBone>,
    }

    #[pymethods]
    impl VbnSkeleton {
        #[new]
        fn new(bones: TypedList<VbnBone>) -> Self {
            Self { bones }
        }

        pub fn model_space_transforms(&self, py: Python) -> PyResult<Py<PyArray3<f32>>> {
            let skeleton: sm4sh_model::VbnSkeleton = self.clone().map_py(py)?;
            skeleton.model_space_transforms().map_py(py)
        }
    }

    #[pyclass(get_all, set_all)]
    #[derive(Debug, Clone, MapPy)]
    #[map(sm4sh_model::VbnBone)]
    pub struct VbnBone {
        pub name: String,
        pub hash: u32,
        pub parent_bone_index: Option<usize>,
        pub bone_type: BoneType,
        pub translation: [f32; 3],
        pub rotation: [f32; 3],
        pub scale: [f32; 3],
    }

    #[pymethods]
    impl VbnBone {
        #[new]
        fn new(
            name: String,
            hash: u32,
            parent_bone_index: Option<usize>,
            bone_type: BoneType,
            translation: [f32; 3],
            rotation: [f32; 3],
            scale: [f32; 3],
        ) -> Self {
            Self {
                name,
                hash,
                parent_bone_index,
                bone_type,
                translation,
                rotation,
                scale,
            }
        }
    }

    #[pymodule_export]
    use super::BoneFlags;

    #[pymodule_export]
    use super::PrimitiveType;

    #[pymodule_export]
    use super::SrcFactor;

    #[pymodule_export]
    use super::DstFactor;

    #[pymodule_export]
    use super::AlphaFunc;

    #[pymodule_export]
    use super::CullMode;

    #[pymodule_export]
    use super::MapMode;

    #[pymodule_export]
    use super::MinFilter;

    #[pymodule_export]
    use super::MagFilter;

    #[pymodule_export]
    use super::MipDetail;

    #[pymodule_export]
    use super::WrapMode;

    #[pymodule_export]
    use super::NutFormat;

    #[pymodule_export]
    use super::BoneType;

    #[pymodule]
    mod animation {
        use map_py::{MapPy, TypedList};
        use numpy::{PyArray2, PyArray3};
        use pyo3::{
            prelude::*,
            types::{PyDict, PyList},
        };

        use super::VbnSkeleton;

        #[pyfunction]
        fn load_animations(py: Python, path: &str) -> PyResult<TypedList<(String, Animation)>> {
            // TODO: Create an error type.
            let animations = sm4sh_model::animation::load_animations(path).unwrap();
            // TODO: Derive mappy conversions for tuples?
            let elements = animations
                .into_iter()
                .map(|(name, animation)| {
                    let a: Animation = animation.map_py(py)?;
                    Ok((name, a))
                })
                .collect::<PyResult<Vec<_>>>()?;

            // TODO: Add typedlist constructor to mappy
            let mut list = TypedList::empty(py);
            list.list = PyList::new(py, elements)?.into();
            Ok(list)
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::animation::Animation)]
        pub struct Animation {
            pub nodes: TypedList<AnimationNode>,
            pub frame_count: usize,
        }

        #[pymethods]
        impl Animation {
            #[new]
            fn new(nodes: TypedList<AnimationNode>, frame_count: usize) -> Self {
                Self { nodes, frame_count }
            }

            pub fn skinning_transforms(
                &self,
                py: Python,
                skeleton: VbnSkeleton,
                frame: f32,
            ) -> PyResult<Py<PyArray3<f32>>> {
                let animation: sm4sh_model::animation::Animation = self.clone().map_py(py)?;
                let skeleton = skeleton.map_py(py)?;
                let transforms = animation.skinning_transforms(&skeleton, frame);
                transforms.map_py(py)
            }

            pub fn model_space_transforms(
                &self,
                py: Python,
                skeleton: VbnSkeleton,
                frame: f32,
            ) -> PyResult<Py<PyArray3<f32>>> {
                let animation: sm4sh_model::animation::Animation = self.clone().map_py(py)?;
                let skeleton = skeleton.map_py(py)?;
                let transforms = animation.model_space_transforms(&skeleton, frame);
                transforms.map_py(py)
            }

            pub fn local_space_transforms(
                &self,
                py: Python,
                skeleton: VbnSkeleton,
                frame: f32,
            ) -> PyResult<Py<PyArray3<f32>>> {
                let animation: sm4sh_model::animation::Animation = self.clone().map_py(py)?;
                let skeleton = skeleton.map_py(py)?;
                let transforms = animation.local_space_transforms(&skeleton, frame);
                transforms.map_py(py)
            }

            pub fn fcurves(
                &self,
                py: Python,
                skeleton: VbnSkeleton,
                use_blender_coordinates: bool,
            ) -> PyResult<FCurves> {
                let animation: sm4sh_model::animation::Animation = self.clone().map_py(py)?;
                let skeleton = skeleton.map_py(py)?;
                let fcurves = animation.fcurves(&skeleton, use_blender_coordinates);

                let translation = PyDict::new(py);
                for (k, v) in &fcurves.translation {
                    let v: Py<PyArray2<f32>> = v.clone().map_py(py)?;
                    translation.set_item(k, v.into_pyobject(py)?)?;
                }

                let rotation = PyDict::new(py);
                for (k, v) in &fcurves.rotation {
                    let v: Py<PyArray2<f32>> = v.clone().map_py(py)?;
                    rotation.set_item(k, v.into_pyobject(py)?)?;
                }

                let scale = PyDict::new(py);
                for (k, v) in &fcurves.scale {
                    let v: Py<PyArray2<f32>> = v.clone().map_py(py)?;
                    scale.set_item(k, v.into_pyobject(py)?)?;
                }

                Ok(FCurves {
                    translation: translation.into(),
                    rotation: rotation.into(),
                    scale: scale.into(),
                })
            }
        }

        #[pyclass]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::animation::AnimationNode)]
        pub struct AnimationNode(sm4sh_model::animation::AnimationNode);

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone)]
        pub struct FCurves {
            pub translation: Py<PyDict>,
            pub rotation: Py<PyDict>,
            pub scale: Py<PyDict>,
        }
    }

    #[pymodule]
    mod database {
        use map_py::{MapPy, TypedDict, TypedList};
        use pyo3::prelude::*;

        #[pyclass]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::ShaderDatabase)]
        pub struct ShaderDatabase(sm4sh_model::database::ShaderDatabase);

        #[pymethods]
        impl ShaderDatabase {
            #[staticmethod]
            pub fn from_file(path: &str) -> PyResult<Self> {
                // TODO: Avoid unwrap.
                Ok(Self(
                    sm4sh_model::database::ShaderDatabase::from_file(path).unwrap(),
                ))
            }

            pub fn get_shader(
                &self,
                py: Python,
                shader_id: u32,
            ) -> PyResult<Option<ShaderProgram>> {
                self.0
                    .get_shader(shader_id)
                    .map(|s| s.clone().map_py(py))
                    .transpose()
            }
        }

        #[pyclass(get_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::ShaderProgram)]
        pub struct ShaderProgram {
            pub output_dependencies: TypedDict<String, usize>,
            pub exprs: TypedList<OutputExpr>,
            pub attributes: TypedList<String>,
            pub samplers: TypedList<String>,
            pub parameters: TypedList<String>,
        }

        #[pymethods]
        impl ShaderProgram {
            pub fn parameter_value(
                &self,
                py: Python,
                parameter: Parameter,
            ) -> PyResult<Option<f32>> {
                let program: sm4sh_model::database::ShaderProgram = self.clone().map_py(py)?;
                let parameter: sm4sh_model::database::Parameter = parameter.map_py(py)?;
                Ok(program.parameter_value(&parameter))
            }
        }

        #[pyclass]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::OutputExpr<sm4sh_model::database::Operation>)]
        pub struct OutputExpr(sm4sh_model::database::OutputExpr<sm4sh_model::database::Operation>);

        #[pymodule_export]
        use super::super::Operation;

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone)]
        pub struct OutputExprFunc {
            pub op: Operation,
            pub args: Vec<usize>,
        }

        #[pyclass]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::Value)]
        pub struct Value(sm4sh_model::database::Value);

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::Parameter)]
        pub struct Parameter {
            pub name: String,
            pub field: String,
            pub index: Option<usize>,
            pub channel: Option<char>,
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::Texture)]
        pub struct Texture {
            pub name: String,
            pub channel: Option<char>,
            pub texcoords: Vec<usize>,
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::database::Attribute)]
        pub struct Attribute {
            pub name: String,
            pub channel: Option<char>,
        }

        #[pymethods]
        impl OutputExpr {
            pub fn value(&self) -> Option<Value> {
                match &self.0 {
                    sm4sh_model::database::OutputExpr::Value(v) => Some(Value(v.clone())),
                    _ => None,
                }
            }

            pub fn func(&self) -> Option<OutputExprFunc> {
                match &self.0 {
                    sm4sh_model::database::OutputExpr::Func { op, args } => Some(OutputExprFunc {
                        op: (*op).into(),
                        args: args.clone(),
                    }),
                    _ => None,
                }
            }
        }

        #[pymethods]
        impl Value {
            pub fn int(&self) -> Option<i32> {
                match &self.0 {
                    sm4sh_model::database::Value::Int(i) => Some(*i),
                    _ => None,
                }
            }

            pub fn float(&self) -> Option<f32> {
                match &self.0 {
                    sm4sh_model::database::Value::Float(c) => Some(c.0),
                    _ => None,
                }
            }

            pub fn parameter(&self, py: Python) -> PyResult<Option<Parameter>> {
                match &self.0 {
                    sm4sh_model::database::Value::Parameter(b) => b.clone().map_py(py).map(Some),
                    _ => Ok(None),
                }
            }

            pub fn texture(&self, py: Python) -> PyResult<Option<Texture>> {
                match &self.0 {
                    sm4sh_model::database::Value::Texture(t) => t.clone().map_py(py).map(Some),
                    _ => Ok(None),
                }
            }

            pub fn attribute(&self, py: Python) -> PyResult<Option<Attribute>> {
                match &self.0 {
                    sm4sh_model::database::Value::Attribute(a) => a.clone().map_py(py).map(Some),
                    _ => Ok(None),
                }
            }
        }
    }

    #[pymodule]
    mod skinning {
        use map_py::{MapPy, TypedList};
        use numpy::PyArray2;
        use pyo3::prelude::*;

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::skinning::Influence)]
        pub struct Influence {
            pub bone_name: String,
            pub weights: TypedList<VertexWeight>,
        }

        #[pymethods]
        impl Influence {
            #[new]
            fn new(bone_name: String, weights: TypedList<VertexWeight>) -> Self {
                Self { bone_name, weights }
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::skinning::VertexWeight)]
        pub struct VertexWeight {
            pub vertex_index: u32,
            pub weight: f32,
        }

        #[pymethods]
        impl VertexWeight {
            #[new]
            fn new(vertex_index: u32, weight: f32) -> Self {
                Self {
                    vertex_index,
                    weight,
                }
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::skinning::SkinWeights)]
        pub struct SkinWeights {
            pub bone_indices: Py<PyArray2<u32>>,
            pub bone_weights: Py<PyArray2<f32>>,
        }

        #[pymethods]
        impl SkinWeights {
            #[new]
            fn new(bone_indices: Py<PyArray2<u32>>, bone_weights: Py<PyArray2<f32>>) -> Self {
                Self {
                    bone_indices,
                    bone_weights,
                }
            }

            fn to_influences(
                &self,
                py: Python,
                bone_names: TypedList<String>,
            ) -> PyResult<TypedList<Influence>> {
                let weights: sm4sh_model::skinning::SkinWeights = self.clone().map_py(py)?;
                let bone_names: Vec<_> = bone_names.map_py(py)?;
                weights.to_influences(&bone_names).map_py(py)
            }

            #[staticmethod]
            fn from_influences(
                py: Python,
                influences: TypedList<Influence>,
                vertex_count: usize,
                bone_names: TypedList<String>,
            ) -> PyResult<Self> {
                let influences: Vec<_> = influences.map_py(py)?;
                let bone_names: Vec<String> = bone_names.map_py(py)?;
                let weights = sm4sh_model::skinning::SkinWeights::from_influences(
                    &influences,
                    vertex_count,
                    &bone_names,
                );
                weights.map_py(py)
            }
        }
    }

    #[pymodule]
    mod vertex {
        use half::f16;
        use map_py::{
            MapPy, TypedList,
            helpers::{from_option_py, into_option_py},
        };
        use numpy::PyArray2;
        use pyo3::prelude::*;

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::vertex::Vertices)]
        pub struct Vertices {
            pub positions: Py<PyArray2<f32>>,
            pub normals: Normals,
            #[map(from(into_option_py), into(from_option_py))]
            pub bones: Option<Py<Bones>>,
            #[map(from(into_option_py), into(from_option_py))]
            pub colors: Option<Py<Colors>>,
            pub uvs: Uvs,
        }

        #[pymethods]
        impl Vertices {
            #[new]
            fn new(
                positions: Py<PyArray2<f32>>,
                normals: Normals,
                bones: Option<Py<Bones>>,
                colors: Option<Py<Colors>>,
                uvs: Uvs,
            ) -> Self {
                Self {
                    positions,
                    normals,
                    bones,
                    colors,
                    uvs,
                }
            }
        }

        // TODO: Rework these to be representable using numpy arrays.
        // TODO: Complex enums?
        #[pyclass]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::vertex::Normals)]
        pub struct Normals(sm4sh_model::vertex::Normals);

        #[pymethods]
        impl Normals {
            pub fn normals(&self, py: Python) -> PyResult<Option<Py<PyArray2<f32>>>> {
                self.0.normals().map_py(py)
            }

            #[staticmethod]
            fn from_normals_tangents_bitangents_float16(
                py: Python,
                normals: Py<PyArray2<f32>>,
                tangents: Py<PyArray2<f32>>,
                bitangents: Py<PyArray2<f32>>,
            ) -> PyResult<Self> {
                let normals: Vec<[f32; 4]> = normals.map_py(py)?;
                let tangents: Vec<[f32; 4]> = tangents.map_py(py)?;
                let bitangents: Vec<[f32; 4]> = bitangents.map_py(py)?;

                let items = normals
                    .into_iter()
                    .zip(tangents)
                    .zip(bitangents)
                    .map(|((normal, tangent), bitangent)| {
                        sm4sh_model::vertex::NormalsTangentBitangentFloat16 {
                            normal: normal.map(f16::from_f32),
                            bitangent: bitangent.map(f16::from_f32),
                            tangent: tangent.map(f16::from_f32),
                        }
                    })
                    .collect();

                Ok(Self(
                    sm4sh_model::vertex::Normals::NormalsTangentBitangentFloat16(items),
                ))
            }

            #[staticmethod]
            fn from_normals_tangents_bitangents_float32(
                py: Python,
                normals: Py<PyArray2<f32>>,
                tangents: Py<PyArray2<f32>>,
                bitangents: Py<PyArray2<f32>>,
            ) -> PyResult<Self> {
                let normals: Vec<[f32; 4]> = normals.map_py(py)?;
                let tangents: Vec<[f32; 4]> = tangents.map_py(py)?;
                let bitangents: Vec<[f32; 4]> = bitangents.map_py(py)?;

                let items = normals
                    .into_iter()
                    .zip(tangents)
                    .zip(bitangents)
                    .map(|((normal, tangent), bitangent)| {
                        sm4sh_model::vertex::NormalsTangentBitangentFloat32 {
                            unk1: 1.0,
                            normal,
                            bitangent,
                            tangent,
                        }
                    })
                    .collect();

                Ok(Self(
                    sm4sh_model::vertex::Normals::NormalsTangentBitangentFloat32(items),
                ))
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::vertex::Bones)]
        pub struct Bones {
            pub bone_indices: Py<PyArray2<u32>>,
            pub weights: Py<PyArray2<f32>>,
            pub element_type: BoneElementType,
        }

        #[pymethods]
        impl Bones {
            #[new]
            fn new(
                bone_indices: Py<PyArray2<u32>>,
                weights: Py<PyArray2<f32>>,
                element_type: BoneElementType,
            ) -> Self {
                Self {
                    bone_indices,
                    weights,
                    element_type,
                }
            }
        }

        #[pymodule_export]
        use super::super::BoneElementType;

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::vertex::Colors)]
        pub struct Colors {
            pub colors: Py<PyArray2<f32>>,
            pub element_type: ColorElementType,
        }

        #[pymethods]
        impl Colors {
            #[new]
            fn new(colors: Py<PyArray2<f32>>, element_type: ColorElementType) -> Self {
                Self {
                    colors,
                    element_type,
                }
            }
        }

        #[pymodule_export]
        use super::super::ColorElementType;

        #[pyclass]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::vertex::Uvs)]
        pub struct Uvs(sm4sh_model::vertex::Uvs);

        #[pymethods]
        impl Uvs {
            pub fn uvs(&self, py: Python) -> PyResult<TypedList<Py<PyArray2<f32>>>> {
                self.0.uvs().map_py(py)
            }

            #[staticmethod]
            fn from_uvs_float16(py: Python, uvs: TypedList<Py<PyArray2<f32>>>) -> PyResult<Self> {
                let uvs: Vec<Vec<[f32; 2]>> = uvs.map_py(py)?;
                let uv_layers = uvs
                    .into_iter()
                    .map(|uvs| {
                        uvs.into_iter()
                            .map(|[u, v]| sm4sh_model::vertex::UvFloat16 {
                                u: f16::from_f32(u),
                                v: f16::from_f32(v),
                            })
                            .collect()
                    })
                    .collect();
                Ok(Self(sm4sh_model::vertex::Uvs::Float16(uv_layers)))
            }

            #[staticmethod]
            fn from_uvs_float32(py: Python, uvs: TypedList<Py<PyArray2<f32>>>) -> PyResult<Self> {
                let uvs: Vec<Vec<[f32; 2]>> = uvs.map_py(py)?;
                let uv_layers = uvs
                    .into_iter()
                    .map(|uvs| {
                        uvs.into_iter()
                            .map(|[u, v]| sm4sh_model::vertex::UvFloat32 { u, v })
                            .collect()
                    })
                    .collect();
                Ok(Self(sm4sh_model::vertex::Uvs::Float32(uv_layers)))
            }
        }
    }
}
