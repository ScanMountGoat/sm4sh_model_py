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
    sm4sh_model::nud::BoneFlags,
    Disabled,
    Skinning,
    ParentBone
);

python_enum!(
    PrimitiveType,
    sm4sh_model::nud::PrimitiveType,
    TriangleList,
    TriangleStrip
);

python_enum!(
    SrcFactor,
    sm4sh_model::nud::SrcFactor,
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
    sm4sh_model::nud::DstFactor,
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
    sm4sh_model::nud::AlphaFunc,
    Disabled,
    Never,
    Less,
    Eq,
    Leq,
    Neq,
    Geq,
    Always
);

python_enum!(
    CullMode,
    sm4sh_model::nud::CullMode,
    Disabled,
    Outside,
    Inside,
    Disabled2,
    Inside2,
    Outside2
);

python_enum!(
    MapMode,
    sm4sh_model::nud::MapMode,
    TexCoord,
    EnvCamera,
    Projection,
    EnvLight,
    EnvSpec
);

python_enum!(
    MinFilter,
    sm4sh_model::nud::MinFilter,
    LinearMipmapLinear,
    Nearest,
    Linear,
    NearestMipmapLinear
);

python_enum!(
    MagFilter,
    sm4sh_model::nud::MagFilter,
    Unk0,
    Nearest,
    Linear
);

python_enum!(
    MipDetail,
    sm4sh_model::nud::MipDetail,
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
    sm4sh_model::nud::WrapMode,
    Repeat,
    MirroredRepeat,
    ClampToEdge
);

python_enum!(
    NutFormat,
    sm4sh_model::nud::NutFormat,
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
    sm4sh_model::nud::BoneType,
    Normal,
    Follow,
    Helper,
    Swing
);

// Match the module hierarchy and types of sm4sh_model as closely as possible.
#[pymodule]
mod sm4sh_model_py {
    use super::*;

    #[pymodule]
    mod nud {
        use map_py::helpers::{from_option_py, into_option_py};
        use map_py::{MapPy, TypedList};
        use numpy::{PyArray1, PyArray3};
        use pyo3::prelude::*;

        #[pyfunction]
        fn load_model(py: Python, path: &str) -> PyResult<NudModel> {
            // TODO: Create an error type.
            let model = sm4sh_model::nud::load_model(path).unwrap();
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

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudModel)]
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
                let model: sm4sh_model::nud::NudModel = self.clone().map_py(py)?;
                let nud = model.to_nud().unwrap();
                Ok(Nud(nud))
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudMeshGroup)]
        pub struct NudMeshGroup {
            pub name: String,
            pub meshes: TypedList<NudMesh>,
            pub sort_bias: f32,
            pub bounding_sphere: [f32; 4],
            pub bone_flags: BoneFlags,
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
                bone_flags: BoneFlags,
                parent_bone_index: Option<usize>,
            ) -> Self {
                Self {
                    name,
                    meshes,
                    sort_bias,
                    bounding_sphere,
                    bone_flags,
                    parent_bone_index,
                }
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudMesh)]
        pub struct NudMesh {
            pub vertices: vertex::Vertices,
            pub vertex_indices: Py<PyArray1<u16>>,
            pub unk3: bool,
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
                unk3: bool,
                primitive_type: PrimitiveType,
                material1: Option<NudMaterial>,
                material2: Option<NudMaterial>,
                material3: Option<NudMaterial>,
                material4: Option<NudMaterial>,
            ) -> Self {
                Self {
                    vertices,
                    vertex_indices,
                    unk3,
                    primitive_type,
                    material1,
                    material2,
                    material3,
                    material4,
                }
            }

            pub fn triangle_list_indices(&self, py: Python) -> PyResult<Py<PyArray1<u16>>> {
                let mesh: sm4sh_model::nud::NudMesh = self.clone().map_py(py)?;
                mesh.triangle_list_indices().to_vec().map_py(py)
            }
        }

        #[pymodule]
        mod vertex {
            use map_py::{MapPy, TypedList};
            use numpy::PyArray2;
            use pyo3::prelude::*;

            #[pyclass(get_all, set_all)]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Vertices)]
            pub struct Vertices {
                pub positions: Py<PyArray2<f32>>,
                pub normals: Normals,
                pub bones: Bones,
                pub colors: Colors,
                pub uvs: TypedList<Uvs>,
            }

            #[pymethods]
            impl Vertices {
                #[new]
                fn new(
                    positions: Py<PyArray2<f32>>,
                    normals: Normals,
                    bones: Bones,
                    colors: Colors,
                    uvs: TypedList<Uvs>,
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
            #[map(sm4sh_model::nud::vertex::Normals)]
            pub struct Normals(sm4sh_model::nud::vertex::Normals);

            #[pymethods]
            impl Normals {
                pub fn normals(&self, py: Python) -> PyResult<Option<Py<PyArray2<f32>>>> {
                    self.0.normals().map_py(py)
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
                        .zip(tangents.into_iter())
                        .zip(bitangents.into_iter())
                        .map(|((normal, tangent), bitangent)| {
                            sm4sh_model::nud::vertex::NormalsTangentBitangentFloat32 {
                                unk1: 1.0,
                                normal,
                                bitangent,
                                tangent,
                            }
                        })
                        .collect();

                    Ok(Self(
                        sm4sh_model::nud::vertex::Normals::NormalsTangentBitangentFloat32(items),
                    ))
                }
            }

            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Bones)]
            pub struct Bones(sm4sh_model::nud::vertex::Bones);

            #[pymethods]
            impl Bones {
                pub fn bone_indices_weights(
                    &self,
                    py: Python,
                ) -> PyResult<Option<(Py<PyArray2<u32>>, Py<PyArray2<f32>>)>> {
                    self.0
                        .bone_indices_weights()
                        .map(|(indices, weights)| Ok((indices.map_py(py)?, weights.map_py(py)?)))
                        .transpose()
                }

                #[staticmethod]
                fn from_bone_indices_weights_float32(
                    py: Python,
                    indices: Py<PyArray2<u32>>,
                    weights: Py<PyArray2<f32>>,
                ) -> PyResult<Self> {
                    let indices: Vec<[u32; 4]> = indices.map_py(py)?;
                    let weights: Vec<[f32; 4]> = weights.map_py(py)?;

                    let items = indices
                        .into_iter()
                        .zip(weights.into_iter())
                        .map(|(bone_indices, bone_weights)| {
                            sm4sh_model::nud::vertex::BonesFloat32 {
                                bone_indices,
                                bone_weights,
                            }
                        })
                        .collect();

                    Ok(Self(sm4sh_model::nud::vertex::Bones::Float32(items)))
                }
            }

            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Colors)]
            pub struct Colors(sm4sh_model::nud::vertex::Colors);

            #[pymethods]
            impl Colors {
                pub fn colors(&self, py: Python) -> PyResult<Option<Py<PyArray2<f32>>>> {
                    self.0.colors().map_py(py)
                }

                #[staticmethod]
                fn from_colors_byte(py: Python, colors: Py<PyArray2<u8>>) -> PyResult<Self> {
                    let colors: Vec<[u8; 4]> = colors.map_py(py)?;
                    let items = colors
                        .into_iter()
                        .map(|rgba| sm4sh_model::nud::vertex::ColorByte { rgba })
                        .collect();
                    Ok(Self(sm4sh_model::nud::vertex::Colors::Byte(items)))
                }
            }

            #[pyclass]
            #[derive(Debug, Clone, MapPy)]
            #[map(sm4sh_model::nud::vertex::Uvs)]
            pub struct Uvs(sm4sh_model::nud::vertex::Uvs);

            #[pymethods]
            impl Uvs {
                pub fn uvs(&self, py: Python) -> PyResult<Py<PyArray2<f32>>> {
                    self.0.uvs().map_py(py)
                }

                #[staticmethod]
                fn from_uvs_float32(py: Python, uvs: Py<PyArray2<f32>>) -> PyResult<Self> {
                    let uvs: Vec<[f32; 2]> = uvs.map_py(py)?;
                    let items = uvs
                        .into_iter()
                        .map(|[u, v]| sm4sh_model::nud::vertex::UvsFloat32 { u, v })
                        .collect();
                    Ok(Self(sm4sh_model::nud::vertex::Uvs::Float32(items)))
                }
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudMaterial)]
        pub struct NudMaterial {
            // TODO: Should this recreate flags or store them directly?
            #[map(from(map_py::helpers::into), into(map_py::helpers::try_into))]
            pub flags: u32,
            pub src_factor: SrcFactor,
            pub dst_factor: DstFactor,
            pub alpha_func: AlphaFunc,
            pub cull_mode: CullMode,
            pub textures: TypedList<NudTexture>,
            pub properties: TypedList<NudProperty>,
        }

        #[pymethods]
        impl NudMaterial {
            #[new]
            fn new(
                flags: u32,
                src_factor: SrcFactor,
                dst_factor: DstFactor,
                alpha_func: AlphaFunc,
                cull_mode: CullMode,
                textures: TypedList<NudTexture>,
                properties: TypedList<NudProperty>,
            ) -> Self {
                Self {
                    flags,
                    src_factor,
                    dst_factor,
                    alpha_func,
                    cull_mode,
                    textures,
                    properties,
                }
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::NudTexture)]
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
        #[map(sm4sh_model::nud::NudProperty)]
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
        #[map(sm4sh_model::nud::ImageTexture)]
        pub struct ImageTexture {
            pub hash_id: u32,
            pub width: u32,
            pub height: u32,
            pub mipmap_count: u32,
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
                image_format: NutFormat,
                image_data: Vec<u8>,
            ) -> Self {
                Self {
                    hash_id,
                    width,
                    height,
                    mipmap_count,
                    image_format,
                    image_data,
                }
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::VbnSkeleton)]
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
                let skeleton: sm4sh_model::nud::VbnSkeleton = self.clone().map_py(py)?;
                skeleton.model_space_transforms().map_py(py)
            }
        }

        #[pyclass(get_all, set_all)]
        #[derive(Debug, Clone, MapPy)]
        #[map(sm4sh_model::nud::VbnBone)]
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
    }

    #[pymodule]
    mod animation {
        use map_py::{MapPy, TypedList};
        use numpy::{PyArray2, PyArray3};
        use pyo3::{
            prelude::*,
            types::{PyDict, PyList},
        };

        use super::nud::VbnSkeleton;

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
}
